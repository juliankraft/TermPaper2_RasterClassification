
import os
import shutil
import yaml
from pathlib import Path
from argparse import ArgumentParser, Namespace
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from typing import cast

from src.dataloader import RSDataModule
from src.augmentors import AugmentorChain
from src.model import LightningResNet, count_trainable_parameters
from src.utils import PredictionWriter


def make_dir_from_args(base_path: Path | str, args: Namespace) -> Path:
    base_path = Path(base_path)
    specific_path = 'augment' if args.use_data_augmentation else 'noaugment'
    path = base_path / specific_path

    if path.exists():
        if args.overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                'Base directory exists, but overwrite is False. '
                f'Use `-o` or `--overwrite` to delete existing runs in `{path}`.'
            )

    os.makedirs(path)

    return path


if __name__ == '__main__':

    # Initialize the argument parser
    parser = ArgumentParser()

    # Add arguments to the parser
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device to run the model on (cuda or cpu).')
    parser.add_argument(
        '--num_classes', type=int, default=10,
        help='Number of classes in the dataset.')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size for training.')
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='The number of dataloader workers.')
    parser.add_argument(
        '--cutout_size', type=int, default=51,
        help='Cutout size in pixels of the input feature image. Must be an odd number.')
    parser.add_argument(
        '--output_patch_size', type=int, default=5,
        help='Cutout size of the prediction patch. Must be an odd number.')
    parser.add_argument(
        '--learning_rate', type=float, default=0.01,
        help='Learning rate for the optimizer.')
    parser.add_argument(
        '--weight_decay', type=float, default=0.0,
        help='Weight decay for the optimizer.')
    parser.add_argument(
        '--use_data_augmentation', action='store_true',
        help='Flag to enable data augmentation.')
    parser.add_argument(
        '--patience', type=int, default=5,
        help='The early stopping patience; how many epochs the validation loss can stagnate before stopping.')
    parser.add_argument(
        '-o', '--overwrite', action='store_true',
        help='Overwrite existing runs. Default is False and an error is thrown if directory exists.')
    parser.add_argument(
        '--dev_run', action='store_true',
        help='Runs 3 epochs with one batch of training and validation each.')
    parser.add_argument(
        '--sample_data', action='store_true',
        help='Use only the small sample data set.')
    parser.add_argument(
        '--use_class_weights', action='store_true',
        help='Apply class weights to the loss function.')

    # Parse the arguments
    args = parser.parse_args()

    if args.use_data_augmentation:
        ac = AugmentorChain.from_args(
            'FlipAugmentor',
            'RotateAugmentor',
            random_seed=1,
        )
    else:
        ac = None

    if args.sample_data:
        ds_path = '../data/sample_combined.zarr'
    else:
        ds_path = '../data/combined.zarr'

    datamodule = RSDataModule(
        ds_path=ds_path,
        num_classes=args.num_classes,
        train_area_ids=[1, 2],
        valid_area_ids=[3],
        test_area_ids=[4],
        cutout_size=args.cutout_size,
        output_patch_size=args.output_patch_size,
        augmentor_chain=ac,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model
    model = LightningResNet(
        num_classes=args.num_classes,
        output_patch_size=args.output_patch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_class_weights=args.use_class_weights,
        feature_weights=datamodule.get_feature_weights()
    )

    log_dir = make_dir_from_args(base_path='../runs', args=args)

    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name='',
        version='',
    )

    csv_logger = CSVLogger(
        save_dir=log_dir,
        name='',
        version='',
    )

    # For dev run, limit batches and max_epochs.
    dev_run_args = {
        'limit_train_batches': 1 if args.dev_run else 0.2,  # (float = fraction, int = num_batches)
        'limit_val_batches': 1 if args.dev_run else 1.0,
        'limit_test_batches': 1 if args.dev_run else 1.0,
        'limit_predict_batches': 200 if args.dev_run else 1.0,
        'max_epochs': 3 if args.dev_run else 300,
        # 'log_every_n_steps': 1 if args.dev_run else 10,
    }

    # Save all configurations to a yaml file.
    num_trainable_params = count_trainable_parameters(model)

    parser_args_dict = vars(args)

    config = {
        **parser_args_dict,
        **dev_run_args,
        'num_trainable_params': num_trainable_params,
    }

    if args.use_class_weights:
        class_weights = datamodule.get_feature_weights().tolist()
        config['class_weights'] = class_weights

    with open(log_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Trainer
    trainer = L.Trainer(
        default_root_dir=log_dir,
        accelerator=args.device,
        callbacks=[
            ModelCheckpoint(
                filename='best',
                save_last=True,
                save_top_k=1,
                every_n_epochs=1
            ),
            EarlyStopping(
                monitor='valid_loss',
                patience=args.patience),
            PredictionWriter(output_dir=log_dir),
        ],
        logger=[tb_logger, csv_logger],
        log_every_n_steps=1,
        **dev_run_args
    )

    trainer.fit(model=model, datamodule=datamodule)

    best_model_path = cast(str, trainer.checkpoint_callback.best_model_path)  # type: ignore
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=best_model_path)
