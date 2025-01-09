import xarray as xr
import numpy as np
import torch
from os import PathLike
from pathlib import Path
import lightning as L
import dask
from torch.utils.data import Dataset, DataLoader
from src.augmentors import AugmentorChain
from src.parallel_stats import par_stats, par_class_weights
from typing import Any, TypedDict

dask.config.set(scheduler='synchronous')


class ReturnType(TypedDict):
    x: np.ndarray
    y: np.ndarray
    central_coord: dict[str, int]
    patch_coord: dict[str, list[int]]


class RSData(Dataset):
    def __init__(
            self,
            ds_path: str | PathLike,
            num_workers: int,
            num_classes: int,
            mask_area_ids: list[int] | int,
            cutout_size: int = 41,
            output_patch_size: int = 5,
            feature_stat_means: xr.DataArray | None = None,
            feature_stat_stds: xr.DataArray | None = None,
            class_weights: torch.Tensor | None = None,
            is_pred: bool = False,
            augmentor_chain: AugmentorChain | None = None):

        super().__init__()

        self.ds = xr.open_zarr(ds_path)
        self.num_workers = num_workers
        self.num_classes = num_classes

        if cutout_size % 2 == 0:
            raise ValueError('`cutout_size` must be an odd integer.')

        if output_patch_size % 2 == 0:
            raise ValueError('`output_patch_size` must be an odd integer.')

        print('computing mask values', flush=True) # Debugging
        self.mask_values = self.get_mask_values(mask_area_ids)
        print('done computing mask values', flush=True) # Debugging

        self.cutout_size = cutout_size
        self.offset = int(self.cutout_size // 2)
        self.output_patch_size = output_patch_size
        self.output_offset = self.output_patch_size // 2

        # Select valid pixels.
        mask = self.ds.mask.isin(self.mask_values).compute()
        # Exclude pixels with no label.
        if not is_pred:
            mask *= (self.ds.label != 255).values

        # Cut off borders from mask by setting them to False.
        mask[{'x': slice(None, self.offset)}] = False
        mask[{'x': slice(-self.offset, None)}] = False
        mask[{'y': slice(None, self.offset)}] = False
        mask[{'y': slice(-self.offset, None)}] = False

        self.mask = mask.compute()

        if is_pred:
            agg_mask = self.mask.coarsen(x=output_patch_size, y=output_patch_size, boundary='pad').any().compute()
        else:
            agg_mask = self.mask.coarsen(x=output_patch_size, y=output_patch_size, boundary='pad').all().compute()

        self.agg_mask = agg_mask

        self.coords = np.argwhere(self.agg_mask.values)

        if len(set([feature_stat_means is None, feature_stat_stds is None, class_weights is None])) != 1:
            raise ValueError(
                'either pass all of `feature_stat_means`, `feature_stat_stds`, and `class_weights` or none.'
            )

        print('computing feature stats', flush=True) # Debugging
        # if feature_stat_means is None:
        #     feature_stat_means = self.ds.rs.where(self.mask).mean(('x', 'y')).compute()
        #     feature_stat_stds = self.ds.rs.where(self.mask).std(('x', 'y')).compute()
        #     class_weights = self.calculating_class_weights(self.ds.where(self.mask), self.num_classes)

        if feature_stat_means is None:
            stats = par_stats(
                            path=ds_path,
                            variables=['rs'],
                            mask=self.mask,
                            num_processes=self.num_workers
                            )
            feature_stat_means = stats['rs']['mean'].astype('float32')
            feature_stat_stds = stats['rs']['std'].astype('float32')
            class_weights = par_class_weights(
                                            path=ds_path,
                                            variable='label',
                                            mask=self.mask,
                                            num_classes=self.num_classes,
                                            num_processes=self.num_workers
                                            )

        self.feature_stat_means = feature_stat_means
        self.feature_stat_stds = feature_stat_stds
        self.class_weights = class_weights

        print('done computing feature stats', flush=True) # Debugging

        if augmentor_chain is None:
            self.augmentor_chain = AugmentorChain(random_seed=0, augmentors=[])
        else:
            self.augmentor_chain = augmentor_chain

    # def calculating_class_weights(self, ds: xr.Dataset, num_classes: int) -> torch.Tensor:
    #     label_count = ds['label'].groupby(ds['label']).count().compute()
    #     rev_weights = label_count / label_count.sum()
    #     rev_weights_reindex = rev_weights.reindex({rev_weights.dims[0]: list(range(0, num_classes))}, fill_value=0)
    #     weights = 1 - rev_weights_reindex
    #     weights_tensor = torch.tensor(weights.values, dtype=torch.float32)

    #     return weights_tensor

    def get_mask_values(self, mask_area: list[int] | int) -> np.ndarray:
        mask_area_ = np.array([mask_area] if isinstance(mask_area, int) else mask_area)

        if any(mask_area_ < 0) or any(mask_area_ > 4):
            raise ValueError('`mask_values` must be in range [0, ..., 4]')

        mask_values = np.argwhere(np.isin((np.arange(1, 13) - 1) % 4, mask_area_ - 1)).flatten() + 1

        if any(mask_area_ == 0):
            mask_values = np.concatenate((mask_values, np.zeros(1, dtype=int)))

        return mask_values

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> ReturnType:
        agg_x_i, agg_y_i = self.coords[index]

        x_i = agg_x_i * self.output_patch_size + self.output_offset
        y_i = agg_y_i * self.output_patch_size + self.output_offset

        # selecting the cutout.
        cutout = self.ds.rs.isel(
            x=slice(x_i - self.offset, x_i + self.offset + 1),
            y=slice(y_i - self.offset, y_i + self.offset + 1),
        )

        # Standardize.
        cutout = (cutout - self.feature_stat_means) / self.feature_stat_stds

        # Transpose, make sure x and y are first dimensions.
        cutout = cutout.transpose('x', 'y', ...).values

        # selecting the labels.
        x_block_from: int = x_i - self.output_offset
        x_block_to: int = x_i + self.output_offset + 1
        y_block_from: int = y_i - self.output_offset
        y_block_to: int = y_i + self.output_offset + 1
        label_sel = self.ds.label.isel(
            x=slice(x_block_from, x_block_to),
            y=slice(y_block_from, y_block_to),
        )

        label_sel = label_sel.transpose('x', 'y', ...).values

        # Augment.
        cutout, label_sel = self.augmentor_chain.augment(cutout, label_sel)

        # Put channel on first dimension, from (x, y, c) to (c, x, y).
        cutout = cutout.transpose(2, 0, 1)

        return {
            'x': cutout.astype('float32'),
            'y': label_sel.astype('int'),
            'central_coord': {'xi': x_i, 'yi': y_i},
            'patch_coord': {'xi': [x_block_from, x_block_to], 'yi': [y_block_from, y_block_to]},
        }


class RSDataModule(L.LightningDataModule):
    def __init__(
            self,
            ds_path: str | PathLike,
            num_classes: int,
            train_area_ids: list[int],
            valid_area_ids: list[int],
            test_area_ids: list[int],
            cutout_size: int,
            output_patch_size: int,
            batch_size: int,
            num_workers: int = 10,
            augmentor_chain: AugmentorChain | None = None):

        super().__init__()

        self.ds_path = Path(ds_path)
        self.num_classes = num_classes
        self.train_area_ids = train_area_ids
        self.valid_area_ids = valid_area_ids
        self.test_area_ids = test_area_ids
        self.cutout_size = cutout_size
        self.output_patch_size = output_patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentor_chain = augmentor_chain
        print('computing training data stats', flush=True) # Debugging
        train_data = self.get_dataset(mode='init')
        self.feature_stat_means = train_data.feature_stat_means
        self.feature_stat_stds = train_data.feature_stat_stds
        self.class_weights = train_data.class_weights
        print('done computing stats', flush=True) # Debugging
        self.dataloader_args: dict[str, Any] = {
            'num_workers': self.num_workers,
            # 'persistent_workers': True
        }

    def get_class_weights(self) -> torch.Tensor:
        return self.class_weights

    def get_dataset(self, mode: str) -> RSData:
        if mode in ('train', 'init'):
            mask_area_ids = self.train_area_ids
            augmentor_chain = self.augmentor_chain
        elif mode == 'valid':
            mask_area_ids = self.valid_area_ids
            augmentor_chain = None
        elif mode == 'test':
            mask_area_ids = self.test_area_ids
            augmentor_chain = None
        elif mode == 'predict':
            mask_area_ids = [0, 1, 2, 3, 4]
            augmentor_chain = None
        else:
            raise ValueError(
                f'`mode` must be one of \'init\', \'train\', \'valid\', \'test\', is \'{mode}\'.'
            )

        print(f'creating {mode} dataset', flush=True) # Debugging
        dataset = RSData(
            ds_path=self.ds_path,
            num_workers=self.num_workers,
            num_classes=self.num_classes,
            mask_area_ids=mask_area_ids,
            cutout_size=self.cutout_size,
            output_patch_size=self.output_patch_size,
            feature_stat_means=None if mode == 'init' else self.feature_stat_means,
            feature_stat_stds=None if mode == 'init' else self.feature_stat_stds,
            class_weights=None if mode == 'init' else self.class_weights,
            is_pred=True if mode == 'pred' else False,
            augmentor_chain=augmentor_chain
        )

        print(f'creating {mode} dataset done', flush=True) # Debugging
        return dataset

    def train_dataloader(self) -> DataLoader:
        dataset = self.get_dataset(mode='train')
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **self.dataloader_args
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self.get_dataset(mode='valid')
        return DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            **self.dataloader_args
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self.get_dataset(mode='test')
        return DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            **self.dataloader_args
        )

    def predict_dataloader(self) -> DataLoader:
        dataset = self.get_dataset(mode='predict')
        return DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            **self.dataloader_args
        )
