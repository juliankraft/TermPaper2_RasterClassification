import xarray as xr
import numpy as np
from os import PathLike
from pathlib import Path
import lightning as L

from torch.utils.data import Dataset, DataLoader
from src.augmentors import AugmentorChain
from typing import Any, TypedDict


class ReturnType(TypedDict):
    x: np.ndarray
    y: np.ndarray
    central_coord: dict[str, int]
    patch_coord: dict[str, list[int]]


class RSData(Dataset):
    def __init__(
            self,
            ds_path: str | PathLike,
            mask_area_ids: list[int] | int,
            cutout_size: int = 41,
            output_patch_size: int = 5,
            feature_stat_means: xr.DataArray | None = None,
            feature_stat_stds: xr.DataArray | None = None,
            is_pred: bool = False,
            augmentor_chain: AugmentorChain | None = None):

        super().__init__()

        self.ds = xr.open_zarr(ds_path)

        if cutout_size % 2 == 0:
            raise ValueError('`cutout_size` must be an odd integer.')

        if output_patch_size % 2 == 0:
            raise ValueError('`output_patch_size` must be an odd integer.')

        self.mask_values = self.get_mask_values(mask_area_ids)

        self.cutout_size = cutout_size
        self.offset = int(self.cutout_size // 2)
        self.output_patch_size = output_patch_size
        self.output_offset = self.output_patch_size // 2

        mask = self.ds.mask.isin(self.mask_values).compute()

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

        if (feature_stat_means is None) != (feature_stat_stds is None):
            raise ValueError(
                'either pass both of `feature_stat_means` and `feature_stat_stds` or none.'
            )

        if feature_stat_means is None:
            feature_stat_means = self.ds.rs.where(self.mask).mean(('x', 'y')).compute()
            feature_stat_stds = self.ds.rs.where(self.mask).std(('x', 'y')).compute()

        self.feature_stat_means = feature_stat_means
        self.feature_stat_stds = feature_stat_stds

        if augmentor_chain is None:
            self.augmentor_chain = AugmentorChain(random_seed=0, augmentors=[])
        else:
            self.augmentor_chain = augmentor_chain

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
        self.train_area_ids = train_area_ids
        self.valid_area_ids = valid_area_ids
        self.test_area_ids = test_area_ids
        self.cutout_size = cutout_size
        self.output_patch_size = output_patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentor_chain = augmentor_chain

        train_data = self.get_dataset(mode='init')
        self.feature_stat_means = train_data.feature_stat_means
        self.feature_stat_stds = train_data.feature_stat_stds

        self.dataloader_args: dict[str, Any] = {
            'num_workers': self.num_workers
        }

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

        dataset = RSData(
            ds_path=self.ds_path,
            mask_area_ids=mask_area_ids,
            cutout_size=self.cutout_size,
            output_patch_size=self.output_patch_size,
            feature_stat_means=None if mode == 'init' else self.feature_stat_means,
            feature_stat_stds=None if mode == 'init' else self.feature_stat_stds,
            is_pred=True if mode == 'pred' else False,
            augmentor_chain=augmentor_chain
        )

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
