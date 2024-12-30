import xarray as xr
import numpy as np
from tqdm import tqdm
import dask
from os import PathLike
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision.utils import _log_api_usage_once
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

from typing import Callable, Type
from torch import Tensor


class BaseAugmentor(object):
    """Base augmentator class.

    - Meant to be subclassed.
    - Method `augment` must be overridden in subclas.
    """
    def set_seed(self, random_seed) -> None:
        self.RS = np.random.RandomState(seed=random_seed)

    def augment(self, cutout: np.ndarray) -> np.ndarray:
        """Augment a 3D array.

        Augments:
            cutout: the input 3D array, shape (x, y, channels).

        Returns:
            Augmented array with same shape as cutout.

        """

        raise NotImplementedError(
            'you must override this method in the subclass.'
        )

    def set_random_state(self, random_seed: int) -> None:
        self.random_state = np.random.RandomState(seed=random_seed)

    @property
    def random_state(self) -> np.random.RandomState:
        if not hasattr(self, '_random_state'):
            raise AttributeError('attribute `random_state` is not set. Use `set_random_state`.')
        else:
            return self._random_state

    @random_state.setter
    def random_state(self, random_state: np.random.RandomState) -> None:
        if not isinstance(random_state, np.random.RandomState):
            raise TypeError(
                f'`random_state` must be of type `np.random.RandomState`, is `{type(random_state).__name__}`.'
            )
        self._random_state = random_state


class FlipAugmentor(BaseAugmentor):
    """Random flip x and y dimensions."""
    def augment(self, cutout: np.ndarray) -> np.ndarray:
        """Augment a 3D array using random horizontal and vertical flipping.

        Augments:
            cutout: the input 3D array, shape (x, y, channels).

        Returns:
            Augmented array with same shape as cutout.

        """

        flip_vertical = self.random_state.randint(0, 2)
        flip_horizontal = self.random_state.randint(0, 2)

        flip_axes = []
        if flip_vertical:
            flip_axes.append(0)

        if flip_horizontal:
            flip_axes.append(1)

        if len(flip_axes) > 0:
            cutout = np.flip(cutout, axis=flip_axes)

        return cutout

    def __repr__(self) -> str:
        return 'FlipAugmentor()'


class RotateAugmentor(BaseAugmentor):
    """Random rotate spatial dimensions."""
    def augment(self, cutout: np.ndarray) -> np.ndarray:
        """Augment a 3D array using random rotation flipping.

        Augments:
            cutout: the input 3D array, shape (x, y, channels).

        Returns:
            Augmented array with same shape as cutout.

        """
        num_rotate = self.random_state.randint(0, 4)

        cutout = np.rot90(cutout, k=num_rotate, axes=(0, 1))

        return cutout

    def __repr__(self) -> str:
        return 'RotateAugmentor()'


class PixelNoiseAugmentor(BaseAugmentor):
    """Random noise at pixel level."""
    def __init__(self, scale: float) -> None:
        """Init PixelNoiseAugmentor.

        Args:
            scale: scale of noise.
        """
        self.scale = scale

    def augment(self, cutout: np.ndarray) -> np.ndarray:
        """Augment a 3D array using pixel-level noise.

        Augments:
            cutout: the input 3D array, shape (x, y, channels).

        Returns:
            Augmented array with same shape as cutout.

        """
        random_noise = self.random_state.randn(*cutout.shape) * self.scale

        return cutout + random_noise

    def __repr__(self) -> str:
        return f'PixelNoiseAugmentor(scale={self.scale})'


class ChannelNoiseAugmentor(BaseAugmentor):
    """Random noise at channel level."""
    def __init__(self, scale: float) -> None:
        """Init ChannelNoiseAugmentor.

        Args:
            scale: scale of noise.
        """
        self.scale = scale

    def augment(self, cutout: np.ndarray) -> np.ndarray:
        """Augment a 3D array using channel-level noise.

        Augments:
            cutout: the input 3D array, shape (x, y, channels).

        Returns:
            Augmented array with same shape as cutout.

        """
        random_noise = self.random_state.randn(cutout.shape[2]) * self.scale

        return cutout + random_noise[np.newaxis, np.newaxis, ...]

    def __repr__(self) -> str:
        return f'ChannelNoiseAugmentor(scale={self.scale})'


class AugmentorChain(object):
    def __init__(self, random_seed: int, augmentors: list[BaseAugmentor] | None) -> None:
        self.random_seed = random_seed
        self.augmentors = augmentors

        if self.augmentors is not None:
            for augmentor in self.augmentors:
                augmentor.set_random_state(random_seed)

    def augment(self, cutout: np.ndarray) -> np.ndarray:

        if self.augmentors is None:
            return cutout

        for augmentor in self.augmentors:
            cutout = augmentor.augment(cutout)

        return cutout

    def __repr__(self) -> str:
        if self.augmentors is None:
            return 'AugmentorChain(augmentors=None)'

        autmentors_repr = [str(augmentor) for augmentor in self.augmentors]
        return f'AugmentorChain(random_seed={self.random_seed}, augmentors=[{", ".join(autmentors_repr)}])'


class RSData(Dataset):
    def __init__(
            self,
            ds_path: str | PathLike,
            mask_area_ids: list[int] | int,
            cutout_size: int = 21,
            feature_stat_means: xr.DataArray | None = None,
            feature_stat_stds: xr.DataArray | None = None,
            augmentor_chain: AugmentorChain | None = None):

        super().__init__()

        self.ds = xr.open_zarr(ds_path)

        if cutout_size % 2 == 0:
            raise ValueError('`cutout_size` must be an odd integer.')

        self.mask_values = self.get_mask_values(mask_area_ids)

        self.cutout_size = cutout_size
        self.offset = int(self.cutout_size // 2)

        mask = self.ds.mask.isin(self.mask_values).compute()

        # Cut off borders from mask ny setting them to False.
        mask[{'x': slice(None, self.offset)}] = False
        mask[{'x': slice(-self.offset, None)}] = False
        mask[{'y': slice(None, self.offset)}] = False
        mask[{'y': slice(-self.offset, None)}] = False

        self.mask = mask

        self.coords = np.argwhere(self.mask.values)

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

    def __getitem__(self, index: int):
        x_i, y_i = self.coords[index]

        cutout = self.ds.rs.isel(
            x=slice(x_i - self.offset, x_i + self.offset + 1),
            y=slice(y_i - self.offset, y_i + self.offset + 1),
        )

        # Standardize.
        cutout = (cutout - self.feature_stat_means) / self.feature_stat_stds

        # Transpose, make sure x and y are first dimensions.
        cutout = cutout.transpose('x', 'y', ...).values

        # Augment.
        cutout = self.augmentor_chain.augment(cutout)

        # Put channel on first dimension, from (x, y, c) to (c, x, y).
        cutout = cutout.transpose(2, 0, 1)

        label_sel = self.ds.label.isel(
            x=x_i,
            y=y_i,
        ).values

        return cutout.astype('float32'), label_sel.astype('int')
