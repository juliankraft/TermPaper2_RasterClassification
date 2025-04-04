import numpy as np
from typing import Tuple


class BaseAugmentor(object):
    """Base augmentor class.

    - Meant to be subclassed.
    - Method `augment` must be overridden in subclass.
    """
    def set_seed(self, random_seed) -> None:
        self.RS = np.random.RandomState(seed=random_seed)

    def augment(self, cutout: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    def augment(self, cutout: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            labels = np.flip(labels, axis=flip_axes)

        return cutout, labels

    def __repr__(self) -> str:
        return 'FlipAugmentor()'


class RotateAugmentor(BaseAugmentor):
    """Random rotate spatial dimensions."""
    def augment(self, cutout: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment a 3D array using random rotation flipping.

        Augments:
            cutout: the input 3D array, shape (x, y, channels).

        Returns:
            Augmented array with same shape as cutout.

        """
        num_rotate = self.random_state.randint(0, 4)

        cutout = np.rot90(cutout, k=num_rotate, axes=(0, 1))
        labels = np.rot90(labels, k=num_rotate, axes=(0, 1))

        return cutout, labels

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

    def augment(self, cutout: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment a 3D array using pixel-level noise.

        Augments:
            cutout: the input 3D array, shape (x, y, channels).

        Returns:
            Augmented array with same shape as cutout.

        """
        random_noise = self.random_state.randn(*cutout.shape) * self.scale

        return cutout + random_noise, labels

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

    def augment(self, cutout: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment a 3D array using channel-level noise.

        Augments:
            cutout: the input 3D array, shape (x, y, channels).

        Returns:
            Augmented array with same shape as cutout.

        """
        random_noise = self.random_state.randn(cutout.shape[2]) * self.scale

        return cutout + random_noise[np.newaxis, np.newaxis, ...], labels

    def __repr__(self) -> str:
        return f'ChannelNoiseAugmentor(scale={self.scale})'


class AugmentorChain(object):

    AUGMENTORS = [FlipAugmentor, RotateAugmentor, PixelNoiseAugmentor, ChannelNoiseAugmentor]

    def __init__(self, random_seed: int, augmentors: list[BaseAugmentor] | None) -> None:
        self.random_seed = random_seed
        self.augmentors = augmentors

        if self.augmentors is not None:
            for augmentor in self.augmentors:
                augmentor.set_random_state(random_seed)

    def augment(self, cutout: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if self.augmentors is None:
            return cutout, labels

        for augmentor in self.augmentors:
            cutout, labels = augmentor.augment(cutout, labels)

        return cutout, labels

    def __repr__(self) -> str:
        if self.augmentors is None:
            return 'AugmentorChain(augmentors=None)'

        autmentors_repr = [str(augmentor) for augmentor in self.augmentors]
        return f'AugmentorChain(random_seed={self.random_seed}, augmentors=[{", ".join(autmentors_repr)}])'

    @classmethod
    def from_args(
            cls,
            *augmentors,
            random_seed,
            **augmentor_kwargs) -> 'AugmentorChain':

        existing_augmentors = {augmentor.__name__: augmentor for augmentor in cls.AUGMENTORS}

        augmentor_list: list[BaseAugmentor] = []

        for augmentor in augmentors:
            augmentor_kwargs.update({augmentor: {}})

        for augmentor_name, augmentor_args in augmentor_kwargs.items():
            if augmentor_name not in existing_augmentors:
                raise ValueError(
                    f'augmentor \'{augmentor_name}\' does not exist, use one of {list(existing_augmentors.keys())}.'
                )
            else:
                augmentor_class = existing_augmentors[augmentor_name]

                try:
                    augmentor = augmentor_class(**augmentor_args)

                except Exception as e:
                    raise RuntimeError(
                        f'Error \'{e}\' occured while initializing \'{augmentor_class.__name__}\' with args '
                        f'{augmentor_args}.'
                    )

                augmentor_list.append(augmentor)

        return cls(random_seed=random_seed, augmentors=augmentor_list)
