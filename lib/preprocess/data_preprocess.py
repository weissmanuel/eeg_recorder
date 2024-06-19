import numpy as np
from numpy import ndarray
from mne import Info
from typing import Union, List, Dict
from enum import Enum
from omegaconf import DictConfig, ListConfig
from lib.utils import config_to_primitive
from abc import ABC, abstractmethod
from typing import Tuple, Union
from collections import deque


class Preprocessors(Enum):
    EEGScaler = 'eeg_scaler'
    EpochWindowSplitter = 'epoch_window_splitter'
    FFT = 'fft'
    SklearnClassifierReshape = 'sklearn_classifier_reshape'


PreprocessResponse = Union[ndarray, Tuple[ndarray, Union[ndarray, None]]]


class Preprocessor(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, data: ndarray, labels: ndarray | None = None, *args, **kwargs) -> PreprocessResponse:
        pass


class EEGScaler(Preprocessor):
    scale: float
    channel_types: List[str]

    def __init__(self, scale: float = 1e-6, channel_types: List[str] = None):
        super().__init__(Preprocessors.EEGScaler.name)
        self.scale = scale
        if channel_types is None:
            self.channel_types = ['eeg']
        else:
            self.channel_types = channel_types

    def __call__(self, data: ndarray, labels: ndarray | None = None, *args, **kwargs) -> ndarray:
        info: Info | None = kwargs.get('info', None)
        assert info is not None, 'info must be provided for EEGScaler'
        ch_idx = [i for i, ch_type in enumerate(info.get_channel_types()) if ch_type in self.channel_types]
        if len(ch_idx) > 0:
            data[ch_idx] = data[ch_idx] * self.scale
        return data


class EpochWindowSplitter(Preprocessor):
    sfreq: float

    window_size_seconds: float
    window_shift_seconds: float
    window_size: int
    window_shift: int

    use_averaging: bool
    average_size: int

    def __init__(self,
                 sfreq: float,
                 window_size_seconds: float,
                 window_shift_seconds: float,
                 use_averaging: bool = False,
                 average_size: int = 5):
        super().__init__(Preprocessors.EpochWindowSplitter.name)
        self.sfreq = sfreq
        self.window_size_seconds = window_size_seconds
        self.window_shift_seconds = window_shift_seconds

        self.window_size = int(window_size_seconds * self.sfreq)
        self.window_shift = int(window_shift_seconds * self.sfreq)

        self.use_averaging = use_averaging
        self.average_size = average_size

    def __call__(self, data: ndarray, labels: ndarray | None = None, *args, **kwargs) -> PreprocessResponse:
        assert labels is not None, 'Labels must be provided for EpochWindowSplitter'
        assert len(data.shape) == 3, 'Data must be 3D (n_epochs, n_channels, n_samples)'

        n_epochs, n_channels, n_samples = data.shape

        windows = []
        window_labels = []

        for i in range(n_epochs):
            queue = deque(maxlen=self.average_size)
            for j in range(0, n_samples - self.window_size, self.window_shift):
                window = data[i, :, j:j + self.window_size]
                if self.use_averaging:
                    queue.append(window)
                    window = np.mean(list(queue), axis=0)
                windows.append(window)
                window_labels.append(labels[i])

        windows = np.array(windows)
        window_labels = np.array(window_labels)

        return windows, window_labels


class FFT(Preprocessor):

    def __init__(self):
        super().__init__(Preprocessors.FFT.name)

    def __call__(self, data: ndarray, *args, **kwargs) -> PreprocessResponse:
        data_freq = np.fft.fft(data, axis=2)
        data_freq = 2 / data_freq.shape[2] * np.abs(data_freq[:, :, :data_freq.shape[2] // 2])
        return data_freq


class SklearnClassifierReshape(Preprocessor):
    def __init__(self):
        super().__init__(Preprocessors.SklearnClassifierReshape.name)

    def __call__(self, data: ndarray, *args, **kwargs) -> PreprocessResponse:
        return data.reshape(data.shape[0], -1), None


def get_preprocessor(name: str, **kwargs) -> Preprocessor:
    match name:
        case Preprocessors.EEGScaler.value:
            return EEGScaler(**kwargs)
        case Preprocessors.EpochWindowSplitter.value:
            return EpochWindowSplitter(**kwargs)
        case Preprocessors.FFT.value:
            return FFT()
        case _:
            raise ValueError(f'Unknown preprocessor {name}')


def get_preprocessors(configs: Union[List[Union[Dict, DictConfig]], ListConfig]) -> List[Preprocessor]:
    configs = config_to_primitive(configs)
    return [get_preprocessor(**config) for config in configs]
