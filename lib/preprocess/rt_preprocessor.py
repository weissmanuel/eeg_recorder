from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from enum import Enum
from scipy.signal import butter, sosfilt, lfilter, sosfilt_zi, lfilter_zi, iirnotch
from lib.utils import config_to_primitive
from typing import List, Dict, Union
from omegaconf import DictConfig, ListConfig
from typing import Tuple


class RTPreprocessors(Enum):
    BANDPASS_FILTER = 'bandpass_filter'
    NOTCH_FILTER = 'notch_filter'


class RTPreprocessor(ABC):
    name: str

    def __init__(self, name: str, **kwargs):
        self.name = name

    @abstractmethod
    def __call__(self, data: ndarray, *args, **kwargs) -> ndarray:
        pass


class RTNotchFilter(RTPreprocessor):
    coefficients: ndarray
    zis: List[ndarray] | None = None

    def __init__(self,
                 freqs: List[float] | float,
                 quality: float,
                 sfreq: float,
                 n_channels: int):
        super().__init__(RTPreprocessors.NOTCH_FILTER.name)
        self.quality = quality
        self.sfreq = sfreq
        self.n_channels = n_channels

        freqs = [freqs] if isinstance(freqs, (float, int)) else freqs
        self.freqs = freqs
        coefficients = []
        for freq in self.freqs:
            b, a = iirnotch(freq, quality, sfreq)
            coefficients.append((b, a))
        self.coefficients = np.array(coefficients)
        self.zis = None

    def init_zis(self):
        self.zis: List[ndarray] = []
        for coefficients in self.coefficients:
            b, a = coefficients
            zi = lfilter_zi(b, a)
            zi = np.tile(zi, (self.n_channels, 1))
            self.zis.append(zi)

    def __call__(self, data: ndarray, *args, **kwargs) -> ndarray:
        if self.zis is None:
            self.init_zis()
        for i, (zi, (b, a)) in enumerate(zip(self.zis, self.coefficients)):
            data, zi = lfilter(b, a, data, zi=zi, axis=-1)
            self.zis[i] = zi
        return data


class RTBandPassFilter(RTPreprocessor):

    def __init__(self, low_cut: float, high_cut: float, sfreq: float, n_channels: int, order: int = 4, **kwargs):
        super().__init__(RTPreprocessors.BANDPASS_FILTER.name, **kwargs)

        self.low_cut = low_cut
        self.high_cut = high_cut
        self.sfreq = sfreq
        self.order = order
        self.n_channels = n_channels

        self.b, self.a = butter(self.order, [self.low_cut, self.high_cut], analog=False, btype='band',
                                fs=self.sfreq)

        self.zi = None

    def init_zi(self):
        zi = lfilter_zi(self.b, self.a)
        zi = np.tile(zi, (self.n_channels, 1))
        self.zi = zi

    def __call__(self, data: ndarray, *args, **kwargs) -> ndarray:
        if self.zi is None:
            self.init_zi()
        filtered_data, self.zi = lfilter(self.b, self.a, data, zi=self.zi, axis=-1)
        return filtered_data


def get_rt_preprocessor(name: str, **kwargs) -> RTPreprocessor:
    match name:
        case RTPreprocessors.BANDPASS_FILTER.value:
            return RTBandPassFilter(**kwargs)
        case RTPreprocessors.NOTCH_FILTER.value:
            return RTNotchFilter(**kwargs)
        case _:
            raise ValueError(f'Unknown preprocessor: {name}')


def get_rt_preprocessors(configs: Union[List[Union[Dict, DictConfig]], ListConfig]) -> List[RTPreprocessor]:
    configs = config_to_primitive(configs)
    return [get_rt_preprocessor(**config) for config in configs]
