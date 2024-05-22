import numpy as np
from numpy import ndarray
from mne import Info
from typing import Union, List, Dict
from enum import Enum
from omegaconf import DictConfig, ListConfig
from lib.utils import config_to_primitive


class Preprocessors(Enum):
    EEGScaler = 'eeg_scaler'


class Preprocessor:
    name: str

    def __init__(self, name: str):
        self.name = name

    def __call__(self, info: Info, signal: ndarray, *args, **kwargs) -> ndarray:
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

    def __call__(self, info: Info, signal: ndarray, *args, **kwargs) -> ndarray:
        ch_idx = [i for i, ch_type in enumerate(info.get_channel_types()) if ch_type in self.channel_types]
        if len(ch_idx) > 0:
            signal[ch_idx] = signal[ch_idx] * self.scale
        return signal


def get_preprocessor(name: str, **kwargs) -> Preprocessor:
    match name:
        case Preprocessors.EEGScaler.value:
            return EEGScaler(**kwargs)
        case _:
            raise ValueError(f'Unknown preprocessor {name}')


def get_preprocessors(configs: Union[List[Union[Dict, DictConfig]], ListConfig]) -> List[Preprocessor]:
    configs = config_to_primitive(configs)
    return [get_preprocessor(**config) for config in configs]
