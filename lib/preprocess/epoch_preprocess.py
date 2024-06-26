from abc import ABC, abstractmethod
from mne import Epochs
from enum import Enum
from lib.utils import config_to_primitive


class EpochPreprocessors(Enum):
    EpochFilter = 'bandpass_filter'


class EpochPreprocessor(ABC):

    @abstractmethod
    def preprocess(self, epochs: Epochs) -> Epochs:
        pass


class EpochFilter(EpochPreprocessor):

    def __init__(self,
                 low_cut: float,
                 high_cut: float,
                 method: str = 'iir'
                 ):
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.method = method

    def preprocess(self, epochs: Epochs) -> Epochs:
        epochs = epochs.filter(
            l_freq=self.low_cut,
            h_freq=self.high_cut,
            method=self.method,
            verbose=False
        )
        return epochs


def get_epoch_preprocessor(name: str, **kwargs) -> EpochPreprocessor:
    kwargs = config_to_primitive(kwargs)
    match name:
        case EpochPreprocessors.EpochFilter.value:
            return EpochFilter(**kwargs)
        case _:
            raise ValueError(f'Epoch preprocessor {name} not found')


def get_epoch_preprocessors(configs: list[dict]) -> list[EpochPreprocessor]:
    return [get_epoch_preprocessor(config['name'], **config['kwargs']) for config in configs]
