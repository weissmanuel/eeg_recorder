from abc import ABC, abstractmethod
from mne import Epochs
from enum import Enum
from lib.utils import config_to_primitive
from .models import ProcessStage
from typing import List


class EpochPreprocessors(Enum):
    EpochFilter = 'bandpass_filter'


class EpochPreprocessor(ABC):

    name: str
    stages: List[ProcessStage] | None = None

    def __init__(self, name: str, stages: List[ProcessStage] | List[str] | None = None):
        self.name = name
        if stages is not None:
            self.stages = [ProcessStage.from_str(stage) if isinstance(stage, str) else stage for stage in stages]
        else:
            self.stages = None

    @abstractmethod
    def preprocess(self, epochs: Epochs) -> Epochs:
        pass


class EpochFilter(EpochPreprocessor):

    def __init__(self,
                 low_cut: float,
                 high_cut: float,
                 method: str = 'iir',
                 stages: List[ProcessStage] | List[str] | None = None
                 ):
        super().__init__(EpochPreprocessors.EpochFilter.name, stages)
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


def get_epoch_preprocessors(configs: List[dict], stages: List[ProcessStage] | None = None) -> List[EpochPreprocessor]:
    preprocessors = [get_epoch_preprocessor(config['name'], **config['kwargs']) for config in configs]
    if stages is not None:
        preprocessors = [preprocessor for preprocessor in preprocessors if preprocessor.stages is None or any(
            stage in preprocessor.stages for stage in stages)]
    return preprocessors

