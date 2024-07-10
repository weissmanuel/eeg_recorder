from abc import ABC, abstractmethod
from mne.io import BaseRaw
from typing import List
from enum import Enum
from lib.utils import config_to_primitive
from .models import ProcessStage


class RawPreprocessors(Enum):
    NOTCH_FILTER = 'notch_filter'
    BANDPASS_FILTER = 'bandpass_filter'
    RESAMPLE = 'resample'
    REFERENCE = 'reference'


class RawPreprocessor(ABC):

    name: str
    stages: List[ProcessStage] | None = None

    def __init__(self, name: str, stages: List[ProcessStage] | List[str] | None = None):
        self.name = name
        if stages is not None:
            self.stages = [ProcessStage.from_str(stage) if isinstance(stage, str) else stage for stage in stages]
        else:
            self.stages = None

    @abstractmethod
    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        pass

    def __call__(self, raw: BaseRaw) -> BaseRaw:
        return self.preprocess(raw)


class RawFilter(RawPreprocessor):

    def __init__(self,
                 low_cut: float,
                 high_cut: float,
                 method: str = 'iir',
                 stages: List[ProcessStage] | List[str] | None = None
                 ):
        super().__init__(RawPreprocessors.BANDPASS_FILTER.name, stages)
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.method = method

    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        raw.filter(l_freq=self.low_cut, h_freq=self.high_cut,
                   method=self.method, verbose=False)
        return raw


class RawNotchFilter(RawPreprocessor):

    def __init__(self,
                 freqs: float | List[float],
                 method: str = 'iir',
                 filter_length: float | List[float] | None = 'auto',
                 notch_widths: float | List[float] | None = 1,
                 stages: List[ProcessStage] | List[str] | None = None
                 ):
        super().__init__(RawPreprocessors.NOTCH_FILTER.name, stages)
        self.freqs = freqs
        self.method = method
        self.filter_length = filter_length
        self.notch_widths = notch_widths

    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        raw.notch_filter(self.freqs, method=self.method, verbose=False,
                         filter_length=self.filter_length, notch_widths=self.notch_widths)
        return raw


class Resample(RawPreprocessor):

    def __init__(self, sfreq: float, stages: List[ProcessStage] | List[str] | None = None):
        super().__init__(RawPreprocessors.RESAMPLE.name, stages)
        self.sfreq = sfreq

    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        raw.resample(sfreq=self.sfreq)
        return raw


class Reference(RawPreprocessor):

    def __init__(self,
                 ref_channels: str | List[str] = 'average',
                 projection: bool = False,
                 stages: List[ProcessStage] | List[str] | None = None
                 ):
        super().__init__(RawPreprocessors.REFERENCE.name, stages)
        self.ref_channels = ref_channels
        self.projection = projection

    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        raw.set_eeg_reference(ref_channels=self.ref_channels, projection=self.projection)
        return raw


def get_raw_preprocessor(name: str, **kwargs) -> RawPreprocessor:
    kwargs = config_to_primitive(kwargs)
    match name:
        case RawPreprocessors.BANDPASS_FILTER.value:
            return RawFilter(**kwargs)
        case RawPreprocessors.NOTCH_FILTER.value:
            return RawNotchFilter(**kwargs)
        case RawPreprocessors.RESAMPLE.value:
            return Resample(**kwargs)
        case RawPreprocessors.REFERENCE.value:
            return Reference(**kwargs)
        case _:
            raise ValueError(f'Unknown raw preprocessor {name}')


def get_raw_preprocessors(configs: List[dict], stages: List[ProcessStage] | None = None) -> List[RawPreprocessor]:
    preprocessors = [get_raw_preprocessor(config['name'], **config['kwargs']) for config in configs]
    if stages is not None:
        preprocessors = [preprocessor for preprocessor in preprocessors if preprocessor.stages is None or any(
            stage in preprocessor.stages for stage in stages)]
    return preprocessors
