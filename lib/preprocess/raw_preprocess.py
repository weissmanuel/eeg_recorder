from abc import ABC, abstractmethod
from mne.io import BaseRaw
from typing import List
from enum import Enum
from lib.utils import config_to_primitive
from .models import ProcessStage
import mne
import numpy as np


class RawPreprocessors(Enum):
    NOTCH_FILTER = 'notch_filter'
    BANDPASS_FILTER = 'bandpass_filter'
    RESAMPLE = 'resample'
    REFERENCE = 'reference'
    CHANNEL_INTERPOLATION = 'channel_interpolation'
    CHANNEL_PICKER = 'channel_picker'


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


class ChannelInterpolation(RawPreprocessor):

    def __init__(self, threshold: float, stages: List[ProcessStage] | List[str] | None = None,
                 verbose: bool = False):
        super().__init__(RawPreprocessors.CHANNEL_INTERPOLATION.name, stages)
        self.threshold = threshold
        self.verbose = verbose

    def get_bad_channels(self, raw: BaseRaw):
        try:
            data = raw.get_data(verbose=self.verbose)
            if data.size == 0:
                raise ValueError("The data is empty. Please check the raw instance.")
            if not np.issubdtype(data.dtype, np.number):
                raise ValueError("Data type is not numeric. Please provide valid EEG data.")
            if self.threshold <= 0:
                raise ValueError("Threshold must be a positive value.")

            bad_segments = np.abs(data) > self.threshold
            if not np.any(bad_segments):
                if self.verbose:
                    print("No segments exceed the threshold.")
                return np.array([])
            bad_channels = np.any(bad_segments, axis=1)
            return np.where(bad_channels)[0]
        except Exception as e:
            print(f"An error occurred while marking bad segments: {e}")
            return np.array([])

    def interpolate_bad_segments(self, raw: BaseRaw):
        try:
            bad_channels = self.get_bad_channels(raw)
            if bad_channels.size == 0:
                if self.verbose:
                    print("No bad segments to interpolate.")
                return raw
            ch_names = raw.info['ch_names']
            raw.info['bads'] = [ch_names[i] for i in bad_channels]
            raw = raw.interpolate_bads(reset_bads=True, mode='accurate', verbose=50)
            return raw
        except IndexError as e:
            print(f"Index error occurred during interpolation: {e}")
            return raw
        except Exception as e:
            print(f"An error occurred during interpolation: {e}")
            return raw

    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        try:
            raw = self.interpolate_bad_segments(raw)
            return raw
        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
            return raw


class ChannelPicker(RawPreprocessor):

    def __init__(self, picks: List[str], stages: List[ProcessStage] | List[str] | None = None):
        super().__init__(RawPreprocessors.CHANNEL_PICKER.name, stages)
        self.picks = picks

    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        raw.pick(self.picks)
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
        case RawPreprocessors.CHANNEL_INTERPOLATION.value:
            return ChannelInterpolation(**kwargs)
        case RawPreprocessors.CHANNEL_PICKER.value:
            return ChannelPicker(**kwargs)
        case _:
            raise ValueError(f'Unknown raw preprocessor {name}')


def get_raw_preprocessors(configs: List[dict], stages: List[ProcessStage] | None = None) -> List[RawPreprocessor]:
    preprocessors = [get_raw_preprocessor(config['name'], **config['kwargs']) for config in configs]
    if stages is not None:
        preprocessors = [preprocessor for preprocessor in preprocessors if preprocessor.stages is None or any(
            stage in preprocessor.stages for stage in stages)]
    return preprocessors
