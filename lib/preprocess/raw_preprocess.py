from abc import ABC, abstractmethod
from mne.io import BaseRaw


class RawPreprocessor(ABC):

    @abstractmethod
    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        pass


class RawFilter(RawPreprocessor):

    def __init__(self, low_freq: float, high_freq: float):
        self.low_freq = low_freq
        self.high_freq = high_freq

    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        raw.filter(self.low_freq, self.high_freq, method='iir')
        return raw


class RawNotchFilter(RawPreprocessor):

    def __init__(self, freqs: list | float):
        self.freqs = freqs

    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        raw.notch_filter(self.freqs, method='iir')
        return raw


class Resample(RawPreprocessor):

    def __init__(self, sfreq: float):
        self.sfreq = sfreq

    def preprocess(self, raw: BaseRaw) -> BaseRaw:
        raw.resample(sfreq=self.sfreq)
        return raw
