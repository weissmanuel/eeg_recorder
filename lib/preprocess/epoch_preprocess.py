from abc import ABC, abstractmethod
from mne import Epochs


class EpochPreprocessor(ABC):

    @abstractmethod
    def preprocess(self, epochs: Epochs) -> Epochs:
        pass


class EpochFilter(EpochPreprocessor):

    def __init__(self, low_freq: float, high_freq: float):
        self.low_freq = low_freq
        self.high_freq = high_freq

    def preprocess(self, epochs: Epochs) -> Epochs:
        epochs.filter(self.low_freq, self.high_freq, method='iir')
        return epochs