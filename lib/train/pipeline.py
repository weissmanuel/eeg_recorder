from sklearn.ensemble import RandomForestClassifier
from enum import Enum
from typing import List, Dict
from sklearn.pipeline import Pipeline, make_pipeline
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from numpy import ndarray
import numpy as np
from sklearn.svm import SVC
from pyriemann.classification import MDM, SVC as R_SVC
from pyriemann.estimation import XdawnCovariances
from lib.preprocess.algorithms.utils import filterbank
from lib.preprocess.algorithms.cca import ECCA, SCCA_canoncorr
from lib.preprocess.algorithms.trca import MSCCA_and_MSETRCA
from lib.utils import config_to_primitive


class PipelineSteps(Enum):
    CHANNEL_RESHAPE = 'CHANNEL_RESHAPE'
    FFT = 'FFT'
    FILTERBANK = 'FILTERBANK'
    XDAWN = 'XDAWN'
    RANDOM_FOREST = 'RANDOM_FOREST'
    SVM = 'SVM'
    R_SVM = 'R_SVM'
    MDM = 'MDM'
    CCA = 'CCA'
    ECCA = 'ECCA'
    MSCCA_AND_MSETRCA = 'MSCCA_AND_MSETRCA'

    @staticmethod
    def from_str(label: str):
        try:
            return PipelineSteps[label.upper()]
        except KeyError:
            raise ValueError(f'Unknown pipeline step: {label}')


class FFT(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: ndarray):
        data_freq = np.fft.fft(X, axis=2)
        data_freq = 2 / data_freq.shape[2] * np.abs(data_freq[:, :, :data_freq.shape[2] // 2])
        return data_freq

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class ChannelReshape(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: ndarray):
        if len(X.shape) > 2:
            return np.reshape(X, (X.shape[0], -1))
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class ThresholdingDecoder(BaseEstimator, TransformerMixin):

    def __init__(self, sfreq: float, target_frequencies: List[float], channel: int = 0, band_width: float = 1.0):
        self.sfreq = sfreq
        self.target_frequencies = target_frequencies
        self.band_width = band_width
        self.channel = channel

    def fit(self, X, y=None):
        return self

    def transform(self, X: ndarray):
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def predict(self, X: ndarray):
        if len(X.shape) > 2 and X.shape[0] != 1:
            raise ValueError('Thresholding transformer can only predict one sample at a time')

        X = X.squeeze()

        data = X[self.channel]
        num_samples = data.shape[-1]
        fft_data = np.fft.fft(data)
        fft_magnitude = np.abs(fft_data)
        frequencies = np.fft.fftfreq(num_samples, 1 / self.sfreq)

        max_magnitude = 0
        most_prominent_frequency = None

        for target_frequency in self.target_frequencies:
            # Find the indices of the FFT bins within the band around the target frequency
            band_indices = np.where((frequencies >= target_frequency - self.band_width) &
                                    (frequencies <= target_frequency + self.band_width))[0]

            # Sum the magnitudes of these bins
            band_magnitude = np.sum(fft_magnitude[band_indices])

            # Check if this is the most prominent frequency so far
            if band_magnitude > max_magnitude:
                max_magnitude = band_magnitude
                most_prominent_frequency = target_frequency
        return np.array([most_prominent_frequency])


class Filterbank(BaseEstimator, TransformerMixin):

    def __init__(self, sfreq: float):
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X: ndarray):
        result = [filterbank(self.sfreq, trial) for trial in X]
        return np.array(result)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


def build_pipeline(steps: List[Dict]) -> Pipeline:
    steps = get_pipeline_steps(steps)
    return make_pipeline(*steps)


def get_pipeline_step(step: PipelineSteps | str, **kwargs):
    step = check_step_type(step)
    kwargs = config_to_primitive(kwargs)
    match step:
        case PipelineSteps.CHANNEL_RESHAPE:
            return ChannelReshape()
        case PipelineSteps.FFT:
            return FFT()
        case PipelineSteps.FILTERBANK:
            return Filterbank(**kwargs)
        case PipelineSteps.XDAWN:
            return XdawnCovariances(**kwargs)
        case PipelineSteps.RANDOM_FOREST:
            return RandomForestClassifier(**kwargs)
        case PipelineSteps.SVM:
            return SVC(**kwargs)
        case PipelineSteps.R_SVM:
            return R_SVC(**kwargs)
        case PipelineSteps.MDM:
            return MDM()
        case PipelineSteps.CCA:
            return SCCA_canoncorr(**kwargs)
        case PipelineSteps.ECCA:
            return ECCA(**kwargs)
        case PipelineSteps.MSCCA_AND_MSETRCA:
            return MSCCA_and_MSETRCA(**kwargs)
        case _:
            raise ValueError(f'Unknown pipeline step: {step}')


def get_pipeline_steps(steps: List[Dict]):
    return [get_pipeline_step(step['name'], **step['kwargs']) for step in steps]


def check_step_type(step: PipelineSteps | str):
    return step if isinstance(step, PipelineSteps) else PipelineSteps.from_str(step)


def save_pipeline(pipeline: Pipeline, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Pipeline file {path} does not exist')
    return joblib.load(path)
