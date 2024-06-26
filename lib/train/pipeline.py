from sklearn.ensemble import RandomForestClassifier
from enum import Enum
from typing import List, Dict
from sklearn.pipeline import Pipeline, make_pipeline
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from numpy import ndarray
import numpy as np
from sklearn.svm import SVC
from pyriemann.classification import MDM, SVC as R_SVC
from pyriemann.estimation import XdawnCovariances
from lib.train.algorithms.utils import filterbank
from lib.train.algorithms.cca import ECCA
from lib.train.algorithms.cca_new import CCAClassifier
from lib.train.algorithms.trca import MSCCA_and_MSETRCA
from lib.utils import config_to_primitive
from collections import deque
from typing import Tuple
from scipy import stats


class PipelineSteps(Enum):
    CHANNEL_RESHAPE = 'CHANNEL_RESHAPE'
    WINDOW_SPLITTER = 'WINDOW_SPLITTER'
    AVERAGER = 'AVERAGER'
    FFT = 'FFT'
    FILTERBANK = 'FILTERBANK'
    XDAWN = 'XDAWN'
    CONSECUTIVE_MAJORITY_VOTE_CLASSIFIER = 'CONSECUTIVE_MAJORITY_VOTE_CLASSIFIER'
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

    def transform(self, X, y=None, **fit_params):
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


class WindowSplitter(BaseEstimator, TransformerMixin):
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
        self.sfreq = sfreq
        self.window_size_seconds = window_size_seconds
        self.window_shift_seconds = window_shift_seconds

        self.window_size = int(window_size_seconds * self.sfreq)
        self.window_shift = int(window_shift_seconds * self.sfreq)

        self.use_averaging = use_averaging
        self.average_size = average_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **fit_params):
        assert y is not None, 'Labels must be provided for EpochWindowSplitter'
        assert len(X.shape) == 3, 'Data must be 3D (n_epochs, n_channels, n_samples)'

        n_epochs, n_channels, n_samples = X.shape

        windows = []
        window_labels = []

        for i in range(n_epochs):
            queue = deque(maxlen=self.average_size)
            for j in range(0, n_samples - self.window_size, self.window_shift):
                window = X[i, :, j:j + self.window_size]
                if self.use_averaging:
                    queue.append(window)
                    window = np.mean(list(queue), axis=0)
                windows.append(window)
                window_labels.append(y[i])

        windows = np.array(windows)
        window_labels = np.array(window_labels)

        return windows, window_labels

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)


class ConsecutiveMajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier: Tuple[BaseEstimator, dict], window_size=5):
        self.base_classifier = get_pipeline_step(base_classifier['name'], **base_classifier['kwargs']) \
            if isinstance(base_classifier, dict) else base_classifier
        self.window_size = window_size

    def fit(self, X: ndarray, y: ndarray):
        self.base_classifier.fit(X, y)
        return self

    def predict(self, X):
        orig_pred_y = np.array(self.base_classifier.predict(X))
        queue = deque(maxlen=self.window_size)
        pred_y = []
        for i in range(len(orig_pred_y)):
            queue.append(orig_pred_y[i])
            pred_y.append(stats.mode(queue).mode)
        out = np.array(pred_y)
        assert out.shape == orig_pred_y.shape, f'Expected shape {orig_pred_y.shape}, got {out.shape}'
        return out

    def predict_proba(self, X):

        if not hasattr(self.base_classifier, 'predict_proba'):
            orig_pred_y = self.predict(X)
            orig_pred_y_proba = np.zeros((orig_pred_y.size, orig_pred_y.max() + 1))
            orig_pred_y_proba[np.arange(orig_pred_y.size), orig_pred_y] = 1
        else:
            orig_pred_y_proba = np.array(self.base_classifier.predict_proba(X))
        queue = deque(maxlen=self.window_size)
        pred_y_proba = []
        for i in range(len(orig_pred_y_proba)):
            queue.append(orig_pred_y_proba[i])
            pred_y_proba.append(np.mean(queue, axis=0))
        out = np.array(pred_y_proba)
        assert out.shape == orig_pred_y_proba.shape, f'Expected shape {orig_pred_y_proba.shape}, got {out.shape}'
        return out



class Averager(BaseEstimator, TransformerMixin):

    def __init__(self, window_size: int):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X: ndarray):
        result = []
        queue = deque(maxlen=self.window_size)
        for trial in X:
            queue.append(trial)
            result.append(np.mean(list(queue), axis=0))
        out = np.array(result)
        assert out.shape == X.shape, f'Expected shape {X.shape}, got {out.shape}'
        return out


class ThresholdingDecoder(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 sfreq: float, target_frequencies: List[float],
                 channel: int = 0, band_width: float = 1.0,
                 average: int | None = None):
        self.sfreq = sfreq
        self.target_frequencies = target_frequencies
        self.band_width = band_width
        self.channel = channel
        self.average = average
        self.queue = deque(maxlen=average) if average is not None else None


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
        if self.average is not None:
            self.queue.append(fft_magnitude)
            fft_magnitude = np.mean(list(self.queue), axis=0)

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
        return np.array([self.target_frequencies.index(most_prominent_frequency)])


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
        case PipelineSteps.WINDOW_SPLITTER:
            return WindowSplitter(**kwargs)
        case PipelineSteps.AVERAGER:
            return Averager(**kwargs)
        case PipelineSteps.FFT:
            return FFT()
        case PipelineSteps.FILTERBANK:
            return Filterbank(**kwargs)
        case PipelineSteps.XDAWN:
            return XdawnCovariances(**kwargs)
        case PipelineSteps.CONSECUTIVE_MAJORITY_VOTE_CLASSIFIER:
            return ConsecutiveMajorityVoteClassifier(**kwargs)
        case PipelineSteps.RANDOM_FOREST:
            return RandomForestClassifier(**kwargs)
        case PipelineSteps.SVM:
            return SVC(**kwargs)
        case PipelineSteps.R_SVM:
            return R_SVC(**kwargs)
        case PipelineSteps.MDM:
            return MDM()
        case PipelineSteps.CCA:
            return CCAClassifier(**kwargs)
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
