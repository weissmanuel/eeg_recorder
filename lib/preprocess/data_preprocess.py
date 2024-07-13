import numpy as np
from numpy import ndarray
from mne import Info
from typing import List, Dict
from enum import Enum
from omegaconf import DictConfig, ListConfig
from lib.utils import config_to_primitive
from abc import ABC, abstractmethod
from typing import Tuple, Union
from collections import deque
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt
from .models import ProcessStage


class Preprocessors(Enum):
    EEGScaler = 'eeg_scaler'
    EpochWindowSplitter = 'epoch_window_splitter'
    FFT = 'fft'
    SklearnClassifierReshape = 'sklearn_classifier_reshape'
    NOTCH_FILTER = 'notch_filter'
    BANDPASS_FILTER = 'bandpass_filter'
    KALMAN_FILTER = 'kalman_filter'


PreprocessResponse = Union[ndarray, Tuple[ndarray, Union[ndarray, None]]]


class Preprocessor(ABC):
    name: str
    stages: List[ProcessStage] | None

    def __init__(self, name: str, stages: List[ProcessStage] | List[str] | None = None):
        self.name = name

        if stages is not None:
            self.stages = [ProcessStage.from_str(stage) if isinstance(stage, str) else stage for stage in stages]
        else:
            self.stages = None

    @abstractmethod
    def __call__(self, data: ndarray, labels: ndarray | None = None, *args, **kwargs) -> PreprocessResponse:
        pass


class EEGScaler(Preprocessor):
    scale: float
    channel_types: List[str]

    def __init__(self,
                 scale: float = 1e-6,
                 channel_types: List[str] = None,
                 stages: List[ProcessStage] | List[str] | None = None):
        super().__init__(Preprocessors.EEGScaler.name, stages)
        self.scale = scale
        if channel_types is None:
            self.channel_types = ['eeg']
        else:
            self.channel_types = channel_types

    def __call__(self, data: ndarray, labels: ndarray | None = None, *args, **kwargs) -> ndarray:
        info: Info | None = kwargs.get('info', None)
        assert info is not None, 'info must be provided for EEGScaler'
        ch_idx = [i for i, ch_type in enumerate(info.get_channel_types()) if ch_type in self.channel_types]
        if len(ch_idx) > 0:
            data[ch_idx] = data[ch_idx] * self.scale
        return data


class NotchFilter(Preprocessor):
    freqs: List[float]
    quality: float
    sfreq: float

    coefficients: ndarray

    def __init__(self,
                 freqs: List[float] | float,
                 quality: float, sfreq: float,
                 stages: List[ProcessStage] | List[str] | None = None):
        super().__init__(Preprocessors.NOTCH_FILTER.name, stages)
        self.quality = quality
        self.sfreq = sfreq

        freqs = [freqs] if isinstance(freqs, (float, int)) else freqs
        self.freqs = freqs

        coefficients = []
        for freq in self.freqs:
            b, a = iirnotch(freq, quality, sfreq)
            coefficients.append((b, a))
        self.coefficients = np.array(coefficients)

    def __call__(self, data: ndarray, *args, **kwargs) -> ndarray:
        for b, a in self.coefficients:
            data = filtfilt(b, a, data)
        return data


class BandpassFilter(Preprocessor):

    def __init__(self,
                 low_cut: float,
                 high_cut: float,
                 sfreq: float,
                 order: int = 4,
                 padtype: str = 'odd',
                 padlen: int | None = None,
                 stages: List[ProcessStage] | List[str] | None = None):
        super().__init__(Preprocessors.BANDPASS_FILTER.name, stages)
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.sfreq = sfreq
        self.order = order
        self.padtype = padtype
        self.padlen = padlen

        self.sos = butter(order, [low_cut, high_cut], analog=False, btype='band', fs=sfreq, output='sos')

    def __call__(self, data: ndarray, *args, **kwargs) -> ndarray:
        return sosfiltfilt(self.sos, data, padtype=self.padtype, padlen=self.padlen)


class EpochWindowSplitter(Preprocessor):
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
                 average_size: int = 5,
                 stages: List[ProcessStage] | List[str] | None = None):
        super().__init__(Preprocessors.EpochWindowSplitter.name, stages)
        self.sfreq = sfreq
        self.window_size_seconds = window_size_seconds
        self.window_shift_seconds = window_shift_seconds

        self.window_size = int(window_size_seconds * self.sfreq)
        self.window_shift = int(window_shift_seconds * self.sfreq)

        self.use_averaging = use_averaging
        self.average_size = average_size

    def __call__(self, data: ndarray, labels: ndarray | None = None, *args, **kwargs) -> PreprocessResponse:
        assert labels is not None, 'Labels must be provided for EpochWindowSplitter'
        assert len(data.shape) == 3, 'Data must be 3D (n_epochs, n_channels, n_samples)'

        n_epochs, n_channels, n_samples = data.shape

        windows = []
        window_labels = []

        for i in range(n_epochs):
            queue = deque(maxlen=self.average_size)
            for j in range(0, n_samples - self.window_size, self.window_shift):
                window = data[i, :, j:j + self.window_size]
                if self.use_averaging:
                    queue.append(window)
                    window = np.mean(list(queue), axis=0)
                windows.append(window)
                window_labels.append(labels[i])

        windows = np.array(windows)
        window_labels = np.array(window_labels)

        return windows, window_labels


class FFT(Preprocessor):

    def __init__(self, stages: List[ProcessStage] | List[str] | None = None):
        super().__init__(Preprocessors.FFT.name, stages)

    def __call__(self, data: ndarray, *args, **kwargs) -> PreprocessResponse:
        data_freq = np.fft.fft(data, axis=2)
        data_freq = 2 / data_freq.shape[2] * np.abs(data_freq[:, :, :data_freq.shape[2] // 2])
        return data_freq


class KalmanFilterPreprocessor(Preprocessor):
    def __init__(self, stages: List[ProcessStage] | List[str] | None = None):
        super().__init__(Preprocessors.KALMAN_FILTER.name, stages)
        # Initialize Kalman filter parameters
        self.A = np.eye(1)  # State transition matrix
        self.H = np.eye(1)  # Observation matrix
        self.Q = np.eye(1) * 0.0001  # Process noise covariance
        self.R = np.eye(1) * 0.01  # Measurement noise covariance
        self.P = np.eye(1)  # Estimate error covariance
        self.x = np.zeros((1, 1))  # Initial state

    def __call__(self, data: ndarray, labels: ndarray | None = None, *args, **kwargs) -> PreprocessResponse:
        n_channels, n_times = data.shape
        filtered_data = np.zeros((n_channels, n_times))

        for channel in range(n_channels):
            for t in range(n_times):
                # Prediction
                self.x = self.A @ self.x
                self.P = self.A @ self.P @ self.A.T + self.Q

                # Update
                z = np.array([[data[channel, t]]])  # Observation
                y = z - self.H @ self.x
                S = self.H @ self.P @ self.H.T + self.R
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y
                self.P = (np.eye(1) - K @ self.H) @ self.P

                # Save filtered value
                filtered_data[channel, t] = self.x

        return filtered_data


class SklearnClassifierReshape(Preprocessor):
    def __init__(self, stages: List[ProcessStage] | List[str] | None = None):
        super().__init__(Preprocessors.SklearnClassifierReshape.name, stages)

    def __call__(self, data: ndarray, *args, **kwargs) -> PreprocessResponse:
        return data.reshape(data.shape[0], -1), None


def get_preprocessor(name: str, **kwargs) -> Preprocessor:
    match name:
        case Preprocessors.EEGScaler.value:
            return EEGScaler(**kwargs)
        case Preprocessors.NOTCH_FILTER.value:
            return NotchFilter(**kwargs)
        case Preprocessors.BANDPASS_FILTER.value:
            return BandpassFilter(**kwargs)
        case Preprocessors.EpochWindowSplitter.value:
            return EpochWindowSplitter(**kwargs)
        case Preprocessors.FFT.value:
            return FFT()
        case Preprocessors.KALMAN_FILTER.value:
            return KalmanFilterPreprocessor(**kwargs)
        case _:
            raise ValueError(f'Unknown preprocessor {name}')


def get_preprocessors(configs: Union[List[Union[Dict, DictConfig]], ListConfig],
                      stages: List[ProcessStage] | None = None) -> List[Preprocessor]:
    configs = config_to_primitive(configs)
    preprocessors = [get_preprocessor(**config) for config in configs]
    if stages is not None:
        preprocessors = [preprocessor for preprocessor in preprocessors if preprocessor.stages is None or any(
            stage in preprocessor.stages for stage in stages)]
    return preprocessors
