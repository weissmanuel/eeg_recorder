from abc import ABC, abstractmethod
from .store import StreamStore, RecorderStore, StreamType
import time
from datetime import datetime, timedelta
import logging
from typing import Union, List
from numpy import ndarray
from mne_lsl.stream import StreamLSL
import math
from mne_lsl.lsl import local_clock
from lib.utils import format_seconds
from multiprocessing import Process
from lib.lsl import connect, disconnect
from lib.store import RecorderStore, StreamStore, RealTimeStore
import time
from omegaconf import DictConfig
from lib.persist import Persister, PersistingMode, MneRawPersister
import numpy as np
import threading
from threading import Thread
import scipy
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from typing import Tuple


class Worker(ABC):
    logger = logging.getLogger(__name__)

    process: Union[Process, Thread]

    @abstractmethod
    def get_new_process(self) -> Union[Process, Thread]:
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def work(self):
        pass


class ProcessWorker(Worker):

    def get_new_process(self):
        return Process(target=self.work)

    def start(self):
        self.process.start()

    def stop(self):
        try:
            self.process.terminate()
            self.process.join()
        except Exception as e:
            self.logger.error(f"Error while stopping worker: {e}")
        self.process = self.get_new_process()

    @abstractmethod
    def work(self):
        pass


class ThreadWorker(Worker):

    def get_new_process(self):
        return Thread(target=self.work)

    def start(self):
        self.process.start()

    def stop(self):
        try:
            self.process.join()
        except Exception as e:
            self.logger.error(f"Error while stopping worker: {e}")
        self.process = self.get_new_process()

    @abstractmethod
    def work(self):
        pass


class RecordingWorker(ProcessWorker):
    recorder_store: RecorderStore
    stream_store: StreamStore

    buffer_size_seconds: float

    def __init__(self,
                 recorder_store: RecorderStore,
                 stream_store: StreamStore,
                 buffer_size_seconds: float
                 ):

        self.recorder_store = recorder_store
        self.stream_store = stream_store

        self.buffer_size_seconds = buffer_size_seconds

        self.process = self.get_new_process()
        self.sleep_time = 0.01

    def buffer_size(self, stream: StreamLSL) -> int:
        if stream is not None:
            sfreq = self.stream.info['sfreq']
            return math.ceil(self.buffer_size_seconds * sfreq)
        return 0

    @property
    def source_id(self) -> str:
        return self.stream_store.source_id

    @property
    def stream_type(self) -> StreamType:
        return self.stream_store.stream_type

    @property
    def recording_completed(self) -> bool:
        return self.stream_store.recording_completed

    def reset(self):
        self.stream_store.reset()

    def retrieve_stream_info(self, stream: StreamLSL):
        self.stream_store.stream_info = stream.info
        self.stream_store.n_channels = len(stream.ch_names)
        self.stream_store.has_stream = True
        self.stream_store.sfreq = stream.info['sfreq']
        self.stream_store.time_shift = 0.0
        self.stream_store.start_time_seconds = local_clock()

    def evaluate_time_shift(self, times: Union[List, None] = None, recording_stopped_at: Union[float, None] = None):
        if times is not None:
            self.stream_store.last_batch_received_time = times[-1]
        self.stream_store.current_time = recording_stopped_at if recording_stopped_at is not None else local_clock()
        self.stream_store.time_shift = self.stream_store.current_time - self.stream_store.last_batch_received_time

    def add_data(self, data: any, times: any, recording_stopped_at: Union[float, None] = None):
        self.stream_store.append_data(data.copy())
        self.stream_store.append_times(times.copy())

        self.evaluate_time_shift(times, recording_stopped_at)

    def log_first_iteration(self, times: List[any]):
        self.stream_store.first_sample_lsl_seconds = times[0].copy()
        self.stream_store.first_sample_system_seconds = local_clock()
        self.stream_store.first_sample_datetime = datetime.utcnow()
        self.logger.debug(f"First Signal Time of {self.source_id}: {self.stream_store.first_sample_datetime}")

    def log_recording_completed(self):
        n_samples = self.stream_store.n_samples
        expected_samples = self.stream_store.expected_samples
        difference = self.stream_store.sample_deviation
        duration = self.stream_store.duration

        if len(self.stream_store.times) > 1:
            delta_seconds = self.stream_store.times[-1] - self.stream_store.times[0]
            self.stream_store.last_sample_datetime = self.stream_store.first_sample_datetime + timedelta(
                seconds=delta_seconds)
        else:
            self.stream_store.last_sample_datetime = self.stream_store.first_sample_datetime

        self.logger.info(f"Recording stopped for ${self.source_id}. Recorded: {n_samples}, "
                         f"Expected: {expected_samples}, Difference: {difference}, "
                         f"Duration: {format_seconds(duration)}")

    def continue_recording(self) -> bool:
        if self.stream_store.stream_type.is_signal:
            return (self.recorder_store.is_recording or
                    (self.stream_store.time_shift > -self.stream_store.safety_offset_seconds
                     and self.stream_store.n_samples > 0))
        else:
            offset = self.stream_store.current_time - self.stream_store.last_sample_lsl_seconds
            return self.recorder_store.is_recording or offset < self.stream_store.safety_offset_seconds

    def work(self):

        stream = connect(self.stream_store.source_id, self.stream_store.stream_type, self.buffer_size_seconds)

        if stream is not None and stream.connected:

            self.logger.info(f"Start Recording of Stream: {self.stream_store.source_id}")
            self.retrieve_stream_info(stream)

            recording_stopped_at: Union[float, None] = None

            while self.continue_recording():

                if self.recorder_store.is_paused:
                    time.sleep(0.5)
                    continue

                if not self.recorder_store.is_recording and recording_stopped_at is None:
                    recording_stopped_at = local_clock()

                if self.stream_store.sfreq is not None and self.stream_store.sfreq > 0:
                    window_size = stream.n_new_samples / self.stream_store.sfreq
                else:
                    window_size = stream.n_new_samples

                if window_size > 0:
                    (values, times) = stream.get_data(winsize=window_size)

                    if values is not None and len(values) > 0:
                        self.add_data(values, times, recording_stopped_at)

                        if self.stream_store.iterations == 0:
                            self.log_first_iteration(times)

                        self.stream_store.increment_iterations()
                else:
                    self.evaluate_time_shift(recording_stopped_at=recording_stopped_at)
                time.sleep(self.sleep_time)

            self.stream_store.end_time_seconds = (recording_stopped_at if
                                                  recording_stopped_at is not None else local_clock())

            if self.stream_store.sample_deviation > 0:
                self.logger.warning(
                    f"Recorded less samples than expected: {self.stream_store.sample_deviation} samples")

            self.log_recording_completed()

            self.logger.debug(f"Finished Signal Recording for Stream: {self.source_id}")
            self.stream_store.recording_completed = True
            disconnect(stream)
        else:
            self.stream_store.recording_completed = True


class PersistenceWorker(ProcessWorker):

    def __init__(self,
                 interval: int,
                 recorder_store: RecorderStore,
                 stream_stores: List[StreamStore],
                 persister: MneRawPersister
                 ):
        self.interval = interval

        self.recorder_store = recorder_store
        self.stream_stores = stream_stores

        self.persister = persister

        self.process = self.get_new_process()

    def work(self):
        while self.recorder_store.is_recording:
            time.sleep(self.interval)
            self.recorder_store.pause_recording()
            self.persister.save(self.stream_stores, intermediate_save=True)
            self.recorder_store.resume_recording()


class RealTimeWorker(ProcessWorker):

    def __init__(self,
                 recorder_store: RecorderStore,
                 real_time_store: RealTimeStore
                 ):
        self.recorder_store = recorder_store
        self.real_time_store = real_time_store

    def work(self):
        pass


class RealTimeRecorder(RealTimeWorker):

    def __init__(self,
                 recorder_store: RecorderStore,
                 real_time_store: RealTimeStore
                 ):
        super().__init__(recorder_store, real_time_store)

        self.process: Thread = self.get_new_process()

    def prepare_data(self, data: ndarray) -> List:
        return (data.transpose()).tolist()

    @staticmethod
    def generate_data(sample_frequency: float, num_channels: int = 2) -> ndarray:
        data = []
        timesteps = np.linspace(start=0.0, stop=1.0, num=int(sample_frequency), endpoint=False)
        ch = lambda x: np.sin(2 * np.pi * 4 * x) + 0.5 * np.sin(2 * np.pi * 8 * x) + 0.3 * np.sin(
            2 * np.pi * 10 * x) + 0.5 * np.sin(2 * np.pi * 20 * x)
        for i in range(num_channels):
            data.append([ch(t) for t in timesteps])
        return np.array(data)

    def get_stream_data(self, stream: StreamLSL) -> ndarray | None:
        window_size = stream.n_new_samples / self.real_time_store.sfreq
        if window_size > 0:
            (values, times) = stream.get_data(winsize=window_size)
            return values

    def get_data(self, stream: StreamLSL) -> ndarray | None:
        if self.real_time_store.source_id == 'demo':
            return RealTimeRecorder.generate_data(sample_frequency=self.real_time_store.sfreq, num_channels=2)
        else:
            return self.get_stream_data(stream)

    def work(self):
        iteration = 0
        source_id = self.real_time_store.source_id
        stream = connect(source_id, self.real_time_store.stream_type, self.real_time_store.buffer_size_seconds)

        if (stream is not None and stream.connected) or source_id == 'demo':

            self.logger.info(f"Start Real-Time Recording for Stream: {source_id}")
            while self.recorder_store.is_recording:
                data = self.get_data(stream)
                if data is not None:
                    data = self.prepare_data(data)
                    self.real_time_store.add_data(data)
                time.sleep(1 if source_id == 'demo' else 0.01)
                iteration += 1


class RealTimeSSVEPDecoder(RealTimeWorker):

    def __init__(self,
                 recorder_store: RecorderStore,
                 real_time_store: RealTimeStore
                 ):
        super().__init__(recorder_store, real_time_store)

        self.process: Thread = self.get_new_process()
        self.band_pass_coefficients = self.butter_bandpass()

    def butter_bandpass(self, order=5):
        nyq = 0.5 * self.real_time_store.sfreq
        low = self.real_time_store.low_cut / nyq
        high = self.real_time_store.high_cut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data):
        y = lfilter(self.band_pass_coefficients[0], self.band_pass_coefficients[1], data, axis=1)
        return y

    def reshape_data(self, data: ndarray) -> ndarray:
        return data.transpose()

    def preprocess_data(self, data: ndarray) -> ndarray:
        data = self.butter_bandpass_filter(data)
        return data

    def spectral_analysis(self, data: ndarray) -> Tuple[ndarray, ndarray, float]:
        band_width = 0.5

        num_samples = data.shape[-1]
        fft_data = np.fft.fft(data, axis=1)
        fft_magnitude = np.abs(fft_data[0])
        frequencies = np.fft.fftfreq(num_samples, 1 / self.real_time_store.sfreq)

        max_magnitude = 0
        most_prominent_frequency = None

        for target_frequency in self.real_time_store.target_frequencies:
            # Find the indices of the FFT bins within the band around the target frequency
            band_indices = np.where((frequencies >= target_frequency - band_width) &
                                    (frequencies <= target_frequency + band_width))[0]

            # Sum the magnitudes of these bins
            band_magnitude = np.sum(fft_magnitude[band_indices])

            # Check if this is the most prominent frequency so far
            if band_magnitude > max_magnitude:
                max_magnitude = band_magnitude
                most_prominent_frequency = target_frequency

        frequencies = frequencies[:num_samples // 2]
        magnitudes = 2.0 / num_samples * np.abs(fft_data[:, :num_samples // 2])

        return frequencies, magnitudes, most_prominent_frequency

    def process_data(self, data: ndarray) -> Tuple[ndarray, ndarray, float]:
        data = self.preprocess_data(data)
        return self.spectral_analysis(data)

    def work(self):
        iteration: int = 0

        source_id = self.real_time_store.source_id
        self.logger.info(f"Start Real-Time SSVEP Decoding for Stream: {source_id}")

        while self.recorder_store.is_recording:
            data = self.real_time_store.get_data()
            if data is not None:
                data = self.reshape_data(data)
                x, y, max_freq = self.process_data(data)
                print(max_freq)
                with open(f"./data/real_time/data.npy", 'wb') as f:
                    np.save(f, x)
                    np.save(f, y)
            iteration += 1
            time.sleep(1 / 10)


class RealTimeVisualizer(RealTimeWorker):

    def __init__(self,
                 recorder_store: RecorderStore,
                 real_time_store: RealTimeStore
                 ):
        super().__init__(recorder_store, real_time_store)

        self.process: Thread = self.get_new_process()
        self.band_pass_coefficients = self.butter_bandpass(1, 6, 250, 5)

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data):
        y = lfilter(self.band_pass_coefficients[0], self.band_pass_coefficients[1], data, axis=1)
        return y

    def reshape_data(self, data: ndarray) -> ndarray:
        return data.transpose()

    def preprocess_data(self, data: ndarray) -> ndarray:
        data = self.butter_bandpass_filter(data)
        return data

    def spectral_analysis(self, data: ndarray) -> ndarray:
        data = np.fft.fft(data, axis=1)
        return data

    def process_data(self, data: ndarray) -> ndarray:
        data = self.preprocess_data(data)
        return data

    def work(self):
        iteration: int = 0

        while self.recorder_store.is_recording:
            data = self.real_time_store.get_data()
            if data is not None:
                data = self.reshape_data(data)
                data = self.process_data(data)
                x = np.arange(data.shape[-1])
                with open(f"./data/real_time/data.npy", 'wb') as f:
                    np.save(f, x)
                    np.save(f, data)
            iteration += 1
            time.sleep(1 / 10)
