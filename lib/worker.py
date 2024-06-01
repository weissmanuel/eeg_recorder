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


class Worker(ABC):

    @abstractmethod
    def work(self):
        pass


class RecordingWorker(Worker):
    logger = logging.getLogger(__name__)

    stream: Union[StreamLSL, None] = None
    process: Process

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

        self.process = Process(target=self.work)

    @property
    def buffer_size(self) -> int:
        if self.stream is not None:
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

    def connect(self):
        self.logger.debug(f"Start Connecting to LSL Stream: {self.source_id} of type {self.stream_type.value}")
        try:
            self.stream = StreamLSL(bufsize=self.buffer_size_seconds, source_id=self.source_id)
            self.stream.connect()
            self.logger.info(f"Connected to LSL Stream: {self.source_id} of type {self.stream_type.value}")
        except Exception as e:
            self.logger.warning(f"Failed to connect to LSK Streams {self.source_id}: {e}")
            self.stream_store.recording_completed = True
            self.stream = None

    def disconnect(self):
        if self.stream is not None:
            self.stream.disconnect()
            self.stream = None

    def start(self):
        self.process.start()

    def stop(self):
        self.process.terminate()
        self.process.join()

    def reset(self):
        self.stream_store.reset()

    def add_data(self, data: any, times: any, recording_stopped_at: Union[float, None] = None):
        self.stream_store.append_data(data.copy())
        self.stream_store.append_times(times.copy())

        self.stream_store.last_batch_received_time = times[-1]
        self.stream_store.current_time = recording_stopped_at if recording_stopped_at is not None else local_clock()
        self.stream_store.signal_time_shift = self.stream_store.current_time - self.stream_store.last_batch_received_time

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

    def work(self):
        if self.stream is not None and self.stream.connected:

            self.logger.info(f"Start Recording of Stream: {self.stream_store.source_id}")
            values: List[ndarray] = []
            times: List[ndarray] = []

            sfreq: float = self.stream.info['sfreq']
            self.stream_store.sfreq = sfreq
            self.stream_store.time_shift = 0.0
            self.stream_store.start_time_seconds = local_clock()

            recording_stopped_at: Union[float, None] = None

            while (self.recorder_store.is_recording
                   or self.stream_store.time_shift > -self.stream_store.safety_offset_seconds):

                if not self.recorder_store.is_recording and recording_stopped_at is None:
                    recording_stopped_at = local_clock()

                if sfreq is not None and sfreq > 0:
                    window_size = self.stream.n_new_samples / sfreq
                else:
                    window_size = self.stream.n_new_samples

                if window_size > 0:
                    (values, times) = self.stream.get_data(winsize=window_size)

                    if values is not None and len(values) > 0:
                        self.add_data(values, times, recording_stopped_at)

                        if self.stream_store.iterations == 0:
                            self.log_first_iteration(times)

                        self.stream_store.increment_iterations()
                time.sleep(0.01)

            self.stream_store.end_time_seconds = (recording_stopped_at if
                                                  recording_stopped_at is not None else local_clock())

            if self.stream_store.sample_deviation > 0:
                self.logger.warning(f"Recorded less samples than expected: {self.stream_store.sample_deviation} samples")

            self.log_recording_completed()
            # np.abs(np.abs(np.diff(np.abs(self.signal_times - start_time))) - 1 / sfreq)

            self.logger.debug(f"Finished Signal Recording for Stream: {self.source_id}")
            self.stream_store.recording_completed = True
