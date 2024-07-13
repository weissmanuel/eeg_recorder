import copy
import os.path
from abc import ABC, abstractmethod
from .store import StreamType
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
from lib.store import RecorderStore, StreamStore, RealTimeStore, PlotStore
import time
from lib.persist import Persister
import numpy as np
import threading
from threading import Thread
from typing import Tuple
from multiprocessing import Lock
import dearpygui.dearpygui as dpg
from lib.mne import create_raw, create_info
from omegaconf import DictConfig
from mne.io import RawArray
from mne import Info
from lib.preprocess.data_preprocess import get_preprocessors, Preprocessor
from lib.preprocess.raw_preprocess import get_raw_preprocessors, RawPreprocessor
from lib.preprocess.rt_preprocessor import get_rt_preprocessors, RTPreprocessor
from collections import deque
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from lib.train.pipeline import ThresholdingDecoder, load_pipeline
from lib.utils import generate_demo_data
from lib.preprocess.models import ProcessStage
from enum import Enum


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
    def work(self, lock: Lock):
        pass


class ProcessWorker(Worker):
    lock: Lock

    def __init__(self, lock: Lock):
        self.lock = lock

    def get_new_process(self):
        return Process(target=self.work, args=(self.lock,))

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
    def work(self, lock: Lock):
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
    def work(self, lock: Lock):
        pass


class RecordingWorker(ProcessWorker):
    recorder_store: RecorderStore
    stream_store: StreamStore

    buffer_size_seconds: float

    def __init__(self,
                 lock: Lock,
                 recorder_store: RecorderStore,
                 stream_store: StreamStore,
                 buffer_size_seconds: float
                 ):

        super().__init__(lock)

        self.recorder_store = recorder_store
        self.stream_store = stream_store

        self.buffer_size_seconds = buffer_size_seconds

        self.process = self.get_new_process()
        self.sleep_time = 0.01

    def buffer_size(self, stream: StreamLSL) -> int:
        if stream is not None:
            sfreq = stream.info['sfreq']
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
        self.lock.acquire()
        self.stream_store.append_data(data.copy())
        self.stream_store.append_times(times.copy())
        self.lock.release()

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
            delta_seconds = float(self.stream_store.times[-1] - self.stream_store.times[0])
            self.stream_store.last_sample_datetime = (self.stream_store.first_sample_datetime +
                                                      timedelta(seconds=delta_seconds))
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

    def work(self, lock: Lock):

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
                            self.log_first_iteration(times.tolist())

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
                 lock: Lock,
                 interval: int,
                 recorder_store: RecorderStore,
                 stream_stores: List[StreamStore],
                 persisters: List[Persister]
                 ):
        super().__init__(lock)

        self.interval = interval

        self.recorder_store = recorder_store
        self.stream_stores = stream_stores

        self.persisters = persisters

        self.process = self.get_new_process()

    def persist(self):
        for persister in self.persisters:
            persister.save(self.stream_stores, intermediate_save=True)

    def work(self, lock: Lock):
        while self.recorder_store.is_recording:
            time.sleep(self.interval)
            self.recorder_store.pause_recording()
            self.lock.acquire()
            self.persist()
            self.lock.release()
            self.recorder_store.resume_recording()


class RealTimeWorker(ProcessWorker):

    def __init__(self,
                 lock: Lock,
                 recorder_store: RecorderStore,
                 real_time_store: RealTimeStore
                 ):
        super().__init__(lock)

        self.recorder_store = recorder_store
        self.real_time_store = real_time_store

    def work(self, lock: Lock):
        pass


class _RealTimeRecorderMixin:

    def __init__(self, real_time_store: RealTimeStore):
        self.real_time_store = real_time_store

    def get_stream_data(self, stream: StreamLSL) -> Tuple[ndarray, float] | Tuple[None, None]:
        window_size = stream.n_new_samples / self.real_time_store.sfreq
        if window_size > 0:
            (values, times) = stream.get_data(winsize=window_size)
            last_time = times[-1] if times is not None and len(times) > 0 else local_clock()
            return values, last_time
        else:
            return None, None

    def get_data(self, stream: StreamLSL) -> ndarray | None:
        if self.real_time_store.source_id == 'demo':
            target_frequencies = self.real_time_store.labels if self.real_time_store.labels is not None else None
            return generate_demo_data(self.real_time_store, num_channels=2, target_frequencies=target_frequencies)
        else:
            return self.get_stream_data(stream)


class RealTimeRecorder(RealTimeWorker, _RealTimeRecorderMixin):

    def __init__(self,
                 lock: Lock,
                 recorder_store: RecorderStore,
                 real_time_store: RealTimeStore
                 ):
        super().__init__(lock, recorder_store, real_time_store)

        self.process: Process = self.get_new_process()

    def work(self, lock: Lock):
        source_id = self.real_time_store.source_id
        stream = connect(source_id, self.real_time_store.stream_type, self.real_time_store.buffer_size_seconds)

        if (stream is not None and stream.connected) or source_id == 'demo':

            self.logger.info(f"Start Real-Time Recording for Stream: {source_id}")
            while self.recorder_store.is_recording:
                data, last_timestep = self.get_data(stream)
                if data is not None:
                    self.lock.acquire()
                    self.real_time_store.add_data(data.T)
                    self.real_time_store.add_times(last_timestep, local_clock())
                    self.lock.release()


class RealTimeRecordingMode(Enum):
    DECODER = 'DECODER'
    RECORDER = 'RECORDER'

    def __str__(self):
        return self.value()

    @staticmethod
    def from_str(label: str) -> 'RealTimeRecordingMode':
        return RealTimeRecordingMode[label.upper()]


class RealTimeDecoder(RealTimeWorker, _RealTimeRecorderMixin):
    stream: StreamLSL | None

    def __init__(self,
                 recorder_lock: Lock,
                 recorder_store: RecorderStore,
                 real_time_store: RealTimeStore,
                 visualizer_lock: Lock,
                 plot_store: PlotStore,
                 config: DictConfig,
                 recording_mode: RealTimeRecordingMode = RealTimeRecordingMode.DECODER
                 ):
        super().__init__(recorder_lock, recorder_store, real_time_store)

        self.recorder_lock = recorder_lock
        self.visualizer_lock = visualizer_lock
        self.plot_store = plot_store
        self.config = config
        self.recording_mode = recording_mode

        self.preprocessors: List[Preprocessor] = get_preprocessors(self.config.experiment.preprocessors,
                                                                   stages=[ProcessStage.INFERENCE])
        self.raw_preprocessors: List[RawPreprocessor] = get_raw_preprocessors(self.config.experiment.raw_preprocessors,
                                                                              stages=[ProcessStage.INFERENCE])
        self.rt_preprocessors: List[RTPreprocessor] = get_rt_preprocessors(self.config.experiment.rt_preprocessors)
        self.decoder = self.init_decoder()
        self.queue = deque(maxlen=self.real_time_store.visualisation_window_size)

    def init_recording(self):
        if self.recording_mode == RealTimeRecordingMode.DECODER:
            source_id = self.real_time_store.source_id
            self.stream = connect(source_id, self.real_time_store.stream_type, self.real_time_store.buffer_size_seconds)
            self.logger.info(f"Start Real-Time SSVEP Decoding for Stream: {source_id}")

    def init_decoder(self) -> Pipeline | BaseEstimator | None:
        model_type = self.config.experiment.decoding.type
        # ToDo: Implement as usual trained pipeline
        if model_type == 'threshold':
            return ThresholdingDecoder(sfreq=self.real_time_store.sfreq,
                                       target_frequencies=self.config.experiment.labels,
                                       channel=self.config.headset.target_channel,
                                       band_width=1.0)
        elif model_type == 'model':
            if self.config.experiment.decoding.decoder_path is not None and os.path.exists(
                    self.config.experiment.decoding.decoder_path):
                return load_pipeline(self.config.experiment.decoding.decoder_path)
        else:
            return None

    def retrieve_direct_data(self) -> Tuple[ndarray, float, float] | Tuple[None, None, None]:
        data, last_sample_time = self.get_data(self.stream)
        last_received_time = local_clock()
        if data is not None:
            data = self.rt_preprocess(data)
            self.queue.extend(data.T.tolist())
            if len(self.queue) >= self.real_time_store.visualisation_window_size:
                return np.array(self.queue).T, last_sample_time, last_received_time
        return None, None, None

    def retrieve_recorder_data(self) -> Tuple[ndarray, float, float] | Tuple[None, None, None]:
        self.recorder_lock.acquire()
        data = self.real_time_store.get_data()
        last_sample_time = self.real_time_store.last_sample_time
        last_received_time = self.real_time_store.last_received_time
        self.recorder_lock.release()
        if data is not None:
            data = data.T
            return data, last_sample_time, last_received_time
        return None, None, None

    def retrieve_data(self) -> Tuple[ndarray, float, float] | Tuple[None, None, None]:
        if self.recording_mode == RealTimeRecordingMode.DECODER:
            return self.retrieve_direct_data()
        else:
            return self.retrieve_recorder_data()

    def rt_preprocess(self, data: ndarray) -> ndarray:
        if self.rt_preprocessors is not None and len(self.rt_preprocessors) > 0:
            for preprocessor in self.rt_preprocessors:
                data = preprocessor(data)
        return data

    def preprocess(self, info: Info, data: ndarray) -> ndarray:
        if self.preprocessors is not None and len(self.preprocessors) > 0:
            for preprocessor in self.preprocessors:
                data = preprocessor(data, info=info)
        return data

    def preprocess_raw(self, raw: RawArray) -> RawArray:
        if self.raw_preprocessors is not None and len(self.raw_preprocessors) > 0:
            for preprocessor in self.raw_preprocessors:
                raw = preprocessor(raw)
        return raw

    def preprocess_data(self, data: ndarray) -> RawArray:
        info = create_info(self.config)
        data = self.preprocess(info=info, data=data)
        raw = create_raw(data=data, info=info)
        raw = raw.pick(picks=['eeg'])
        raw = self.preprocess_raw(raw)
        return raw


class RealTimeSSVEPDecoder(RealTimeDecoder):

    def __init__(self,
                 recorder_lock: Lock,
                 recorder_store: RecorderStore,
                 real_time_store: RealTimeStore,
                 visualizer_lock: Lock,
                 plot_store: PlotStore,
                 config: DictConfig,
                 spectral_average: int | None = None,
                 recording_mode: RealTimeRecordingMode = RealTimeRecordingMode.DECODER
                 ):
        super().__init__(recorder_lock=recorder_lock,
                         recorder_store=recorder_store,
                         real_time_store=real_time_store,
                         visualizer_lock=visualizer_lock,
                         plot_store=plot_store,
                         config=config,
                         recording_mode=recording_mode)

        self.process: Process = self.get_new_process()

        self.spectral_average = spectral_average
        self.spectral_queue = deque(maxlen=spectral_average) if spectral_average is not None else None

        self.labels = config.experiment.labels
        self.recorder_lock = recorder_lock
        self.visualizer_lock = visualizer_lock

    def spectral_analysis(self, data: ndarray, channel: int = 0) -> Tuple[ndarray, ndarray]:

        data = data[channel]

        num_samples = data.shape[-1]
        fft_data = np.fft.fft(data)
        frequencies = np.fft.fftfreq(num_samples, 1 / self.real_time_store.sfreq)

        frequencies = frequencies[:num_samples // 2]
        magnitudes = 2.0 / num_samples * np.abs(fft_data[:num_samples // 2])

        if self.spectral_average is not None:
            self.spectral_queue.append(magnitudes)
            magnitudes = np.mean(self.spectral_queue, axis=0)

        return magnitudes, frequencies

    def compute_psd(self, raw: RawArray) -> Tuple[ndarray, ndarray]:
        psd_conf = self.config.psd
        method = psd_conf.method
        if self.config.psd.method == 'scipy':
            return self.spectral_analysis(raw.get_data(), self.config.headset.target_channel)
        else:
            amps, freqs = (raw.compute_psd(method=method, **psd_conf.vis_kwargs, verbose=False)
                           .get_data(return_freqs=True, fmin=psd_conf.vis_kwargs.fmin, fmax=psd_conf.vis_kwargs.fmax))
            return amps[self.config.headset.target_channel], freqs

    def get_time_axis(self, data: ndarray) -> ndarray:
        sfreq = self.real_time_store.sfreq
        n_times = data.shape[-1]
        times = np.arange(0, n_times)
        times = np.round(times / sfreq, 4)
        return times

    def assign_time_data(self, data: ndarray, channel: int = 0):
        self.plot_store.x_time = self.get_time_axis(data).tolist()
        self.plot_store.y_time = data[channel].tolist()

    def predict(self, data: ndarray) -> float:
        if False:
            y_pred = self.decoder.predict(data)
            if len(y_pred) == 1:
                return self.labels[int(y_pred[0])]
            else:
                raise ValueError('Currently only one prediction is supported')
        else:
            return 0

    def assign_times(self, last_sample_time: float, last_received_time: float):
        self.plot_store.set_times(last_sample_time, last_received_time, local_clock())

    def process_data(self, data: ndarray) -> Tuple[ndarray, ndarray, float]:
        raw = self.preprocess_data(data)
        data = raw.get_data()
        self.assign_time_data(data)
        amps, freqs = self.compute_psd(raw)
        data = np.expand_dims(data, axis=0)
        result = self.predict(data[:, :, -self.real_time_store.window_size:])
        return freqs, amps, result

    def work(self, lock: Lock):
        self.init_recording()
        while self.recorder_store.is_recording:
            data, last_sample_time, last_received_time = self.retrieve_data()
            if data is not None:
                # self.visualizer_lock.acquire()
                freqs, amps, result = self.process_data(data)
                self.plot_store.set_freq_data(freqs, amps, result)
                self.assign_times(last_sample_time, last_received_time)
                # self.visualizer_lock.release()


class RealTimeVisualizer(RealTimeWorker):

    def __init__(self,
                 lock: Lock,
                 recorder_store: RecorderStore,
                 real_time_store: RealTimeStore,
                 plot_store: PlotStore,
                 config: DictConfig
                 ):
        super().__init__(lock, recorder_store, real_time_store)

        self.process: Thread = self.get_new_process()
        self.plot_store = plot_store
        self.plotting: bool = False

        vis_config = config.experiment.visualisation

        self.freq_x_limits = vis_config.freq_range
        self.freq_y_limits = vis_config.freq_y_limits
        self.freq_normalize = vis_config.freq_normalize
        self.freq_scale = vis_config.freq_scale
        self.freq_x_label = vis_config.freq_x_label
        self.freq_y_label = vis_config.freq_y_label

        self.time_y_limits = vis_config.time_y_limits
        self.time_normalize = vis_config.time_normalize
        self.time_x_label = vis_config.time_x_label
        self.time_y_label = vis_config.time_y_label

        self.config = config
        self.last_freq_max = 0
        self.last_y_time_max = 0
        self.last_x_time_max = 0

    def work(self, lock: Lock):
        dpg.create_context()

        def update_freq_data():
            x_freq, y_freq, max_freq = copy.copy(self.plot_store.get_freq_data())

            if x_freq is not None and y_freq is not None and len(x_freq) > 0 and len(y_freq) > 0:
                if self.freq_y_limits is None:
                    y_freq_max = np.max(y_freq)
                    if y_freq_max > self.last_freq_max * 1.1 or y_freq_max < self.last_freq_max * 0.9:
                        self.last_freq_max = y_freq_max
                        dpg.set_axis_limits("y_freq", 0, y_freq_max * 1.1)
                dpg.set_value('freq_series', [x_freq, y_freq])
                dpg.set_value('result_text', max_freq)

        def update_time_data():
            x_time, y_time = copy.copy(self.plot_store.get_time_data())
            if x_time is not None and y_time is not None and len(x_time) > 0 and len(y_time) > 0:
                if self.time_y_limits is None:
                    y_time_max = np.max(y_time)
                    if y_time_max > self.last_y_time_max * 1.1 or y_time_max < self.last_y_time_max * 0.9:
                        self.last_y_time_max = y_time_max
                        dpg.set_axis_limits("y_time", np.min(y_time) * 0.9, y_time_max * 1.1)
                dpg.set_value('time_series', [x_time, y_time])

        def update_metadata():
            received_delay, processing_delay, total_delay = copy.copy(self.plot_store.get_delays())
            dpg.set_value('total_delay_text', np.round(total_delay, 3))
            dpg.set_value('receive_delay_text', np.round(received_delay, 3))
            dpg.set_value('processing_delay_text', np.round(processing_delay, 3))

        def update_series():
            while dpg.is_dearpygui_running() and self.recorder_store.is_recording:
                update_freq_data()
                update_time_data()
                update_metadata()
                time.sleep(0.01)

        with dpg.window(label="Spectral Analysis", tag="win_feq", pos=[0, 0], width=400, height=400):
            with dpg.plot(label="Spectral Analysis", height=400, width=400):
                dpg.add_plot_legend()

                dpg.add_plot_axis(dpg.mvXAxis, label=self.freq_x_label, tag='x_freq')
                dpg.add_plot_axis(dpg.mvYAxis, label=self.freq_y_label, tag="y_freq")

                if self.freq_y_limits is not None:
                    dpg.set_axis_limits("y_freq", self.freq_y_limits[0], self.freq_y_limits[1])

                if self.freq_x_limits is not None:
                    dpg.set_axis_limits("x_freq", self.freq_x_limits[0], self.freq_x_limits[1])

                dpg.add_line_series(self.plot_store.x_freq, self.plot_store.y_freq, label="Spectral Analysis",
                                    parent="y_freq", tag="freq_series")

        with dpg.window(label="Result", tag="win_result", pos=[400, 0], width=200, height=400):
            dpg.add_text("Max Frequency", tag='result_title')
            dpg.add_text("0", tag='result_text')
            dpg.add_text("Total Delay", tag='total_delay_title')
            dpg.add_text("0", tag='total_delay_text')
            dpg.add_text("Receive Delay", tag='receive_delay_title')
            dpg.add_text("0", tag='receive_delay_text')
            dpg.add_text("Processing Delay", tag='processing_delay_title')
            dpg.add_text("0", tag='processing_delay_text')

        with dpg.window(label="EEG Data", tag="win_time", width=600, height=450, pos=[0, 400]):
            with dpg.plot(label="EEG Data", width=600, height=400):
                dpg.add_plot_legend()

                dpg.add_plot_axis(dpg.mvXAxis, label=self.time_x_label, tag='x_time')
                dpg.set_axis_limits("x_time", 0, self.config.experiment.visualisation.window_size_seconds)

                dpg.add_plot_axis(dpg.mvYAxis, label=self.time_y_label, tag="y_time")
                if self.time_y_limits is not None:
                    y_limits = self.time_y_limits
                    dpg.set_axis_limits("y_time", y_limits[0], y_limits[1])

                dpg.add_line_series(self.plot_store.x_time, self.plot_store.x_time, label="EEG Signal", parent="y_time",
                                    tag="time_series")

        thread = threading.Thread(target=update_series)
        thread.start()

        dpg.create_viewport(title='EEG Visualizer', width=620, height=900)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
