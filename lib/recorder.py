import mne
from mne.io import RawArray
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
from typing import Tuple, Union, List
from numpy import ndarray
import time
from lib.utils import format_seconds
from omegaconf import DictConfig
from lib.utils import config_to_primitive
from mne import Info
from lib.preprocess import get_preprocessors, Preprocessor
from multiprocessing import Manager, Lock
from lib.worker import RecordingWorker, PersistenceWorker, Worker, RealTimeRecorder, RealTimeWorker, RealTimeSSVEPDecoder
from lib.store import StreamType, StreamStore, RecorderStore, RealTimeStore, PlotStore
from lib.persist import MneRawPersister, PersistingMode


class InletInfo:
    source_id: str | None
    sfreq: float | None
    n_channels: int | None
    iterations: int | None
    time_shift: float | None
    samples_recorded: int | None
    samples_expected: int | None

    def __init__(self,
                 source_id: str | None = None,
                 sfreq: float | None = None,
                 n_channels: int | None = None,
                 iterations: int | None = None,
                 time_shift: float | None = None,
                 samples_recorded: int | None = None,
                 samples_expected: int | None = None):
        self.source_id = source_id
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.iterations = iterations
        self.time_shift = time_shift
        self.samples_recorded = samples_recorded
        self.samples_expected = samples_expected


class RecordingInfo:
    start_time: datetime
    end_time: datetime
    duration: timedelta

    signal_info: InletInfo
    marker_info: InletInfo

    file_path: str

    def __init__(self,
                 start_time: datetime | None = None,
                 end_time: datetime | None = None,
                 duration: timedelta | float | None = None,
                 signal_info: InletInfo | None = None,
                 marker_info: InletInfo | None = None,
                 file_path: str | None = None):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.signal_info = signal_info
        self.marker_info = marker_info
        self.file_path = file_path


class Recorder:
    sources: List[Tuple[str, str]] = []
    recorders: List[RecordingWorker] = []

    real_time_workers: List[RealTimeWorker] = []

    recorder_store: RecorderStore
    real_time_store: RealTimeStore
    plot_store: PlotStore | None = None

    safety_offset_seconds: float = 1.0

    log_level: int = logging.INFO
    logger = logging.getLogger(__name__)

    manager: Manager
    persister: MneRawPersister | None
    persister_workers: List[PersistenceWorker] = []

    def __init__(self,
                 config: Union[dict | DictConfig],
                 sources: List[Union[Tuple[str, str], List[str]]] | None = None,
                 buffer_size_seconds: float = 60.0,
                 ):
        logging.basicConfig(level=self.log_level)
        self.config = config

        self.sources = sources if sources is not None else []
        self.buffer_size_seconds = buffer_size_seconds

        self.manager = Manager()
        self.lock = Lock()
        # self.persister = MneRawPersister(config=config)
        self.persister = None
        self.initialise_recorders()
        self.initialise_persisters(config)
        self.initialise_real_time(config)
        self.logger.info("Recorder Initialised")

    def initialise_recorders(self):
        self.recorder_store = RecorderStore(self.manager)
        for source in self.sources:
            source_id, stream_type = source
            stream_type = StreamType.from_str(stream_type)
            stream_store = StreamStore(self.manager, source_id, stream_type)
            recorder = RecordingWorker(self.lock, self.recorder_store, stream_store, self.buffer_size_seconds)
            self.recorders.append(recorder)

    def initialise_persisters(self, config: DictConfig):
        if 'persister_workers' in config and config.persister_workers is not None:
            stream_stores = [recorder.stream_store for recorder in self.recorders]
            for persister_config in config.persister_workers:

                persister_worker = PersistenceWorker(lock=self.lock,
                                                     interval=persister_config.interval,
                                                     recorder_store=self.recorder_store,
                                                     stream_stores=stream_stores,
                                                     persister=self.persister)
                self.persister_workers.append(persister_worker)

    def initialise_real_time(self, config: DictConfig):
        if 'real_time' in config and config.real_time is not None:
            self.real_time_store = RealTimeStore.from_config(config.real_time, self.manager)
            self.plot_store = PlotStore(self.manager, 'Test Plot', 'Frequencies', 'Amplitude')
            self.real_time_workers.append(RealTimeRecorder(self.lock, self.recorder_store, self.real_time_store))
            self.real_time_workers.append(RealTimeSSVEPDecoder(self.lock, self.recorder_store, self.real_time_store, plot_store=self.plot_store))
            # self.real_time_workers.append(RealTimeVisualizer(self.recorder_store, self.real_time_store))



    @property
    def is_recording(self) -> bool:
        return self.recorder_store.is_recording

    @property
    def recording_completed(self) -> bool:
        return all([recorder.recording_completed for recorder in self.recorders])

    def reset_recording(self):
        self.logger.info("Resetting Recorder")
        self.recorder_store.reset()
        for recorder in self.recorders:
            recorder.reset()
        if self.persister is not None and self.persister.persisting_mode == PersistingMode.CONTINUOUS:
            self.persister.delete()

    def get_workers(self) -> List[Worker]:
        return self.recorders + self.persister_workers + self.real_time_workers

    def start(self):
        if not self.is_recording:
            self.reset_recording()
            self.logger.info("Starting Recording")
            self.recorder_store.start()

            for worker in self.get_workers():
                worker.start()

    def stop(self):
        if self.is_recording:
            self.logger.info("Stopping Recording")
            self.recorder_store.stop()
            self.recorder_store.recording_end_time = datetime.utcnow()
            while not self.recording_completed:
                time.sleep(1)
            self.summary()
            self.logger.info("Recording Stopped")

    def get_signal_recorder(self) -> RecordingWorker:
        return next((recorder for recorder in self.recorders if recorder.stream_type.is_signal), None)

    def get_marker_recorder(self) -> RecordingWorker:
        return next((recorder for recorder in self.recorders if recorder.stream_type.is_marker), None)

    def has_signal_stream(self) -> bool:
        return self.get_signal_recorder().stream_store.has_stream

    def has_marker_stream(self) -> bool:
        return self.get_marker_recorder().stream_store.has_stream

    def filter_recorders(self, stream_type: StreamType) -> List[RecordingWorker]:
        return [recorder for recorder in self.recorders if recorder.stream_type == stream_type]

    def create_mne_info(self) -> Info:
        signal_recorder = self.get_signal_recorder()
        info = signal_recorder.stream_store.stream_info.copy()
        if 'channel_names_mapping' in self.config.headset:
            info.rename_channels(config_to_primitive(self.config.headset.channel_names_mapping))
        if 'channel_types_mapping' in self.config.headset:
            info.set_channel_types(config_to_primitive(self.config.headset.channel_types_mapping))
        if 'montage' in self.config.headset:
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage)
        return info

    def preprocess(self, info: Info, data: ndarray) -> ndarray:
        preprocessors: List[Preprocessor] = get_preprocessors(self.config.preprocessors)
        if preprocessors is not None and len(preprocessors) > 0:
            for preprocessor in preprocessors:
                data = preprocessor(info, data)
        return data

    def add_annotations(self, raw: RawArray, marker_store: StreamStore) -> RawArray:
        if marker_store is not None and marker_store.stream_type == StreamType.MARKER and marker_store.n_samples > 0:
            signal_recorder: RecordingWorker = self.get_signal_recorder()
            first_signal_lsl_seconds = signal_recorder.stream_store.first_sample_lsl_seconds
            marker_values = marker_store.data
            marker_times = marker_store.times - first_signal_lsl_seconds
            raw.set_annotations(mne.Annotations(onset=marker_times, duration=[0.05] * len(marker_times),
                                                description=marker_values.astype(str)))
        return raw

    def get_raw(self):
        assert not self.is_recording, "You cannot generate an MNE Raw object while recording"
        signal_recorder: RecordingWorker = self.get_signal_recorder()
        signal_store: StreamStore = signal_recorder.stream_store
        signal = signal_store.data
        assert signal_store.n_samples > 0, "No signal data recorded"
        self.logger.info("Generating MNE Raw Object")
        info = self.create_mne_info()
        signal = self.preprocess(info, signal)
        raw = RawArray(signal, info)

        for recorder in self.filter_recorders(StreamType.MARKER):
            raw = self.add_annotations(raw, recorder.stream_store)

        raw.set_meas_date(signal_store.first_sample_datetime.replace(tzinfo=timezone.utc).timestamp())

        if len(raw.info['device_info']) == 0:
            raw.info['device_info'] = None
        self.logger.info("MNE Raw Object Generated")
        return raw

    def get_file_path(self):
        if 'recording' in self.config:
            subject = self.config.recording.subject
            session = self.config.recording.session
            block = self.config.recording.block
            return f"./data/recordings/recoding_subject_{subject}_session_{session}_block_{block}.fif"

    def save(self, file_path: Union[str, None] = None) -> Tuple[RawArray, Path]:
        assert not self.is_recording, "You cannot generate an MNE Raw object while recording"
        stores = [recorder.stream_store for recorder in self.recorders]
        return self.persister.save(stores, file_path)

    def complete(self, file_path: Union[str, None] = None) -> Tuple[Union[RawArray, None], RecordingInfo]:
        if self.is_recording:
            self.stop()
            if self.persister is not None:
                raw, path = self.save(file_path)
                info = self.get_info()
                info.file_path = path
                self.logger.info("Recording Completed")
                for worker in self.get_workers():
                    worker.stop()
                return raw, info
        return None, RecordingInfo()

    def get_stream_info(self, recorder: RecordingWorker) -> InletInfo:
        stream_store = recorder.stream_store
        return InletInfo(source_id=stream_store.source_id,
                         sfreq=stream_store.sfreq,
                         n_channels=stream_store.n_channels,
                         iterations=stream_store.iterations,
                         time_shift=stream_store.time_shift,
                         samples_recorded=stream_store.n_samples,
                         samples_expected=stream_store.expected_samples)

    def get_info(self) -> RecordingInfo:
        signal_info = InletInfo()
        if self.has_signal_stream():
            signal_info = self.get_stream_info(self.get_signal_recorder())

        marker_info = InletInfo()
        if self.has_marker_stream():
            marker_info = self.get_stream_info(self.get_marker_recorder())

        return RecordingInfo(start_time=self.recorder_store.recording_start_time,
                             end_time=self.recorder_store.recording_end_time,
                             duration=self.recorder_store.recording_duration,
                             signal_info=signal_info,
                             marker_info=marker_info,
                             file_path=None)

    def stream_summary(self, recorder: RecordingWorker):
        if recorder.stream_store.has_stream is not None:
            stream_store = recorder.stream_store
            print(f"Steam: {stream_store.source_id}")
            print(f"Stream Type: {stream_store.stream_type}")
            print(f"Number of Channels: {stream_store.n_channels}")
            print(f"Sampling Frequency: {stream_store.sfreq}")
            print(f"First Sample Time: {stream_store.first_sample_datetime}")
            print(f"Last Sample Time: {stream_store.last_sample_datetime}")
            print(f"Recording Duration: {format_seconds(stream_store.duration)}")
            print(f"Number of Samples: {stream_store.n_samples} \n")

    def summary(self):
        print("--------------------------------------------------------------------------------------")
        print(f"Recording Summary")
        print("--------------------------------------------------------------------------------------", "\n")
        print(f"Recording Started at: {self.recorder_store.recording_start_time}")
        print(f"Recording Ended at: {self.recorder_store.recording_end_time}")
        print(f"Recoding Duration: {self.recorder_store.recording_duration} \n")

        for recorder in self.recorders:
            self.stream_summary(recorder)
        print("--------------------------------------------------------------------------------------")
