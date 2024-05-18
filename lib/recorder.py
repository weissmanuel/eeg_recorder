from mne_lsl.lsl import local_clock, StreamInlet, resolve_streams
from mne_lsl.stream import StreamLSL
from threading import Thread
import math
import numpy as np
import mne
from mne.io import RawArray
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
from typing import Tuple, Union, List, Callable
from numpy import ndarray
import time
from lib.utils import format_seconds


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
    signal_id: Union[str, None]
    marker_id: Union[str, None]

    safety_offset_seconds: float = 1.0

    signal_stream: Union[StreamLSL, None] = None
    marker_stream: Union[StreamInlet, None] = None

    signal_values: ndarray = []
    signal_times: ndarray = []
    first_signal_lsl_seconds: float = 0
    first_signal_system_seconds: float = 0

    marker_values: ndarray = []
    marker_times: ndarray = []
    first_marker_lsl_seconds: float = 0
    first_marker_system_seconds: float = 0

    is_recording: bool = False
    _signal_thread: Thread
    _marker_thread: Thread
    signal_recording_completed: bool = False
    marker_recording_completed: bool = False

    log_level: int = logging.INFO
    logger = logging.getLogger(__name__)

    recording_start_time: datetime | None = None
    recording_end_time: datetime | None = None
    first_signal_datetime: datetime | None = None
    last_signal_datetime: datetime | None = None
    first_marker_datetime: datetime | None = None
    last_marker_datetime: datetime | None = None

    signal_time_shift: float | None
    marker_time_shift: float | None

    def __init__(self, signal_id: str, marker_id: str | None, buffer_size_seconds: float):
        logging.basicConfig(level=self.log_level)
        self.signal_id = signal_id
        self.marker_id = marker_id

        self.buffer_size_seconds = buffer_size_seconds

    @property
    def buffer_size(self) -> int:
        if self.signal_stream is not None:
            sfreq = self.signal_stream.info['sfreq']
            return math.ceil(self.buffer_size_seconds * sfreq)
        return 0

    @property
    def recording_completed(self) -> bool:
        return self.signal_recording_completed and self.marker_recording_completed

    def connect(self):
        self.logger.debug("Connecting to LSL Streams")
        if self.signal_id is not None:
            self.signal_stream = StreamLSL(bufsize=self.buffer_size_seconds, source_id=self.signal_id)
            self.signal_stream.connect(processing_flags=['clocksync', 'dejitter', 'monotize'])
            self.logger.debug(f"Signal Stream Connected with id: {self.signal_id}")
        else:
            self.signal_recording_completed = True

        if self.marker_id is not None:
            streams = resolve_streams(source_id=self.marker_id)
            if len(streams) == 1:
                self.marker_stream = StreamInlet(streams[0])
                self.marker_stream.open_stream()
                self.logger.debug(f"Marker Stream Connected with id: {self.marker_id}")
            else:
                self.logger.info("No marker Stream Connected -> Ignoring Marker Recording")
                self.marker_recording_completed = True
        else:
            self.marker_recording_completed = True

    def disconnect(self):
        self.logger.debug("Disconnecting from LSL Streams")
        if self.signal_stream is not None:
            self.signal_stream.disconnect()
        if self.marker_stream is not None:
            self.marker_stream.close_stream()

    def _handle_marker_recording(self):
        if self.marker_stream is not None:
            iteration = 0
            marker_values = []
            marker_times = []

            start_time: float = local_clock()
            # sfreq: float = self.signal_stream.info['sfreq']
            self.marker_time_shift = 0

            recording_stopped_at: Union[float, None] = None

            while self.is_recording:
                if not self.is_recording and recording_stopped_at is None:
                    recording_stopped_at = local_clock()

                markers, timestamps = self.marker_stream.pull_chunk()

                if markers is not None and len(markers) > 0:
                    marker_values.append(markers.copy())
                    marker_times.append(timestamps.copy())

                    time_last_received_sample: float = timestamps[-1]
                    current_time: float = recording_stopped_at if recording_stopped_at is not None else local_clock()
                    self.marker_time_shift: float = current_time - time_last_received_sample

                    self.logger.debug(f"Markers Recorded: {len(timestamps)} Markers")
                    self.logger.debug(f"Time Shift {self.marker_time_shift}")

                    if iteration == 0:
                        self.first_marker_lsl_seconds = timestamps[0].copy()
                        self.first_marker_system_seconds = local_clock()
                        self.first_marker_datetime = datetime.utcnow()
                        self.logger.debug(f"First Marker Time: {self.first_marker_datetime}")
                    iteration += 1
                time.sleep(0.01)

            end_time: float = recording_stopped_at if recording_stopped_at is not None else local_clock()

            self.logger.debug(f"Concatenating markers")
            self.marker_values = np.concatenate(marker_values).flatten() if len(marker_values) > 0 else np.array([])
            self.marker_times = np.concatenate(marker_times).flatten().astype(float) if len(marker_times) > 0 else np.array([])

            duration: float = end_time - start_time
            n_samples: int = len(self.signal_times)

            self.logger.info(f"Marker Recording Stopped. Recorded: {n_samples}, Duration: {format_seconds(duration)}")

            if len(self.marker_times) > 1:
                delta_seconds = self.marker_times[-1] - self.marker_times[0]
                self.last_marker_datetime = self.first_marker_datetime + timedelta(seconds=delta_seconds)
            else:
                self.last_marker_datetime = self.first_marker_datetime
            self.logger.debug("Finished Marker Recording")
            self.marker_recording_completed = True

    def _handle_signal_recording(self):
        if self.signal_stream is not None and self.signal_stream.connected:

            self.logger.debug("Starting Signal Recording")
            iteration: int = 0
            signal_values: List[ndarray] = []
            signal_times: List[ndarray] = []

            start_time: float = local_clock()
            sfreq: float = self.signal_stream.info['sfreq']
            self.signal_time_shift: float = 0

            recording_stopped_at: Union[float, None] = None

            while self.is_recording or self.signal_time_shift > -self.safety_offset_seconds:
                if not self.is_recording and recording_stopped_at is None:
                    recording_stopped_at = local_clock()

                window_size = self.signal_stream.n_new_samples / self.signal_stream.info['sfreq']

                if window_size > 0:
                    (values, times) = self.signal_stream.get_data(winsize=window_size)

                    if values is not None and len(values) > 0:
                        signal_values.append(values.copy())
                        signal_times.append(times.copy())

                        time_last_received_sample: float = times[-1]
                        current_time: float = recording_stopped_at if recording_stopped_at is not None else local_clock()
                        self.signal_time_shift = current_time - time_last_received_sample

                        if iteration == 0:
                            self.first_signal_lsl_seconds = times[0].copy()
                            self.first_signal_system_seconds = local_clock()
                            self.first_signal_datetime = datetime.utcnow()
                            self.logger.debug(f"First Signal Time: {self.first_signal_datetime}")

                        iteration += 1
                time.sleep(0.01)

            end_time: float = recording_stopped_at if recording_stopped_at is not None else local_clock()
            print("pp")

            self.logger.debug(f"Concatenating signals")
            self.signal_values = np.concatenate(signal_values, axis=1) if len(signal_values) > 0 else np.array([])
            self.signal_times = np.concatenate(signal_times).flatten().astype(float) if len(signal_times) > 0 else np.array([])

            duration: float = end_time - start_time
            expected_samples: int = math.ceil(duration * sfreq)
            n_samples: int = len(self.signal_times)
            difference: int = expected_samples - n_samples

            if difference > 0:
                self.logger.warning(f"Recorded less samples than expected: {difference} samples")

            self.logger.info(f"Signal Recording Stopped. Recorded: {n_samples}, Expected: {expected_samples}, "
                             f"Difference: {difference}, Duration: {format_seconds(duration)}")
            # np.abs(np.abs(np.diff(np.abs(self.signal_times - start_time))) - 1 / sfreq)

            if len(self.signal_values) > 1:
                delta_seconds = self.signal_times[-1] - self.signal_times[0]
                self.last_signal_datetime = self.first_signal_datetime + timedelta(seconds=delta_seconds)
            else:
                self.last_signal_datetime = self.first_signal_datetime
            self.logger.debug("Finished Signal Recording")
            self.signal_recording_completed = True

    def reset_recording(self):
        self.signal_values = np.array([])
        self.marker_values = np.array([])
        self.marker_times = np.array([])
        self.first_signal_lsl_seconds = 0
        self.first_signal_system_seconds = 0
        self.first_marker_lsl_seconds = 0
        self.first_marker_system_seconds = 0
        if self.signal_stream is not None:
            self.signal_recording_completed = False
        if self.marker_stream is not None:
            self.marker_recording_completed = False

    def start(self):
        if not self.is_recording:
            self.logger.info("Connecting to LSL Streams")
            self.connect()
            self.logger.info("Starting Recording")
            self.reset_recording()
            self.is_recording = True
            self.recording_start_time = datetime.utcnow()

            if self.signal_stream is not None:
                self._signal_thread = Thread(target=self._handle_signal_recording)
                self._signal_thread.daemon = True
                self._signal_thread.start()

            if self.marker_stream is not None:
                self._marker_thread = Thread(target=self._handle_marker_recording)
                self._marker_thread.daemon = True
                self._marker_thread.start()

    def stop(self):
        if self.is_recording:
            self.is_recording = False
            self.recording_end_time = datetime.utcnow()
            self.logger.info("Stopping Recording")
            while not self.recording_completed:
                time.sleep(1)
            self.summary()
            self.logger.info("Recording Stopped")

    def get_raw(self):
        assert not self.is_recording, "You cannot generate an MNE Raw object while recording"
        assert len(self.signal_values) > 0, "No signal data recorded"
        self.logger.info("Generating MNE Raw Object")
        info = self.signal_stream.info.copy()
        raw = RawArray(self.signal_values, info)
        if self.marker_stream is not None and self.marker_values is not None and len(self.marker_values) > 0:
            marker_times = self.marker_times - self.first_signal_lsl_seconds
            raw.set_annotations(mne.Annotations(onset=marker_times, duration=[0.05] * len(marker_times),
                                                description=self.marker_values.astype(str)))
        raw.set_meas_date(self.first_signal_datetime.replace(tzinfo=timezone.utc).timestamp())
        if len(raw.info['device_info']) == 0:
            raw.info['device_info'] = None
        self.logger.info("MNE Raw Object Generated")
        return raw

    def save_raw(self, file_path: str) -> Tuple[RawArray, Path]:
        raw = self.get_raw()
        self.logger.info(f"Saving raw data to {file_path}")
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(file_path, overwrite=True)
        self.logger.info(f"Saved raw data to {file_path}")
        return raw, file_path

    def complete(self, file_path: str) -> Tuple[RawArray, RecordingInfo]:
        self.stop()
        raw, path = self.save_raw(file_path)
        info = self.get_info()
        info.file_path = path
        self.disconnect()
        self.logger.info("Recording Completed")
        return raw, info

    def get_info(self) -> RecordingInfo:
        signal_info = InletInfo()
        if self.signal_stream is not None:
            signal_info.source_id = self.signal_stream.source_id
            signal_info.sfreq = self.signal_stream.info['sfreq']
            signal_info.n_channels = len(self.signal_stream.ch_names)
            signal_info.iterations = len(self.signal_values)
            signal_info.time_shift = self.signal_time_shift
            signal_info.samples_recorded = len(self.signal_values)
            signal_info.samples_expected = math.ceil((self.recording_end_time - self.recording_start_time).seconds * signal_info.sfreq)

        marker_info = InletInfo()
        if self.marker_stream is not None:
            marker_info.source_id = self.marker_stream.name
            marker_info.sfreq = self.marker_stream.sfreq
            marker_info.n_channels = self.marker_stream.n_channels
            marker_info.iterations = len(self.marker_values)
            marker_info.time_shift = self.marker_time_shift
            marker_info.samples_recorded = len(self.marker_values)
            marker_info.samples_expected = math.ceil((self.recording_end_time - self.recording_start_time).seconds * marker_info.sfreq)

        return RecordingInfo(start_time=self.recording_start_time,
                             end_time=self.recording_end_time,
                             duration=self.recording_end_time - self.recording_start_time,
                             signal_info=signal_info,
                             marker_info=marker_info,
                             file_path=None)

    def summary(self):
        print("--------------------------------------------------------------------------------------")
        print(f"Recording Summary")
        print("--------------------------------------------------------------------------------------", "\n")
        print(f"Recording Started at: {self.recording_start_time}")
        print(f"Recording Ended at: {self.recording_end_time}")
        print(f"Recoding Duration: {self.recording_end_time - self.recording_start_time} \n")

        if self.signal_stream is not None:
            print(f"First Signal Time: {self.first_signal_datetime}")
            print(f"Last Signal Time: {self.last_signal_datetime}")
            signal_duration = self.last_signal_datetime - self.first_signal_datetime
            print(f"Signal Recording Duration: {format_seconds(signal_duration.seconds)}")
            print(f"Number of Signal Windows: {len(self.signal_values)} \n")

        if self.marker_stream is not None:
            print(f"First Marker Time: {self.first_marker_datetime}")
            print(f"Last Marker Time: {self.last_marker_datetime}")
            if self.last_marker_datetime is not None and self.first_marker_datetime is not None:
                marker_duration = self.last_marker_datetime - self.first_marker_datetime
            else:
                marker_duration = timedelta(seconds=0)
            print(f"Marker Recording Duration: {format_seconds(marker_duration.seconds)}")
            print(f"Number of Markers: {len(self.marker_values)}")
        print("--------------------------------------------------------------------------------------")
