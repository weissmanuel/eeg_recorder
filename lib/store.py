from multiprocessing import Manager
from typing import List, Tuple, Union
from datetime import datetime
from enum import Enum
import numpy as np
from numpy import ndarray
from mne import Info


class StreamType(Enum):
    EEG = 'EEG'
    MARKER = 'MARKER'

    @property
    def is_signal(self):
        return self in [StreamType.EEG]

    @property
    def is_marker(self):
        return self in [StreamType.MARKER]

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(value: str):
        value = value.upper()
        assert value in [e.value for e in StreamType], f"Unsupported StreamType: {value}"
        return StreamType(value)


class StreamStore:
    source_id: str
    stream_type: StreamType

    safety_offset_seconds: float

    def __init__(self, manager: Manager, source_id: str, stream_type: StreamType):
        self.source_id = source_id
        self.stream_type = stream_type

        self._data = manager.list()
        self._times = manager.list()

        self._iterations = manager.Value('i', 0)
        self._first_sample_lsl_seconds = manager.Value('d', 0.0)
        self._first_sample_system_seconds = manager.Value('d', 0.0)

        self.safety_offset_seconds = 1.0
        self._recording_completed = manager.Value('b', False)

        self._first_sample_datetime = manager.Value('O', None)
        self._last_sample_datetime = manager.Value('O', None)

        self._time_shift = manager.Value('d', 0.0)
        self._last_batch_received_time = manager.Value('d', 0.0)
        self._current_time = manager.Value('d', 0.0)

        self._start_time_seconds = manager.Value('d', 0.0)
        self._end_time_seconds = manager.Value('d', 0.0)
        self._sfreq = manager.Value('d', 0.0)

        self._stream_info = manager.Value('O', None)
        self._n_channels = manager.Value('i', 0)
        self._has_stream = manager.Value('b', False)

    @property
    def has_stream(self) -> bool:
        return self._has_stream.value

    @has_stream.setter
    def has_stream(self, value: bool) -> None:
        self._has_stream.value = value

    @property
    def raw_data_array(self) -> List[any]:
        return list(self._data)

    @property
    def data(self) -> ndarray:
        data = list(self._data)
        data = np.concatenate(data, axis=1) if len(data) > 0 else np.array([])
        if self.stream_type.is_marker:
            data = data.flatten()
        return data

    def has_data(self) -> bool:
        return len(self._data) > 0

    def set_data(self, data: List[any]) -> None:
        self._data[:] = data

    def extend_data(self, data: List[any]) -> None:
        self._data.extend(data)

    def append_data(self, value: any) -> None:
        self._data.append(value)

    def clear_data(self) -> None:
        self._data[:] = []

    def get_and_clear_data(self) -> ndarray:
        data = self.data
        self.clear_data()
        return data

    @property
    def raw_times_array(self) -> List[float]:
        return list(self._times)

    @property
    def times(self) -> ndarray:
        times = list(self._times)
        times = np.concatenate(times).flatten().astype(float) if len(times) > 0 else np.array([])
        return times

    def set_times(self, times: List[any]) -> None:
        self._times[:] = times

    def extend_times(self, times: List[any]) -> None:
        self._times.extend(times)

    def append_times(self, times: any) -> None:
        self._times.append(times)

    def clear_times(self) -> None:
        self._times[:] = []

    def get_and_clear_times(self) -> ndarray:
        times = self.times
        self.clear_times()
        return times

    def get_and_clear_data_times(self) -> Tuple[ndarray, ndarray]:
        data, times = self.get_and_clear_data(), self.get_and_clear_times()
        self.iterations = 0
        self.first_sample_lsl_seconds = 0.0
        self.first_sample_system_seconds = 0.0
        self.first_sample_datetime = None
        self.last_sample_datetime = None
        assert len(data) == len(
            times), f"Data and Times must have the same length. Data: {len(data)}, Times: {len(times)}"
        return data, times

    @property
    def iterations(self) -> int:
        return self._iterations.value

    @iterations.setter
    def iterations(self, value: int) -> None:
        self._iterations.value = value

    def increment_iterations(self) -> None:
        self._iterations.value += 1

    @property
    def first_sample_lsl_seconds(self) -> float:
        return self._first_sample_lsl_seconds.value

    @first_sample_lsl_seconds.setter
    def first_sample_lsl_seconds(self, value: float) -> None:
        self._first_sample_lsl_seconds.value = value

    @property
    def last_sample_lsl_seconds(self) -> float:
        if len(self.times) > 0:
            return float(self.times[-1])
        else:
            return 0.0

    @property
    def first_sample_system_seconds(self) -> float:
        return self._first_sample_system_seconds.value

    @first_sample_system_seconds.setter
    def first_sample_system_seconds(self, value: float) -> None:
        self._first_sample_system_seconds.value = value

    @property
    def recording_completed(self) -> bool:
        return self._recording_completed.value

    @recording_completed.setter
    def recording_completed(self, value: bool) -> None:
        self._recording_completed.value = value

    @property
    def first_sample_datetime(self) -> datetime:
        return self._first_sample_datetime.value

    @first_sample_datetime.setter
    def first_sample_datetime(self, value: datetime) -> None:
        self._first_sample_datetime.value = value

    @property
    def last_sample_datetime(self) -> datetime:
        return self._last_sample_datetime.value

    @last_sample_datetime.setter
    def last_sample_datetime(self, value: datetime) -> None:
        self._last_sample_datetime.value = value

    @property
    def time_shift(self) -> float:
        return self._time_shift.value

    @time_shift.setter
    def time_shift(self, value: float) -> None:
        self._time_shift.value = value

    @property
    def last_batch_received_time(self) -> float:
        return self._last_batch_received_time.value

    @last_batch_received_time.setter
    def last_batch_received_time(self, value: float) -> None:
        self._last_batch_received_time.value = value

    @property
    def current_time(self) -> float:
        return self._current_time.value

    @current_time.setter
    def current_time(self, value: float) -> None:
        self._current_time.value = value

    @property
    def start_time_seconds(self) -> float:
        return self._start_time_seconds.value

    @start_time_seconds.setter
    def start_time_seconds(self, value: float) -> None:
        self._start_time_seconds.value = value

    @property
    def end_time_seconds(self) -> float:
        return self._end_time_seconds.value

    @end_time_seconds.setter
    def end_time_seconds(self, value: float) -> None:
        self._end_time_seconds.value = value

    @property
    def sfreq(self) -> float:
        if self._sfreq.value is None:
            return 0.0
        return self._sfreq.value

    @sfreq.setter
    def sfreq(self, value: float) -> None:
        self._sfreq.value = value

    @property
    def stream_info(self) -> Union[Info, None]:
        return self._stream_info.value or None

    @stream_info.setter
    def stream_info(self, value: Info) -> None:
        self._stream_info.value = value

    @property
    def n_channels(self) -> int:
        return self._n_channels.value

    @n_channels.setter
    def n_channels(self, value: int) -> None:
        self._n_channels.value = value

    @property
    def duration(self) -> float:
        if self.start_time_seconds is not None and self.end_time_seconds is not None:
            return self.end_time_seconds - self.start_time_seconds
        return 0.0

    @property
    def expected_samples(self) -> int:
        return int(self.sfreq * self.duration)

    @property
    def n_samples(self) -> int:
        return self.data.shape[-1]

    @property
    def sample_deviation(self) -> int:
        return self.expected_samples - self.n_samples

    def reset(self):
        self.clear_data()
        self.clear_times()
        self.iterations = 0
        self.first_sample_lsl_seconds = 0.0
        self.first_sample_system_seconds = 0.0
        self.recording_completed = False
        self.first_sample_datetime = None
        self.last_sample_datetime = None
        self.time_shift = 0.0
        self.last_batch_received_time = 0.0
        self.current_time = 0.0
        self.start_time_seconds = 0.0
        self.end_time_seconds = 0.0
        self.sfreq = 0.0

    def copy_state(self):
        return {
            'data': self.data,
            'times': self.times,
            'iterations': self.iterations,
            'first_sample_lsl_seconds': self.first_sample_lsl_seconds,
            'first_sample_system_seconds': self.first_sample_system_seconds,
            'recording_completed': self.recording_completed,
            'first_sample_datetime': self.first_sample_datetime,
            'last_sample_datetime': self.last_sample_datetime,
            'time_shift': self.time_shift,
            'last_batch_received_time': self.last_batch_received_time,
            'current_time': self.current_time,
            'start_time_seconds': self.start_time_seconds,
            'end_time_seconds': self.end_time_seconds,
            'sfreq': self.sfreq,
            'stream_info': self.stream_info,
            'n_channels': self.n_channels,
            'has_stream': self.has_stream
        }


class RecorderStore:

    def __init__(self, manager: Manager):
        self._is_recording = manager.Value('b', False)
        self._recording_start_time = manager.Value('O', None)
        self._recording_end_time = manager.Value('O', None)

    @property
    def is_recording(self) -> bool:
        return self._is_recording.value

    @is_recording.setter
    def is_recording(self, value: bool) -> None:
        self._is_recording.value = value

    @property
    def recording_start_time(self) -> datetime:
        return self._recording_start_time.value

    @recording_start_time.setter
    def recording_start_time(self, value: datetime) -> None:
        self._recording_start_time.value = value if value is not None else datetime.now()

    @property
    def recording_end_time(self) -> datetime:
        return self._recording_end_time.value

    @recording_end_time.setter
    def recording_end_time(self, value: datetime | None) -> None:
        self._recording_end_time.value = value if value is not None else datetime.now()

    @property
    def recording_duration(self) -> float:
        if self.recording_start_time is not None and self.recording_end_time is not None:
            return (self.recording_end_time - self.recording_start_time).total_seconds()
        return 0.0

    def start(self):
        self.is_recording = True
        self.recording_start_time = datetime.utcnow()

    def stop(self):
        self.is_recording = False
        self.recording_end_time = datetime.utcnow()

    def reset(self):
        self.is_recording = False
        self.recording_start_time = None
        self.recording_end_time = None
