import time

from mne_lsl.stream import StreamLSL
from mne_lsl.lsl import local_clock, StreamInlet, resolve_streams
from typing import List
from threading import Thread
import math
import numpy as np
import mne
from mne.io import RawArray
from datetime import datetime, timedelta, timezone
import tkinter as tk
from tkinter import Tk
from pathlib import Path
import logging
from typing import Tuple, Union
from numpy import ndarray

class Recorder:
    signal_id: Union[str, None]
    marker_id: Union[str, None]

    signal_stream: StreamLSL
    marker_stream: any

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

    recording_start_time: datetime
    recording_end_time: datetime
    first_signal_datetime: datetime
    last_signal_datetime: datetime
    first_marker_datetime: datetime
    last_marker_datetime: datetime

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
            self.signal_stream.connect()
            self.logger.debug(f"Signal Stream Connected with id: {self.signal_id}")

        if self.marker_id is not None:
            streams = resolve_streams(source_id=self.marker_id)
            if len(streams) == 1:
                self.marker_stream = StreamInlet(streams[0])
                self.marker_stream.open_stream()
                self.logger.debug(f"Marker Stream Connected with id: {self.marker_id}")

    def disconnect(self):
        self.logger.debug("Disconnecting from LSL Streams")
        self.signal_stream.disconnect()
        if self.marker_stream is not None:
            self.marker_stream.close_stream()

    def _handle_marker_recording(self):
        if self.marker_stream is not None:
            iteration = 0
            marker_values = []
            marker_times = []
            while self.is_recording:
                markers, timestamps = self.marker_stream.pull_chunk()
                if markers is not None and len(markers) > 0:
                    marker_values.append(markers.copy())
                    marker_times.append(timestamps.copy())
                    if iteration == 0:
                        self.first_marker_lsl_seconds = timestamps[0].copy()
                        self.first_marker_system_seconds = local_clock()
                        self.first_marker_datetime = datetime.utcnow()
                        self.logger.debug(f"First Marker Time: {self.first_marker_datetime}")
                    iteration += 1
            self.logger.debug(f"Concatenating markers")
            self.marker_values = np.concatenate(marker_values).flatten()
            self.marker_times = np.concatenate(marker_times).flatten().astype(float)
            self.logger.debug(f"Marker Recording Stopped. Recorded {len(self.marker_values)} markers")
            if len(self.marker_times) > 1:
                delta_seconds = self.marker_times[-1] - self.marker_times[0]
                self.last_marker_datetime = self.first_marker_datetime + timedelta(seconds=delta_seconds)
            else:
                self.last_marker_datetime = self.first_marker_datetime
            self.logger.debug("Finished Marker Recording")
            self.marker_recording_completed = True

    def _handle_signal_recording(self):
        self.logger.debug("Starting Signal Recording")
        iteration = 0
        signal_values = []
        signal_times = []
        while self.is_recording:
            window_size = self.signal_stream.n_new_samples / self.signal_stream.info['sfreq']
            if window_size > 0:
                values, times = self.signal_stream.get_data(winsize=window_size)
                if values is not None and len(values) > 0:
                    signal_values.append(values.copy())
                    signal_times.append(times.copy())
                    if iteration == 0:
                        self.first_signal_lsl_seconds = times[0].copy()
                        self.first_signal_system_seconds = local_clock()
                        self.first_signal_datetime = datetime.utcnow()
                        self.logger.debug(f"First Signal Time: {self.first_signal_datetime}")
                    iteration += 1
        self.logger.debug(f"Concatenating signals")
        self.signal_values = np.concatenate(signal_values, axis=1)
        self.signal_times = np.concatenate(signal_times).flatten().astype(float)
        self.logger.debug(f"Signal Recording Stopped. Recorded {len(self.signal_values)} windows")
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
        self.signal_recording_completed = False

    def start(self):
        if not self.is_recording:
            self.logger.info("Connecting to LSL Streams")
            self.connect()
            self.logger.info("Starting Recording")
            self.reset_recording()
            self.is_recording = True
            self.recording_start_time = datetime.utcnow()
            self._signal_thread = Thread(target=self._handle_signal_recording)
            self._signal_thread.daemon = True
            self._signal_thread.start()
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

    def complete(self, file_path: str) -> Tuple[RawArray, Path]:
        self.stop()
        raw, path = self.save_raw(file_path)
        self.disconnect()
        self.logger.info("Recording Completed")
        return raw, path

    def summary(self):
        print("--------------------------------------------------------------------------------------")
        print(f"Recording Summary")
        print("--------------------------------------------------------------------------------------", "\n")
        print(f"Recording Started at: {self.recording_start_time}")
        print(f"Recording Ended at: {self.recording_end_time}")
        print(f"Recoding Duration: {self.recording_end_time - self.recording_start_time} \n")

        print(f"First Signal Time: {self.first_signal_datetime}")
        print(f"Last Signal Time: {self.last_signal_datetime}")
        print(f"Signal Recording Duration: {self.last_signal_datetime - self.first_signal_datetime}")
        print(f"Number of Signal Windows: {len(self.signal_values)} \n")

        print(f"First Marker Time: {self.first_marker_datetime}")
        print(f"Last Marker Time: {self.last_marker_datetime}")
        print(f"Marker Recording Duration: {self.last_marker_datetime - self.first_marker_datetime}")
        print(f"Number of Markers: {len(self.marker_values)}")
        print("--------------------------------------------------------------------------------------")


def start(recorder: Recorder, root: Tk):
    message = tk.Label(root, text="Start Recording")
    message.pack()
    recorder.start()
    message = tk.Label(root, text="Recording...")
    message.pack()




def stop(recorder: Recorder, root: Tk):
    message = tk.Label(root, text="Stopping Recording...")
    message.pack()
    recorder.complete("./data/recordings/test_raw.fif")
    message = tk.Label(root, text="Recording Completed and Saved")
    message.pack()


def main():
    recorder = Recorder(signal_id='rsvp_eeg', marker_id='rsvp_markers', buffer_size_seconds=60)
    root = tk.Tk()
    root.geometry("200x200")
    start_button = tk.Button(root, text="Start", command=lambda: start(recorder, root))
    start_button.pack()
    stop_button = tk.Button(root, text="Stop", command=lambda: stop(recorder, root))
    stop_button.pack()
    root.mainloop()


if __name__ == "__main__":
    main()
