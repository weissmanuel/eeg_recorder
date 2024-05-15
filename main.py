from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
from mne_lsl.lsl import local_clock
from typing import List
import time
import asyncio
from threading import Thread


class Recorder:
    signal_id: str
    marker_id: str
    _buffer_size: int

    signal_stream: StreamLSL
    marker_stream: StreamLSL | None

    _signal_array: List[float] = []

    is_recording: bool = False
    _thread: Thread

    def __init__(self, signal_id: str, marker_id: str | None, buffer_size: int):
        self.signal_id = signal_id
        self.marker_id = marker_id

        self._buffer_size = buffer_size

        self.signal_stream = StreamLSL(bufsize=self._buffer_size, source_id=self.signal_id)
        self.marker_stream = None
        self.loop = asyncio.get_event_loop()

    @property
    def buffer_size(self) -> int:
        return self._buffer_size * 100

    def connect(self):
        self.signal_stream.connect()

    def _handle_recording(self):
        while self.is_recording:
            print("hejooo")
            window_size = self.signal_stream.n_new_samples / self.signal_stream.info['sfreq']
            data = self.signal_stream.get_data(winsize=window_size)
            self._signal_array.extend(data)

    def start(self):
        if not self.is_recording:
            self.is_recording = True
            self._thread = Thread(target=self._handle_recording)
            self._thread.daemon = True
            self._thread.start()

    def stop(self):
        self.is_recording = False

    def get_raw(self):
        assert not self.is_recording, "You cannot generate an MNE Raw object while recording"
        print("Here")


async def main():
    recorder = Recorder(signal_id='rsvp_eeg', marker_id=None, buffer_size=2)
    duration = 5
    recorder.connect()
    recorder.start()
    print("here")
    await asyncio.sleep(duration)
    recorder.stop()
    recorder.get_raw()


if __name__ == "__main__":
    asyncio.run(main())
