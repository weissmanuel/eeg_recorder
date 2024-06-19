from collections import deque
from typing import List
import itertools
from numpy import ndarray
import numpy as np


class RingBuffer:
    buffer_size = 1000
    _buffer: deque

    n_new_samples: int = 0

    def __init__(self,
                 sfreq: float,
                 window_size_seconds: float = 3,
                 window_shift_seconds: float = 1,
                 buffer_size_seconds: int = 10):
        self.sfreq = sfreq
        self.window_size_seconds = window_size_seconds
        self.window_shift_seconds = window_shift_seconds
        self.window_size = int(window_size_seconds * sfreq)
        self.window_shift = int(window_shift_seconds * sfreq)
        self.buffer_size_seconds = buffer_size_seconds
        self.buffer_size = int(buffer_size_seconds * sfreq)
        print("Buffer size: ", self.buffer_size)
        self._buffer = deque([0] * self.buffer_size, maxlen=self.buffer_size)
        # self.head = self.window_size

    @property
    def tail(self) -> int:
        return max(int(self.head - self.window_size), 0)

    @property
    def head(self) -> int:
        return max(min(self.n_new_samples, self.buffer_size), self.window_size)

    def check_first_window_filled(self):
        return self.n_new_samples >= self.window_size

    def update_n_new_samples(self, n: int):
        self.n_new_samples = max(min(self.n_new_samples + n, self.buffer_size), 0)
        return self.n_new_samples

    def push(self, value: float):
        self._buffer.extend([value])

    def add_data(self, data: List[float | List[float]]):
        num_samples = len(data)
        self._buffer.extend(data)
        self.n_new_samples = self.update_n_new_samples(num_samples)

    def has_new_data(self) -> bool:
        return self.n_new_samples >= self.window_size

    def get_data(self) -> ndarray | None:
        if not self.has_new_data():
            return None
        data = list(itertools.islice(self._buffer, self.buffer_size - self.head, self.buffer_size - self.tail))
        self.update_n_new_samples(-self.window_shift)
        assert len(data) == self.window_size
        return np.array(data)
