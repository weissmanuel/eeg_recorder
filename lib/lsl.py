import numpy as np
from mne_lsl.stream import StreamLSL
from mne_lsl.lsl import StreamOutlet, StreamInfo
import logging
from lib.worker import StreamType
from typing import List, Dict
from numpy import ndarray
from omegaconf import ListConfig

logger = logging.getLogger(__name__)


class Outlet:
    """Base Outlet Stream for EEG signal, marker or actuator streams.

        Parameters
        ----------
        stype : str
            Content type of the stream, e.g. ``"EEG"`` or ``"Gaze"``. If a stream contains
            mixed content, this value should be empty and the description of each channel
            should include its type.
        n_channels : int ``≥ 1``
            Also called ``channel_count``, represents the number of channels per sample.
            This number stays constant for the lifetime of the stream.
        sfreq : float ``≥ 0``
            Also called ``nominal_srate``, represents the sampling rate (in Hz) as
            advertised by the data source. If the sampling rate is irregular (e.g. for a
            trigger stream), the sampling rate is set to ``0``.
        source_id : str
            A unique identifier of the device or source of the data. If not empty, this
            information improves the system robustness since it allows recipients to recover
            from failure by finding a stream with the same ``source_id`` on the network.
        autoconnect : bool
            If ``True``, the stream is connected to the LSL network upon instantiation
        """
    source_id: str
    info: StreamInfo
    _stream_outlet: StreamOutlet | None = None

    def __init__(self,
                 source_id: str,
                 sfreq: float = 0.0,
                 stype: str = 'Markers',
                 n_channels: int = 1,
                 autoconnect: bool = True):
        self.source_id = source_id
        dtype = 'int32' if stype.lower() == 'markers' else 'float32'
        self.info = StreamInfo(name=source_id, stype=stype, n_channels=n_channels,
                               dtype=dtype, source_id=source_id, sfreq=sfreq)
        if autoconnect:
            self.connect()

    def connect(self):
        self._stream_outlet = StreamOutlet(self.info)

    def disconnect(self):
        if self._stream_outlet is not None:
            self._stream_outlet.__del__()
        self._stream_outlet = None

    def push(self, x: int | float | str | List[str] | List[float] | List[int] | ndarray,
             timestamp: float | None = None,
             pushThrough: bool = True):
        if isinstance(x, (int, float, str)):
            x = [x]
        if isinstance(x, list):
            x = np.array(x)

        if timestamp is None:
            # setting timestamp to 0.0 for automatic timestamping by current time
            timestamp = 0.0

        if self._stream_outlet is not None:

            if isinstance(x, ndarray) and 1 < len(x.shape) < 3:
                self._stream_outlet.push_chunk(x, timestamp, pushThrough)
            elif isinstance(x, ndarray) and len(x.shape) == 1:
                self._stream_outlet.push_sample(x, timestamp, pushThrough)
            else:
                logger.warning(f"Failed to push data to LSL Stream {self.source_id}: Unsupported data shape")
        else:
            logger.warning(f"Failed to push data to LSL Stream {self.source_id}: Stream is not connected")


def connect(source_id: str, stream_type: StreamType, buffer_size_seconds: float) -> StreamLSL | None:
    logger.debug(f"Start Connecting to LSL Stream: {source_id} of type {stream_type.value}")

    try:
        stream = StreamLSL(bufsize=buffer_size_seconds, source_id=source_id)
        stream.connect()
        logger.info(f"Connected to LSL Stream: {source_id} of type {stream_type.value}")
        return stream
    except Exception as e:
        logger.warning(f"Failed to connect to LSK Streams {source_id}: {e}")
        return None


def disconnect(stream: StreamLSL) -> None:
    if stream is not None:
        stream.disconnect()
    return None


def get_outlets(config: ListConfig | List[Dict]) -> List[Outlet]:
    outlets = []
    for outlet in config:
        outlets.append(Outlet(**outlet))
    return outlets
