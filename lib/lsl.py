from mne_lsl.stream import StreamLSL
import logging
from lib.worker import StreamType

logger = logging.getLogger(__name__)


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
