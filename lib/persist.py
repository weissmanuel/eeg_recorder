import mne
from mne import Info
from mne.io import RawArray
from lib.preprocess import get_preprocessors, Preprocessor
from numpy import ndarray
from lib.utils import config_to_primitive
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from lib.store import StreamStore, StreamType
from typing import Union, List, Tuple
from pathlib import Path
import logging
from datetime import timezone


class Persister(ABC):
    logger = logging.getLogger(__name__)

    @abstractmethod
    def save(self, stores: List[StreamStore]) -> Union[any, Union[str, Path]]:
        pass


class MneRawPersister(Persister):
    config: DictConfig

    def __init__(self, config):
        self.config = config

    def get_file_path(self):
        if 'recording' in self.config:
            subject = self.config.recording.subject
            session = self.config.recording.session
            block = self.config.recording.block
            return f"./data/recordings/recoding_subject_{subject}_session_{session}_block_{block}.fif"

    def create_mne_info(self, stream_store: StreamStore) -> Info:
        info = stream_store.stream_info.copy()
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

    def add_annotations(self, raw: RawArray, signal_store: StreamStore, marker_store: StreamStore) -> RawArray:
        if marker_store is not None and marker_store.stream_type == StreamType.MARKER and marker_store.n_samples > 0:
            first_signal_lsl_seconds = signal_store.first_sample_lsl_seconds
            marker_values = marker_store.data
            marker_times = marker_store.times - first_signal_lsl_seconds
            raw.set_annotations(mne.Annotations(onset=marker_times, duration=[0.05] * len(marker_times),
                                                description=marker_values.astype(str)))
        return raw

    def get_raw(self, signal_store: StreamStore, marker_stores: List[StreamStore]) -> RawArray:
        signal = signal_store.data
        assert signal_store.n_samples > 0, "No signal data recorded"
        self.logger.info("Generating MNE Raw Object")
        info = self.create_mne_info(signal_store)
        signal = self.preprocess(info, signal)
        raw = RawArray(signal, info)

        for marker_store in marker_stores:
            raw = self.add_annotations(raw, signal_store, marker_store)

        raw.set_meas_date(signal_store.first_sample_datetime.replace(tzinfo=timezone.utc).timestamp())

        if len(raw.info['device_info']) == 0:
            raw.info['device_info'] = None
        self.logger.info("MNE Raw Object Generated")
        return raw

    def save_raw(self,
                 signal_store: StreamStore,
                 marker_stores: List[StreamStore],
                 file_path: Union[str, None] = None) -> Tuple[RawArray, Path]:
        raw = self.get_raw(signal_store, marker_stores)
        self.logger.info(f"Saving raw data to {file_path}")
        file_path = Path(file_path if file_path is not None else self.get_file_path())
        file_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(file_path, overwrite=True)
        self.logger.info(f"Saved raw data to {file_path}")
        return raw, file_path

    def save(self, stores: List[StreamStore], file_path: Union[str, None] = None) -> Tuple[RawArray, Path]:
        signal_store = [store for store in stores if store.stream_type.is_signal]
        assert len(signal_store) == 1, "Only one signal store is currently supported."

        marker_stores = [store for store in stores if store.stream_type.is_marker]
        raw, file_path = self.save_raw(signal_store[0], marker_stores, file_path)

        return raw, file_path
