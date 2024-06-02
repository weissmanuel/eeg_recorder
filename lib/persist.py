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
from enum import Enum

class PersistingMode(Enum):
    REPLACE = "REPLACE"
    CONTINUOUS = "CONTINUOUS"

    @staticmethod
    def from_string(mode: str) -> 'PersistingMode':
        return PersistingMode(mode.upper())


class Persister(ABC):
    logger = logging.getLogger(__name__)

    @abstractmethod
    def save(self, stores: List[StreamStore]) -> Union[any, Union[str, Path]]:
        pass


class MneRawPersister(Persister):
    config: DictConfig
    persisting_mode: PersistingMode

    def __init__(self, config):
        self.config = config

        if 'persisting_mode' in self.config:
            self.persisting_mode = PersistingMode.from_string(self.config.persisting_mode)
        else:
            self.persisting_mode = PersistingMode.REPLACE

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

    def add_annotations(self,
                        raw: RawArray,
                        signal_store: StreamStore,
                        marker_store: StreamStore,
                        intermediate_save: bool = False,
                        ) -> RawArray:
        if marker_store is not None and marker_store.stream_type == StreamType.MARKER and marker_store.n_samples > 0:
            first_signal_lsl_seconds = signal_store.first_sample_lsl_seconds
            if intermediate_save:
                marker_values, marker_times = marker_store.get_and_clear_data_times()
            else:
                marker_values = marker_store.data
                marker_times = marker_store.times - first_signal_lsl_seconds
            raw.set_annotations(mne.Annotations(onset=marker_times, duration=[0.05] * len(marker_times),
                                                description=marker_values.astype(str)))
        return raw

    def get_raw(self,
                signal_store: StreamStore,
                marker_stores: List[StreamStore],
                intermediate_save: bool = False) -> RawArray:

        if intermediate_save:
            orig_state = signal_store.copy()
            signal, _ = signal_store.get_and_clear_data_times()
            new_state = signal_store
            signal_store = orig_state
        else:
            signal = signal_store.data

        assert signal.shape[-1] > 0, "No signal data recorded"
        self.logger.info("Generating MNE Raw Object")
        info = self.create_mne_info(signal_store)
        signal = self.preprocess(info, signal)
        raw = RawArray(signal, info)

        for marker_store in marker_stores:
            raw = self.add_annotations(raw, signal_store, marker_store, intermediate_save)

        raw.set_meas_date(signal_store.first_sample_datetime.replace(tzinfo=timezone.utc).timestamp())

        if len(raw.info['device_info']) == 0:
            raw.info['device_info'] = None
        self.logger.info("MNE Raw Object Generated")
        return raw

    def save_replace(self, raw: RawArray, file_path: Path):
        raw.save(file_path, overwrite=True)
        return raw, file_path

    def save_append(self, raw: RawArray, file_path: Path):
        exists = file_path.exists()
        if exists:
            existing_raw = mne.io.read_raw_fif(file_path)
            data = existing_raw.get_data()
            info = existing_raw.info
            existing_raw_array = RawArray(data, info)
            existing_raw_array.append(raw)
            existing_raw_array.save(file_path, overwrite=True)
        else:
            self.save_replace(raw, file_path)
        return raw, file_path

    def save_by_mode(self, raw: RawArray, file_path: Path):
        if self.persisting_mode == PersistingMode.REPLACE:
            return self.save_replace(raw, file_path)
        elif self.persisting_mode == PersistingMode.CONTINUOUS:
            return self.save_append(raw, file_path)

    def save_raw(self,
                 signal_store: StreamStore,
                 marker_stores: List[StreamStore],
                 file_path: Union[str, None] = None,
                 intermediate_save: bool = False) -> Tuple[RawArray, Path]:
        raw = self.get_raw(signal_store, marker_stores, intermediate_save)
        self.logger.info(f"Saving raw data to {file_path}")
        file_path = Path(file_path if file_path is not None else self.get_file_path())
        file_path.parent.mkdir(parents=True, exist_ok=True)
        raw, file_path = self.save_by_mode(raw, file_path)
        self.logger.info(f"Saved raw data to {file_path}")
        return raw, file_path

    def save(self,
             stores: List[StreamStore],
             file_path: Union[str, None] = None,
             intermediate_save: bool = False) -> Tuple[RawArray, Path]:
        signal_store = [store for store in stores if store.stream_type.is_signal]
        assert len(signal_store) == 1, "Only one signal store is currently supported."

        marker_stores = [store for store in stores if store.stream_type.is_marker]
        raw, file_path = self.save_raw(signal_store[0], marker_stores, file_path, intermediate_save)

        return raw, file_path
