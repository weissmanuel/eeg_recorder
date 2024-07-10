import mne
from mne import Info
from mne.io import RawArray
from lib.preprocess.data_preprocess import get_preprocessors, Preprocessor
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
from lib.utils import get_default_file_path
from lib.preprocess.models import ProcessStage


class PersistingMode(Enum):
    REPLACE = "REPLACE"
    CONTINUOUS = "CONTINUOUS"

    @staticmethod
    def from_string(mode: str) -> 'PersistingMode':
        return PersistingMode(mode.upper())


class Persister(ABC):
    logger = logging.getLogger(__name__)
    config: DictConfig
    persisting_mode: PersistingMode

    def __init__(self, persisting_mode: PersistingMode | str = PersistingMode.REPLACE):
        if isinstance(persisting_mode, str):
            self.persisting_mode = PersistingMode.from_string(persisting_mode)
        else:
            self.persisting_mode = persisting_mode

    @abstractmethod
    def save(self, stores: List[StreamStore], **kwargs) -> Union[any, Union[str, Path]]:
        pass

    @abstractmethod
    def delete(self):
        pass


class MneRawPersister(Persister):
    config: DictConfig
    persisting_mode: PersistingMode

    def __init__(self, config, persisting_mode: PersistingMode | str = PersistingMode.REPLACE):
        super().__init__(persisting_mode)
        self.config = config

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
        preprocessors: List[Preprocessor] = get_preprocessors(self.config.recording.preprocessors,
                                                              stages=[ProcessStage.RECORDING])
        if preprocessors is not None and len(preprocessors) > 0:
            for preprocessor in preprocessors:
                data = preprocessor(data, info=info)
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
        file_path = Path(file_path if file_path is not None else get_default_file_path(config=self.config))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        raw, file_path = self.save_by_mode(raw, file_path)
        self.logger.info(f"Saved raw data to {file_path}")
        return raw, file_path

    def save(self,
             stores: List[StreamStore],
             file_path: Union[str, None] = None,
             intermediate_save: bool = False,
             **kwargs) -> Tuple[RawArray, Path]:
        signal_store = [store for store in stores if store.stream_type.is_signal]
        assert len(signal_store) == 1, "Only one signal store is currently supported."

        marker_stores = [store for store in stores if store.stream_type.is_marker]
        raw, file_path = self.save_raw(signal_store[0], marker_stores, file_path, intermediate_save)

        return raw, file_path

    def delete(self):
        file_path: Path = Path(get_default_file_path(config=self.config))
        try:
            file_path.unlink(missing_ok=True)
            self.logger.info(f"Deleted Existing File: {file_path}")
        except FileNotFoundError:
            self.logger.info(f"Failed to delete file: {file_path}")


def get_persister(name: str | None, config: DictConfig, **kwargs) -> Persister | None:
    if name is None:
        return None
    if name == 'mne_raw_persister':
        return MneRawPersister(config, **kwargs)
    else:
        raise ValueError(f"Persister {name} not found")


def get_persisters(config: DictConfig) -> List[Persister]:
    if hasattr(config.experiment, 'persisters') and config.experiment.persisters is not None:
        return [get_persister(persister, config) for persister in config.experiment.persisters]

