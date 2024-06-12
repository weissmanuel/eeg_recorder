from numpy import ndarray
import mne
from mne import Info
from mne.io import RawArray, BaseRaw
from omegaconf import DictConfig
from lib.utils import config_to_primitive
from abc import ABC, abstractmethod


def create_info(config: DictConfig) -> Info:
    info = mne.create_info(ch_names=config_to_primitive(config.headset.channel_names),
                           sfreq=config.headset.sfreq,
                           ch_types=config_to_primitive(config.headset.channel_types), verbose=False)
    if 'montage' in config.headset:
        montage = mne.channels.make_standard_montage(config.headset.montage)
        info.set_montage(montage, verbose=False)
    return info


def create_raw_from_config(data: ndarray, config: DictConfig) -> RawArray:
    info = create_info(config)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def create_raw(data: ndarray, info: Info) -> RawArray:
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def load_raw(file_path: str) -> BaseRaw:
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    raw = raw.pick_types(eeg=True)
    return raw


def generate_epochs(raw: BaseRaw, event_mapping: dict | None = None,
                    t_min: float = 0.0, t_max: float = 1.0) -> mne.Epochs:
    events, _ = mne.events_from_annotations(raw, event_id=event_mapping)
    return mne.Epochs(raw, events, tmin=t_min, tmax=t_max, baseline=None, preload=True)
