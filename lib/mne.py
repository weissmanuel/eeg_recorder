from numpy import ndarray
import mne
from mne import Info
from mne.io import RawArray
from omegaconf import DictConfig
from lib.utils import config_to_primitive


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
