from datetime import timedelta
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Tuple
import numpy as np
from numpy import ndarray
from mne_lsl.lsl import local_clock
from lib.store import RealTimeStore


def format_seconds(seconds: float) -> str:
    duration = timedelta(seconds=seconds)
    formatted_duration = "{:02}H:{:02}m:{:02}s".format(
        duration.seconds // 3600,
        (duration.seconds // 60) % 60,
        duration.seconds % 60
    )

    if duration.days > 0:
        formatted_duration = "{}d:{}".format(duration.days, formatted_duration)
    return formatted_duration


def config_to_primitive(cfg: any) -> any:
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, dict):
        return {k: config_to_primitive(v) for k, v in cfg.items()}
    return cfg


def generate_demo_data(real_time_store: RealTimeStore, num_channels: int = 2) -> Tuple[ndarray, float]:
    real_time_store.iterations = real_time_store.iterations \
        if real_time_store.iterations < len(real_time_store.demo_time_space) else 0
    t = real_time_store.demo_time_space[real_time_store.iterations]
    ch = lambda x: (np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 8 * x) +
                    0.5 * np.sin(2 * np.pi * 35 * x) + np.sin(2 * np.pi * 50 * x))
    real_time_store.iterations += 1
    data = np.expand_dims(np.repeat(np.array([ch(t)]), num_channels), axis=1)
    return data, local_clock()