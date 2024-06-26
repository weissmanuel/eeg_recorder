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


def generate_channel_data(freqs: ndarray, weights: ndarray, t: ndarray) -> ndarray:
    if len(freqs) != 4:
        freqs = np.array([8, 10, 35, 50])
    signal = np.zeros_like(t)
    for i, freq in enumerate(freqs):
        if freq < 0:
            raise ValueError("Frequency must be greater than 0")
        signal += weights[i] * np.sin(2 * np.pi * freq * t)
    return signal


def generate_demo_data(
        real_time_store: RealTimeStore,
        num_channels: int = 2,
        target_frequencies: ndarray | list[int] | None = np.array([8, 10, 35, 50])
) -> Tuple[ndarray, float]:
    weights = np.array([1, 0.2, 0.2, 1])

    real_time_store.iterations = real_time_store.iterations \
        if real_time_store.iterations < len(real_time_store.demo_time_space) else 0

    t = real_time_store.demo_time_space[real_time_store.iterations]

    real_time_store.iterations += 1
    ch = generate_channel_data(target_frequencies, weights, t)
    data = np.expand_dims(np.repeat(np.array([ch]), num_channels), axis=1)
    return data, local_clock()
