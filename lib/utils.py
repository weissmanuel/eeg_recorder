from datetime import timedelta
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Union


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
    return cfg
