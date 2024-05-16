from datetime import timedelta


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
