from lib.recorder import Recorder
from lib.interface import Interface
import hydra
from omegaconf import DictConfig

def start(recorder: Recorder, interface: Interface):
    interface.set_status(text="Start Recording...")
    recorder.start()
    interface.set_status(text="Recording...")


def stop(recorder: Recorder, interface: Interface):
    interface.set_status(text="Stop Recording...")
    raw, info = recorder.complete("./data/recordings/test_raw.fif")
    interface.set_status(text="Recording Completed")
    interface.set_recording_info(info)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(config: DictConfig):
    # recorder = Recorder(signal_id='UN-2023.05.69', marker_id=None, buffer_size_seconds=60)
    recorder = Recorder(signal_id=config.signal_id, marker_id=config.marker_id,
                        buffer_size_seconds=config.buffer_size_seconds, configs=config)
    # recorder = Recorder(signal_id='rsvp_pacemaker', marker_id='rsvp_markers', buffer_size_seconds=60)
    interface = Interface()
    interface.set_start_action(lambda: start(recorder, interface))
    interface.set_stop_action(lambda: stop(recorder, interface))

    interface.run()


if __name__ == "__main__":
    main()
