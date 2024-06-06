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
    raw, info = recorder.complete()
    interface.set_status(text="Recording Completed")
    interface.set_recording_info(info)


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(config: DictConfig):
    recorder = Recorder(sources=config.sources, buffer_size_seconds=config.buffer_size_seconds, config=config)
    interface = Interface()
    interface.set_start_action(lambda: start(recorder, interface))
    interface.set_stop_action(lambda: stop(recorder, interface))

    interface.run()
    interface.stop()
    print("Stopped Interface...")


if __name__ == "__main__":
    main()
