from lib.recorder import Recorder
from lib.interface import Interface
import hydra
from omegaconf import DictConfig


def start(recorder: Recorder, interface: Interface):
    interface.set_recording_status(text="Start Recording...")
    recorder.start()
    interface.set_recording_status(text="Recording...")


def stop(recorder: Recorder, interface: Interface):
    interface.set_recording_status(text="Stop Recording...")
    raw, info = recorder.complete()
    interface.set_recording_status(text="Recording Completed")
    interface.set_recording_info(info)


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(config: DictConfig):
    recorder = Recorder(sources=config.sources, buffer_size_seconds=config.buffer_size_seconds, config=config)
    interface = Interface(plot_store=recorder.plot_store)
    interface.set_recording_start_action(lambda: start(recorder, interface))
    interface.set_recording_stop_action(lambda: stop(recorder, interface))

    interface.run()
    recorder.kill()
    interface.stop()
    print("Stopped Interface...")


if __name__ == "__main__":
    main()
