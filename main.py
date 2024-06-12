from lib.recorder import Recorder
from lib.interface import Interface
import hydra
from omegaconf import DictConfig


def start_recording(recorder: Recorder, interface: Interface):
    interface.set_recording_status(text="Start Recording...")
    recorder.start_recording()
    interface.set_recording_status(text="Recording...")


def stop_recording(recorder: Recorder, interface: Interface):
    interface.set_recording_status(text="Stop Recording...")
    raw, info = recorder.complete_recording()
    interface.set_recording_status(text="Recording Completed")
    interface.set_recording_info(info)

def start_training(recorder: Recorder, interface: Interface):
    interface.set_training_status(text="Start Training...")
    recorder.start_training()
    interface.set_training_status(text="Training completed")


def start_inference(recorder: Recorder, interface: Interface):
    interface.set_inference_status(text="Start Inference...")
    recorder.start_inference()
    interface.set_inference_status(text="Inference...")


def stop_inference(recorder: Recorder, interface: Interface):
    interface.set_inference_status(text="Stop Inference...")
    recorder.stop_inference()
    interface.set_inference_status(text="Inference Completed")


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(config: DictConfig):
    recorder = Recorder(sources=config.sources, buffer_size_seconds=config.buffer_size_seconds, config=config)
    interface = Interface()
    interface.set_recording_start_action(lambda: start_recording(recorder, interface))
    interface.set_recording_stop_action(lambda: stop_recording(recorder, interface))
    interface.set_training_start_action(lambda: start_training(recorder, interface))
    interface.set_inference_start_action(lambda: start_inference(recorder, interface))
    interface.set_inference_stop_action(lambda: stop_inference(recorder, interface))

    interface.run()
    recorder.kill()
    interface.stop()
    print("Stopped Interface...")


if __name__ == "__main__":
    main()
