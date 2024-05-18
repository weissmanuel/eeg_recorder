from lib.recorder import Recorder
from lib.interface import Interface


def start(recorder: Recorder, interface: Interface):
    interface.set_status(text="Start Recording...")
    recorder.start()
    interface.set_status(text="Recording...")


def stop(recorder: Recorder, interface: Interface):
    interface.set_status(text="Stop Recording...")
    raw, info = recorder.complete("./data/recordings/test_raw.fif")
    interface.set_status(text="Recording Completed")
    interface.set_recording_info(info)


def main():
    # recorder = Recorder(signal_id='UN-2023.05.69', marker_id='rsvp_markers', buffer_size_seconds=60)
    recorder = Recorder(signal_id='rsvp_eeg', marker_id='rsvp_markers', buffer_size_seconds=60)
    interface = Interface()
    interface.set_start_action(lambda: start(recorder, interface))
    interface.set_stop_action(lambda: stop(recorder, interface))

    interface.run()


if __name__ == "__main__":
    main()
