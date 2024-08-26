from mne_lsl.lsl import local_clock, StreamInlet, resolve_streams


def print_stream(stream, idx: int = 0):
    print("\n")
    print("---------------------------------------------------")
    print(f"Stream {idx + 1}: {stream.name}")
    print("---------------------------------------------------")
    print(f"Name: {stream.name}")
    print(f"Source_ID: {stream.source_id}")
    print(f"Number of channels: {stream.n_channels}")
    print(f"Sampling Frequency (sfreq): {stream.sfreq}")
    print(f"Stream Type (stype): {stream.stype}")
    print(f"Stream Dtype (dtype): {stream.dtype}")
    print(f"UID: {stream.uid}")
    print(f"Hostname: {stream.hostname}")
    print("---------------------------------------------------")
    print("\n")


if __name__ == "__main__":
    streams = resolve_streams()
    print(f"Streams found: {len(streams)}")
    if len(streams) > 0:
        for i, stream in enumerate(streams):
            print_stream(stream, i)
