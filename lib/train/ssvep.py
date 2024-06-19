from omegaconf import DictConfig
from lib.mne import load_raw, generate_epochs
from pathlib import Path
from lib.preprocess.raw_preprocess import RawNotchFilter, RawFilter, Resample
from lib.preprocess.epoch_preprocess import EpochFilter
from lib.preprocess.data_preprocess import EpochWindowSplitter
from sklearn.pipeline import Pipeline
from lib.train.pipeline import build_pipeline, save_pipeline
from sklearn.metrics import classification_report
from lib.utils import config_to_primitive
from sklearn.utils import shuffle as sk_shuffle


def get_events_mapping() -> dict:
    return {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
    }


def map_labels(labels: list, real_labels: list) -> list:
    real_labels = config_to_primitive(real_labels)
    return [real_labels[label] for label in labels]


def train_ssvep_classifier(config: DictConfig, file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'File {file_path} does not exist')

    sfreq = config.headset.sfreq
    signal_duration_seconds = config.experiment.signal_duration_seconds
    window_size_seconds = config.experiment.window_size_seconds
    window_shift_seconds = config.experiment.window_shift_seconds
    event_offset_seconds = config.experiment.event_offset_seconds


    raw = load_raw(file_path)
    raw = RawNotchFilter(freqs=50).preprocess(raw)
    raw = RawFilter(9, 23).preprocess(raw)
    raw = raw.set_eeg_reference(ref_channels='average', projection=False)
    raw = Resample(sfreq=sfreq).preprocess(raw)

    epochs = generate_epochs(raw, event_mapping=get_events_mapping(),
                             t_min=0, t_max=signal_duration_seconds)
    epochs = EpochFilter(low_freq=9, high_freq=23).preprocess(epochs)

    num_classes = len(config.experiment.labels)
    num_events = len(epochs.events)
    num_iterations = int(num_events // num_classes)
    num_train_events = (num_iterations - 1) * num_classes

    train_epochs = epochs[:num_train_events]
    test_epochs = epochs[num_train_events:]

    splitter = EpochWindowSplitter(sfreq=sfreq,
                                   window_size_seconds=window_size_seconds,
                                   window_shift_seconds=window_shift_seconds,
                                   use_averaging=False,)

    X_train, y_train = splitter(train_epochs.get_data(copy=True), train_epochs.events[:, 2])
    X_train, y_train = sk_shuffle(X_train, y_train)
    X_test, y_test = splitter(test_epochs.get_data(copy=True), test_epochs.events[:, 2])

    pipeline: Pipeline = build_pipeline(config.experiment.training.pipeline)
    pipeline.fit(X_train, y_train)

    save_pipeline(pipeline, config.experiment.training.pipeline_path)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    print(classification_report(y_test, y_pred))
