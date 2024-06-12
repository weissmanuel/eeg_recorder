from omegaconf import DictConfig
from lib.mne import load_raw, generate_epochs
from pathlib import Path
from lib.preprocess.raw_preprocess import RawNotchFilter
from lib.preprocess.epoch_preprocess import EpochFilter
from lib.preprocess.data_preprocess import EpochWindowSplitter, FFT
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from lib.train.pipeline import build_pipeline, save_pipeline
from sklearn.metrics import classification_report
from lib.utils import config_to_primitive


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

    raw = load_raw(file_path)
    raw = RawNotchFilter(freqs=50).preprocess(raw)

    epochs = generate_epochs(raw, event_mapping=get_events_mapping(), t_min=0, t_max=10)
    epochs = EpochFilter(low_freq=8, high_freq=28).preprocess(epochs)

    splitter = EpochWindowSplitter(sfreq=250, window_size_seconds=5, window_shift_seconds=0.1, use_averaging=True, average_size=5)

    data = epochs.get_data()
    labels = epochs.events[:, 2]

    windows, window_labels = splitter(data, labels)

    X = windows
    y = window_labels
    y = map_labels(y, config.experiment.labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline: Pipeline = build_pipeline(config.experiment.training.pipeline)
    pipeline.fit(X_train, y_train)

    save_pipeline(pipeline, config.experiment.training.pipeline_path)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))



