from sklearn.ensemble import RandomForestClassifier
from enum import Enum
from typing import List, Dict
from sklearn.pipeline import Pipeline, make_pipeline
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from numpy import ndarray
import numpy as np


class PipelineSteps(Enum):
    RANDOM_FOREST = 'RANDOM_FOREST'
    CHANNEL_RESHAPE = 'CHANNEL_RESHAPE'

    @staticmethod
    def from_str(label: str):
        try:
            return PipelineSteps[label.upper()]
        except KeyError:
            raise ValueError(f'Unknown pipeline step: {label}')


class ChannelReshape(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: ndarray):
        if len(X.shape) > 2:
            return np.reshape(X, (X.shape[0], -1))
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


def build_pipeline(steps: List[Dict]) -> Pipeline:
    steps = get_pipeline_steps(steps)
    return make_pipeline(*steps)


def get_pipeline_step(step: PipelineSteps | str, **kwargs):
    step = check_step_type(step)
    match step:
        case PipelineSteps.RANDOM_FOREST:
            return RandomForestClassifier(**kwargs)
        case PipelineSteps.CHANNEL_RESHAPE:
            return ChannelReshape()
        case _:
            raise ValueError(f'Unknown pipeline step: {step}')


def get_pipeline_steps(steps: List[Dict]):
    return [get_pipeline_step(step['name'], **step['kwargs']) for step in steps]


def check_step_type(step: PipelineSteps | str):
    return step if isinstance(step, PipelineSteps) else PipelineSteps.from_str(step)


def save_pipeline(pipeline: Pipeline, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
