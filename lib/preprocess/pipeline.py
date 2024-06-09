from mne.io import BaseRaw
from enum import Enum
from sklearn.base import BaseEstimator, TransformerMixin
from numpy import ndarray
import numpy as np


class PipelinePreprocessors(Enum):
    Scaler = "scaler"


def _sklearn_reshape_apply(func, return_result, X, *args, **kwargs):
    """Reshape epochs and apply function."""
    if not isinstance(X, np.ndarray):
        raise ValueError("data should be an np.ndarray, got %s." % type(X))
    orig_shape = X.shape
    X = np.reshape(X.transpose(0, 2, 1), (-1, orig_shape[1]))
    X = func(X, *args, **kwargs)
    if return_result:
        X.shape = (orig_shape[0], orig_shape[2], orig_shape[1])
        X = X.transpose(0, 2, 1)
        return X


class CustomScaler(TransformerMixin, BaseEstimator):
    """Standardize channel data.

    This class scales data for each channel. It differs from scikit-learn
    classes (e.g., :class:`sklearn.preprocessing.StandardScaler`) in that
    it scales each *channel* by estimating μ and σ using data from all
    time points and epochs, as opposed to standardizing each *feature*
    (i.e., each time point for each channel) by estimating using μ and σ
    using data from all epochs.

    Parameters
    ----------
    scalings : str, default None
        Scaling method to be applied to data channel wise.
        * if ``scalings=='median'``,
          :class:`sklearn.preprocessing.RobustScaler`
          is used (requires sklearn version 0.17+).
        * if ``scalings=='mean'``,
          :class:`sklearn.preprocessing.StandardScaler`
          is used.
        * if ``scalings=='minmax'``,
         :class:`sklearn.preprocessing.MinMaxScaler`
         is used.>

    with_mean : bool, default True
        If True, center the data using mean (or median) before scaling.
        Ignored for channel-type scaling.
    with_std : bool, default True
        If True, scale the data to unit variance (``scalings='mean'``),
        quantile range (``scalings='median``), or using channel type
        if ``scalings`` is a dict or None).
    feature_range : tuple, default (0, 1)
        Desired range of transformed data after scaling when using minmax scaling.
    """

    def __init__(self, scalings: str = 'mean', with_mean: bool = True, with_std: bool = True,
                 feature_range: tuple = (0, 1)):
        self.with_mean = with_mean
        self.with_std = with_std
        self.scalings = scalings
        self.feature_range = feature_range

        if scalings is None:
            raise ValueError(
                'Need to specify "scalings" if scalings is' "%s" % type(scalings)
            )
        if isinstance(scalings, str):
            assert scalings in ("mean", "median", "minmax"), ("scalings should be mean (StandardScaler) or median "
                                                              "(RobustScaler) or minmax (MinMaxScaler)")
        if scalings == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            self._scaler = MinMaxScaler(feature_range=feature_range)
        elif scalings == "mean":
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler(
                with_mean=self.with_mean, with_std=self.with_std
            )
        else:  # scalings == 'median':
            from sklearn.preprocessing import RobustScaler

            self._scaler = RobustScaler(
                with_centering=self.with_mean, with_scaling=self.with_std
            )

    def fit(self, epochs_data: ndarray, y: ndarray | None = None):
        assert isinstance(epochs_data, np.ndarray), f"Data must be of type np.ndarray, got {type(epochs_data)}"
        if epochs_data.ndim == 2:
            epochs_data = epochs_data[..., np.newaxis]
        assert epochs_data.ndim == 3, epochs_data.shape
        _sklearn_reshape_apply(self._scaler.fit, False, epochs_data, y=y)
        return self

    def transform(self, epochs_data: ndarray):
        assert isinstance(epochs_data, np.ndarray), f"Data must be of type np.ndarray, got {type(epochs_data)}"
        if epochs_data.ndim == 2:  # can happen with SlidingEstimator
            epochs_data = epochs_data[..., np.newaxis]
        assert epochs_data.ndim == 3, epochs_data.shape
        return _sklearn_reshape_apply(self._scaler.transform, True, epochs_data)

    def fit_transform(self, epochs_data: ndarray, y: ndarray | None = None, **fit_params):
        return self.fit(epochs_data, y).transform(epochs_data)

    def inverse_transform(self, epochs_data: ndarray):
        assert epochs_data.ndim == 3, epochs_data.shape
        return _sklearn_reshape_apply(self._scaler.inverse_transform, True, epochs_data)
