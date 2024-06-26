from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.cross_decomposition import CCA
import scipy
from scipy.stats import pearsonr, mode
import numpy as np
from numpy import ndarray
from typing import List
from scipy.special import softmax


def filterbank(data: ndarray, fs: float, idx_fb: int = None) -> ndarray:
    if idx_fb == None:
        print('stats:filterbank:MissingInput ' \
              + 'Missing filter index. Default value (idx_fb = 0) will be used.')
        idx_fb = 0
    elif (idx_fb < 0 or 9 < idx_fb):
        raise ValueError('stats:filterbank:InvalidInput ' \
                         + 'The number of sub-bands must be 0 <= idx_fb <= 9.')

    if (len(data.shape) == 2):
        num_chans = data.shape[0]
        num_trials = 1
    else:
        _, num_chans, num_trials = data.shape

    # Nyquist Frequency = Fs/2N
    Nq = fs / 2

    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    Wp = [passband[idx_fb] / Nq, 90 / Nq]

    # print("Wp: ", Wp)
    Ws = [stopband[idx_fb] / Nq, 100 / Nq]

    # print("Ws: ", Ws)
    [N, Wn] = scipy.signal.cheb1ord(Wp, Ws, 3, 40)  # band pass filter StopBand=[Ws(1)~Ws(2)] PassBand=[Wp(1)~Wp(2)]
    [B, A] = scipy.signal.cheby1(N, 0.5, Wn, 'bandpass')  # Wn passband edge frequency

    y = np.zeros(data.shape)

    if (num_trials == 1):
        for ch_i in range(num_chans):
            # apply filter, zero phass filtering by applying a linear filter twice, once forward and once backwards.
            # to match matlab result we need to change padding length
            y[ch_i, :] = scipy.signal.filtfilt(B, A, data[ch_i, :])

    else:
        for trial_i in range(num_trials):
            for ch_i in range(num_chans):
                y[:, ch_i, trial_i] = scipy.signal.filtfilt(B, A, data[:, ch_i, trial_i])
    return y


def cca_reference(list_freqs: List[float] | ndarray, fs: float, n_samples: int, n_harmonics: int = 3):
    num_freqs = len(list_freqs)
    t = np.arange(1, n_samples + 1) / fs  # time index

    y_ref = np.zeros((num_freqs, 2 * n_harmonics, n_samples))
    for freq_i in range(num_freqs):
        tmp = []
        for harm_i in range(1, n_harmonics + 1):
            f = list_freqs[freq_i]  # in HZ
            # Sin and Cos
            tmp.extend([np.sin(2 * np.pi * harm_i * f * t),
                        np.cos(2 * np.pi * harm_i * f * t)])
        y_ref[freq_i] = tmp  # 2*num_harms because include both sin and cos

    return y_ref


class CCAClassifier(TransformerMixin, ClassifierMixin):
    ref_signal: ndarray | None = None
    target_frequencies: ndarray

    def __init__(self,
                 target_frequencies: List[float] | ndarray,
                 signal_duration_seconds: float,
                 sfreq: float,
                 n_components: int = 1,
                 n_filterbanks: int = 3,
                 n_harmonics: int = 6):
        if isinstance(target_frequencies, list):
            target_frequencies = np.array(target_frequencies)

        self.target_frequencies = target_frequencies
        self.signal_duration_seconds = signal_duration_seconds
        self.sfreq = sfreq
        self.n_components = n_components
        self.n_filterbanks = n_filterbanks
        self.n_harmonics = n_harmonics

    def fit(self, X, y):
        return self

    def calculate_correlation(self, X):
        filterbank_coefficients = np.power(np.arange(1, self.n_filterbanks + 1), (-1.25)) + 0.25
        n_classes = len(self.target_frequencies)
        batch_size, n_chans, n_samples = X.shape
        y_ref = cca_reference(self.target_frequencies, self.sfreq, n_samples, self.n_harmonics)
        cca = CCA(n_components=self.n_components)

        result = np.zeros((self.n_filterbanks, n_classes))
        pred = np.zeros(batch_size)
        correlations = np.zeros((batch_size, n_classes))

        for trial_idx, trial in enumerate(X):
            for fb_idx in range(self.n_filterbanks):
                for class_idx in range(n_classes):
                    x = filterbank(trial, self.sfreq, fb_idx)
                    y = y_ref[class_idx]
                    Xw, Yw = cca.fit_transform(x.T, y.T)
                    r_tmp, _ = pearsonr(np.squeeze(Xw), np.squeeze(Yw))
                    if np.isnan(r_tmp):
                        r_tmp = 0
                    result[fb_idx, class_idx] = r_tmp

            rho = np.dot(filterbank_coefficients, result)
            f_max = np.argmax(rho)
            pred[trial_idx] = f_max
            correlations[trial_idx] = abs(rho)
        return pred, correlations

    def predict(self, X):
        pred, _ = self.calculate_correlation(X)
        return pred

    def predict_proba(self, X):
        _, correlations = self.calculate_correlation(X)
        return softmax(correlations, axis=1)


