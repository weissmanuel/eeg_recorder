from typing import List
import numpy as np


def generate_harmonics(frequencies: List[float], duration: float, sfreq: float, num_harmonics=6):
    harmonics_dict = {}
    harmonics_list = []
    T = int(duration * sfreq)
    t = np.linspace(0, (T-1)/sfreq, T)
    for f in frequencies:
        harmonics = []
        for i in range(1, num_harmonics // 2 + 1):
            harmonics.append(np.sin(2 * np.pi * i * f * t))
            harmonics.append(np.cos(2 * np.pi * i * f * t))
        harmonics_dict[f] = np.vstack(harmonics)
        harmonics_list.append(np.vstack(harmonics))

    # return harmonics_dict, harmonics_list
    return harmonics_list
