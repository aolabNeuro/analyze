# utils.py
# all extra utility functions belong here
import numpy as np


def generate_test_signal(T, fs, freq, a):
    '''
    Generates a test time series signal with multiple frequencies, specified in freq, for T timelength at a sampling rate of fs

    Args:
        T (float): Time period in seconds
        fs (int): Sampling frequency in Hz
        freq (1D array): Frequencies to be mixed in the test signal. main frequency in the first element
        a (1D array) : amplitudes for each frequencies and last element of the array to be amplitude of noise (size : len(freq) + 1)

    Returns:
        x (1D array): cosine wave with multiple frequencies and noise
        t (1D array): time vector for x
    '''
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    # a = 0.02
    f0 = freq[0]
    # noise_power = 0.001 * fs / 2
    x = a[-1] * np.sin(2 * np.pi * 1.2 * np.sqrt(t))  # noise
    # x += np.random.normal(scale=np.sqrt(noise_power), size=t.shape)  # noise

    for i in range(len(freq)):
        x += a[i] * np.cos(2 * np.pi * freq[i] * t)

    return x, t
