# precondition.py
# code for preconditioning neural data
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, windows, freqz
import math
import numpy as np
import os
import matplotlib.pyplot as plt
#
import nitime.algorithms as tsa
import nitime.utils as utils
from nitime.viz import winspect
from nitime.viz import plot_spectral_estimate
'''
Filter functions
'''
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

def bandpass_butterworth_params(cutoff_low, cutoff_high, fs, order = 4, filter_type = 'bandpass'):
    '''
    Design Nth-order digital Butterworth filter and return the filter coefficients.

    Args:
        cutoff_low (int): lower cut-off frequency (in Hz)
        cutoff_high (int): higher cutoff frequency (in Hz)
        fs (int): sampling rate (in Hz)
        order (int): Order of the butter worth filter
        filter_type (str) : Type of filter. Accepts one of the four values - {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns:
        tuple (b,a): bandpass filter parameters

    '''
    nyq = 0.5 * fs
    low = cutoff_low / nyq
    high = cutoff_high / nyq

    b,a = butter( order, [cutoff_low, cutoff_high], btype=filter_type, fs =fs)
    return b,a

def bandpass_butterworth_filter_data(data, cutoff_low, cutoff_high, fs, order = 4,filter_type = 'bandpass' ):
    '''
    Apply a digital butterworth filter forward and backward to a timeseries signal.

    Args:
        data (array): neural data
        cutoff_low (float): lower cut-off frequency (in Hz)
        cutoff_high (float): higher cutoff frequency (in Hz)
        fs (int): sampling rate (in Hz)
        order: Order of the butterworth filter
        filter_type (str) : Type of filter. Accepts one of the four values - {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns:
        filtered_data: output bandpass filtered data

    '''

    b,a = butter(order, [cutoff_low, cutoff_high], btype=filter_type, fs= fs)
    # filtered_data = lfilter(b,a,data)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def get_psd_multitaper(data, fs, NW = None, BW= None, adaptive = False, jackknife = True, sides = 'default'):
    '''
     Computes power spectral density using Multitaper functions

    Args:
        data (ndarray):  time series data where time axis is assumed to be on the last axis (n_channels , n_samples)
        fs (float): sampling rate of the signal
        NW (float): Normalized half bandwidth of the data tapers in Hz
        BW (float): sampling bandwidth of the data tapers in Hz
        adaptive (bool): Use an adaptive weighting routine to combine the PSD estimates of different tapers.
        jackknife (bool): Use the jackknife method to make an estimate of the PSD variance at each point.
        sides (str): This determines which sides of the spectrum to return.

    Returns:
        f (ndarray) : Frequency points vector
        psd_est (ndarray): estimated power spectral density (PSD)
        nu (ndarray): if jackknife = True; estimated variance of the log-psd. If Jackknife = False; degrees of freedom in a chi square model of how the estimated psd is distributed wrt true log - PSD
    '''
    f, psd_mt, nu = tsa.multi_taper_psd(data, fs, NW, BW,  adaptive, jackknife, sides )
    return f, psd_mt, nu

def multitaper_lfp_bandpower(f,psd_est, bands, n_channels, no_log):
    '''
    Estimate band power in specified frequency bands using multitaper power spectral density estimate

    Args:
        psd_est (ndarray): power spectral density - output of bandpass_multitaper_filter_data
        bands (list): lfp bands should be a list of tuples representing ranges e.g., bands = [(0, 10), (10, 20), (130, 140)] for 0-10, 10-20, and 130-140 Hz
        n_channels: number of channels
        no_log (bool): boolean to select whether lfp band power should be in log scale or not

    Returns:
        lfp_power (ndarray): lfp band power for each channel for each band specified ( n_channels * n_features, 1)
    '''

    lfp_power = np.zeros((n_channels * len(bands), 1))
    small_epsilon = 0
    fft_inds = dict()

    for band_idx, band in enumerate(bands):
            fft_inds[band_idx] = [freq_idx for freq_idx, freq in enumerate(f) if band[0] <= freq < band[1]]

    for idx, band in enumerate(bands):
        if n_channels == 1:
            if no_log:
                lfp_power[idx * n_channels: (idx + 1) * n_channels, 0] = np.mean(psd_est[fft_inds[idx]], axis=0)
            else:
                lfp_power[idx * n_channels: (idx + 1) * n_channels, 0] = np.mean(np.log10(psd_est[fft_inds[idx]] + small_epsilon), axis=0)
        else:
            if no_log:
                lfp_power[idx * n_channels: (idx + 1) * n_channels, 0] = np.mean(psd_est[:,fft_inds[idx]], axis=1)
            else:
                lfp_power[idx * n_channels: (idx + 1) * n_channels, 0] = np.mean(np.log10(psd_est[:,fft_inds[idx]] + small_epsilon), axis=1)

    return lfp_power

def get_psd_welch(data, fs,l = None):
    '''
    Computes power spectral density using Welch's method

    Args:
        data (ndarray): time series data.
        fs (float): sampling rate
        l (int): no. of frequency points expected
        # average (str):  { ‘mean’, ‘median’ }, optional method to use when averaging periodograms. Defaults to ‘mean’.

    Returns:
        f (ndarray) : frequency points vector
        psd_est (ndarray): estimated power spectral density (PSD)
    '''
    if l:
        f, psd = signal.welch(data, fs, average = 'median',nperseg=2*l)
    else:
        f, psd = signal.welch(data, fs, average = 'median')
    return f,psd



