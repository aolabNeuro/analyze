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
def generate_test_signal(T, fs, freq, a=0.02):
    '''

    Args:
        T (float): Time period in seconds
        fs (int): Sampling frequency in Hz
        freq (1D array): Frequencies to be mixed in the test signal. main frequency in the first element
        plot_true (bool): boolean value if true, plots the test signal in the provided axes
        ax (plot axis handle): axis to plot the noisy test signal

    Returns:
        x (1D array): cosine wave with multiple frequencies and noise
    '''
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    # a = 0.02
    f0 = freq[0]
    # noise_power = 0.001 * fs / 2

    x = a * np.cos(2 * np.pi * f0 * t)  # adding primary frequency
    x += a * 0.5 * np.cos(2 * np.pi * freq[1] * t)
    x += a * 1.5 * np.cos(2 * np.pi * freq[2] * t)
    x += a * 20 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))  # noise
    # x += np.random.normal(scale=np.sqrt(noise_power), size=t.shape)  # noise

    return x, t, f0

def bandpass_butterworth_params(cutoff_low, cutoff_high, fs, order = 4, filter_type = 'bandpass'):
    '''
    Args:
        cutoff_low (int): lower cut-off frequency (in Hz)
        cutoff_high (int): higher cutoff frequency (in Hz)
        fs (int): sampling rate (in Hz)
        order (int): Order of the butter worth filter
        filter_type (str) : Type of filter. Accepts one of the four values - {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns:
        bandpass filter parameters

    '''
    nyq = 0.5 * fs
    low = cutoff_low / nyq
    high = cutoff_high / nyq

    b,a = butter( order, [cutoff_low, cutoff_high], btype=filter_type, fs =fs)
    return b,a

def bandpass_butterworth_filter_data(data, cutoff_low, cutoff_high, fs, order = 4,filter_type = 'bandpass' ):
    '''
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

def bandpass_multitaper_filter_data(data, fs, NW = None, BW= None, adaptive = False, jackknife = True, sides = 'default'):
    '''

    Args:
        data (ndarray):  time series data where time axis is assumed to be on the last axis
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

#TODO: function for LFP band power

def get_psd(data, fs,l):
    f, psd = signal.welch(data, fs, average='mean',nperseg=2*l)
    return f,psd

'''
Plots to test filter performance
'''


def plot_filtered_signal(t,x, x_filter,low, high ):
    # Plotting noisy test signal and filtered signal
    plt.figure()
    plt.clf()
    plt.plot(t, x, label='Noisy signal')
    plt.plot(t, x_filter, label='Filtered signal')
    plt.xlabel('time (seconds)')
    # plt.hlines([-self.a, self.a], 0, self.T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='best')
    plt.show()

def plot_phase_locking(t, a, f0,  x_filter):
    # Plotting filtered signal with original signal frequency
    plt.figure()
    plt.clf()
    x_f0 = a * np.cos(2 * np.pi * f0 * t)
    plt.plot(t, x_f0, label='Original signal (%g Hz)' % f0)
    plt.plot(t, x_filter, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.ylabel('amplitude')
    plt.title('Comparison of Original Vs Filtered Signal')
    plt.legend(loc='best')
    plt.show()

def plot_freq_response_vs_filter_order(x,lowcut, highcut,fs):
    # Plot the frequency response for a few different orders
    plt.figure(2)
    plt.clf()
    for order in [2,3,4,5,6]: # trying  different order of butterworth to see the roll off around cut-off frequencies
        bandpass_butterworth_filter_data(x, lowcut, highcut, fs, order = order)

        b, a = bandpass_butterworth_params(lowcut, highcut,fs, order = order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Comparison of Frequency response for Diff. Orders of Butterworth Filter')
    plt.show()

def plot_psd(x,x_filter, fs):
    # Plot power spectral density of the signal
    f, psd = signal.welch(x, fs, average = 'mean')
    f_filtered, psd_filterred = signal.welch(x_filter, fs, average='median')
    plt.figure()
    plt.semilogy(f, psd, label='test signal')
    plt.semilogy(f_filtered, psd_filterred, label='filtered output')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.title('Power Spectral Density Comparison')
    plt.show()

def plot_db_spectral_estimate(freq, psd, psd_filter, labels):
    # psd = 10* np.log10(psd)
    # psd_filter = 10 * np.log10(psd_filter)
    plt.figure()
    plot_spectral_estimate(freq,psd,(psd_filter,), elabels=(labels))
    plt.show()



