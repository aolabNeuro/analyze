# precondition.py
# code for preconditioning neural data
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
import math
import numpy as np
import os
import matplotlib.pyplot as plt
#
import nitime.algorithms as tsa
import nitime.utils as utils
from nitime.viz import winspect
from nitime.viz import plot_spectral_estimate
from . import utils, analysis, preproc
'''
Filter functions
'''

def butterworth_params(cutoff_low, cutoff_high, fs, order = 4, filter_type = 'bandpass'):
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

    if filter_type == 'lowpass':
        Wn = cutoff_high

    if filter_type == 'highpass':
        Wn = cutoff_low

    if filter_type == 'bandpass' or 'bandstop':
        Wn = [cutoff_low, cutoff_high]

    b,a = butter( order, Wn, btype=filter_type, fs =fs)
    return b,a

def butterworth_filter_data(data,fs, cutoff_freq= None, bands= None,  order= None ,filter_type = 'bandpass' ):
    '''
    Apply a digital butterworth filter forward and backward to a timeseries signal.

    Args:
        data (array): neural data (n_channels x n_samples)
        fs (int): sampling rate (in Hz)
        cutoff_freq(float): cut-off frequency (in Hz); only for 'high pass' or 'low pass' filter. Use bands for 'bandpass' filter
        bands (list): frequency bands should be a list of tuples representing ranges e.g., bands = [(0, 10), (10, 20), (130, 140)] for 0-10, 10-20, and 130-140 Hz
        order: Order of the butterworth filter. If no order is specified, the function will find the minimum order of filter required to maintain +3dB gain in the bandpass range.
        filter_type (str) : Type of filter. Accepts one of the four values - {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns:
        tuple: Tuple containing:
            | **filtered_data (list):** output bandpass filtered data in the form of a list. Each list item is filtered data in corresponding frequency band
            | **Wn (list):** frequency bands
    '''

    if filter_type == 'lowpass' or 'highpass':
        Wn = cutoff_freq

    if filter_type == 'bandpass' or 'bandstop':
        if not bands:
            raise ValueError("Must provide lower and higher cut off frequency")

        Wn = np.zeros((len(bands),2))
        for ib, band in enumerate(bands):
            Wn[ib] = [band[0], band[1]]

    filtered_data = []
    for _, w in enumerate(Wn):
        if order is None:
            wp = w/(2 * fs)
            if len(wp) == 2:
                ws = [wp[0] - 0.1 , wp[1] + 1]
            else:
                ws = wp -0.1
            order, _ = signal.buttord(wp, ws, 3, 40, False)
        b,a = butter(order, w, btype=filter_type, fs= fs)
        # filtered_data = lfilter(b,a,data)
        filtered_data.append(filtfilt(b, a, data, axis = -1))
    return filtered_data, Wn

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
        tuple: Tuple containing:
            | **f (ndarray):** Frequency points vector
            | **psd_est (ndarray):** estimated power spectral density (PSD)
            | **nu (ndarray):** if jackknife = True; estimated variance of the log-psd. If Jackknife = False; degrees of freedom in a chi square model of how the estimated psd is distributed wrt true log - PSD
    '''
    f, psd_mt, nu = tsa.multi_taper_psd(data, fs, NW, BW,  adaptive, jackknife, sides )
    return f, psd_mt, nu

def multitaper_lfp_bandpower(f,psd_est, bands, n_channels, no_log):
    '''
    Estimate band power in specified frequency bands using multitaper power spectral density estimate

    Args:
        f (ndarray) : Frequency points vector
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

def get_psd_welch(data, fs,n_freq = None):
    '''
    Computes power spectral density using Welch's method. Welch’s method computes an estimate of the power spectral density by dividing the data into overlapping segments, computes a modified periodogram for each segment and then averages the periodogram. Periodogram is averaged using median.

    Args:
        data (ndarray): time series data.
        fs (float): sampling rate
        n_freq (int): no. of frequency points expected

    Returns:
        f (ndarray) : frequency points vector
        psd_est (ndarray): estimated power spectral density (PSD)
    '''
    if n_freq:
        f, psd = signal.welch(data, fs, average = 'median',nperseg=2*n_freq)
    else:
        f, psd = signal.welch(data, fs, average = 'median')
    return f,psd

'''
Spike detection functions
'''

def calc_spike_threshold(spike_filt_data, rms_multiplier=3):
    '''
    Use the RMS of each channel to set a different threshold for each channel. Sadtler et al. 2014 set threshold to 3.0x RMS value for each channel, which is the default for this function.
    The threshold value will be calculated with the mean subtracted, then the mean for each signal will be added back to the threshold value.
    
    Args:
        spike_filt_data (nt, ...): Filtered time series data.
        rms_multiplier (float): Value to multiply the rms value of each time series by.

    Returns:
        threshold_values: Threshold values along the first axis. Output dimensions will be the same non-time dimensions as the input signal.
    
    '''
    mean_input_data = np.mean(spike_filt_data, axis=0)
    rms_values = analysis.calc_rms(spike_filt_data, remove_offset=True)

    return (rms_multiplier*rms_values)+mean_input_data


def detect_spikes(spike_filt_data, samplerate, threshold, above_thresh=True, wf_length=1000):
    '''
    This function calculates spike times based on threshold crossing of the input data and returns the waveforms if 'wf_length' is not None. 
    If the threshold desired is a negative value (i.e. extracellular recordings) set 'above_thresh' to False. 
    Data must exceed the threshold instead of equaling it.

    Args:
        spike_filt_data (nt, nch): Time series neural data to detect spikes and extract waveforms from.
        samplerate (float): Sampling rate [Hz]
        threshold (nch): Threshold input data must cross to indicate a spike for each channel. Must have same non time dimensions as spike_filt_data. 
        above_thresh (bool): If True, only spikes above the threshold will be detected. If false, only spikes below threshold will be detected. 
        wf_length (fload): Length of waveforms to output [us]. Actual length will be rounded up. If set to 'None', waveforms will not be returned.
        
        
    Returns: 
        tuple: Tuple containing:
            | **spike_times (list of spike times):**  List of nspike length arrays with each list element corresponding to a channel.
            | **spike_waveforms (list of waveforms):** List of (nspike, nwf_pts) arrays with each list element corresponding to a channel. Returns NaN if there aren't enough data points to retreive a full waveform.
    '''
    nch = spike_filt_data.shape[1]

    # Calculate an array of spike times for each channel organized into a list
    if above_thresh:        
        data_above_thresh_mask = spike_filt_data > threshold
        
    elif above_thresh == False:        
        data_above_thresh_mask = spike_filt_data < threshold

    spike_times = []
    spike_waveforms =[]
    
    for ich in range(nch):
        # Spike times
        temp_spike_times, _ = preproc.detect_edges(data_above_thresh_mask[:,ich], samplerate, rising=True, falling=False)
        spike_times.append(temp_spike_times)

        # Spike waveforms
        if wf_length is not None:
            wf_idx_length = math.ceil(samplerate*(wf_length*(10**-6)))
            nspikes = len(temp_spike_times)

            temp_spike_waveforms = np.zeros((nspikes, wf_idx_length))
            temp_spike_waveforms[:] = np.nan

            for ispike in range(nspikes):
                startidx = int(np.round(temp_spike_times[ispike]*samplerate, decimals=0))
                stopidx = int(np.round(temp_spike_times[ispike]*samplerate,decimals=0) + wf_idx_length)

                if stopidx < spike_filt_data.shape[0]: # Ensure there are enough data points to grab the waveform
                  temp_spike_waveforms[ispike, :] = spike_filt_data[startidx:stopidx,ich]
        
            spike_waveforms.append(temp_spike_waveforms)

    return spike_times, spike_waveforms

def bin_spikes(data, fs, bin_width):
    '''
    Computes binned spikes [spikes/s]. The input data is the sampled thresholded data (0 or 1 data).
    Binned spikes are calculated at each bin whose width is determined by bin_width. 

    Example:
        >>> data = np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]])
        >>> data_T = data.T
        >>> fs = 100
        >>> binned_spikes = precondition.bin_spikes(data_T, fs, 5)
        >>> print(binned_spikes)
        [[40.         60.        ]
        [40.         40.        ]
        [60.         40.        ]
        [33.33333333 33.33333333]]

    Args:
        data (nt, nch): time series spike data with multiple channels.
        fs (float): sampling rate of data
        bin_width (int): Spikes are averaged within 'bin_width' then devided by 'fs'

    Returns:
        binned_spikes (nbin, nch): binned spikes [spikes/s].
    '''

    dT = 1/fs
    bin_width = round(bin_width*fs) # from [s] to index
    nbins = math.ceil(data.shape[0]/bin_width) # the number of bins
    nch = data.shape[1]
    binned_spikes = np.zeros((nbins,nch))

    for idx in range(nbins):
            if idx == nbins - 1:
                binned_spikes[idx,:] = np.mean(data[idx * bin_width :, :], axis = 0)
            else:
                binned_spikes[idx,:] = np.mean(data[idx * bin_width : (idx + 1) * bin_width, :], axis = 0)

    binned_spikes = binned_spikes/dT # convert from [spikes/bin] to [spikes/s]    
    return binned_spikes
