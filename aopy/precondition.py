# precondition.py
# Code for cleaning and preparing neural data for users to interact with,
# for example: down-sampling, outlier detection, and initial filtering

from os import fsdecode, fspath, fstat
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, decimate, windows
import numpy as np
import math
import nitime.algorithms as tsa
from . import analysis, utils
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

    if filter_type in ['bandpass', 'bandstop']:
        Wn = [cutoff_low, cutoff_high]

    b,a = butter( order, Wn, btype=filter_type, fs =fs)
    return b,a

def butterworth_filter_data(data, fs, cutoff_freq=None, bands=None, order=None, filter_type='bandpass'):
    '''
    Apply a digital butterworth filter forward and backward to a timeseries signal.

    Args:
        data (nt, ...): neural data
        fs (int): sampling rate (in Hz)
        cutoff_freq (float): cut-off frequency (in Hz); only for 'high pass' or 'low pass' filter. Use bands for 'bandpass' filter
        bands (list): frequency bands should be a list of tuples representing ranges e.g., bands = [(0, 10), (10, 20), (130, 140)] for 0-10, 10-20, and 130-140 Hz
        order (int): Order of the butterworth filter. If no order is specified, the function will find the minimum order of filter required to maintain +3dB gain in the bandpass range.
        filter_type (str) : Type of filter. Accepts one of the four values - {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns:
        tuple: Tuple containing:
            | **filtered_data (list):** output bandpass filtered data in the form of a list. Each list item is filtered data in corresponding frequency band
            | **Wn (list):** frequency bands
    '''

    if filter_type in ['lowpass', 'highpass']:
        Wn = cutoff_freq

    if filter_type in ['bandpass', 'bandstop']:
        if not bands:
            raise ValueError("Must provide lower and higher cut off frequency")

        Wn = np.zeros((len(bands),2))
        for ib, band in enumerate(bands):
            Wn[ib] = [band[0], band[1]]

    filtered_data = []
    for _, w in enumerate(Wn):

        # Automatically select an order if none is specified
        if order is None:
            wp = w/(2 * fs)
            if len(wp) == 2:
                ws = [wp[0] - 0.1 , wp[1] + 1]
            else:
                ws = wp -0.1
            order, _ = signal.buttord(wp, ws, 3, 40, False)

        # Filter this frequency or band
        b, a = butter(order, w, btype=filter_type, fs= fs)
        # filtered_data = lfilter(b,a,data)
        filtered_data.append(filtfilt(b, a, data, axis=0))
    return filtered_data, Wn

'''
Filter functions related to multitaper method
'''
def dpsschk(tapers):
    '''
    Computes the Discrete Prolate Spheroidal Sequences (DPSS) array based on input TAPERS

    Args:
        tapers (list): tapers in [N, NW, K] form. N is window length and NW is standardized half bandwidth. K is the number of DPSS you use.

    Returns:
        e (N, K): K DPSS windows
        v (K): The concentration ratios for K windows.
    '''

    length = len(tapers)
    flag = 0

    if length == 3:
        flag = 1

    if flag:
        N = tapers[0]
        NW = tapers[1]
        K = tapers[2]

        N = round(N)

        if K < 1:
            raise Exception('Error:  K must be greater than or equal to 1')
        elif K > 2*NW-1:
            raise Exception('Error:  K must be less than 2*P-1')

        e, v = windows.dpss(N, NW, K,return_ratios=True)
        e = e.T

    else:
        print('Tapers already calculated')
        e = tapers
        v = 0
    
    return e, v

def dp_proj(tapers, fs=1, f0=0):
    '''
    Generates a prolate projection operator

    Args:
        tapers(list):   tapers in [N, NW] or [N, P, K] form. If tapers in [N, NW] form, tapers is converted into [N, P, K] form 
                        P is computed by N*NW and K is given by math.floor(2*P-1)
                        (N*smapling rate) represents duration of data. 
                        NW is standardized half bandwidth corresponding to 2*NW = BW/f0 = BW*N*dt where dt is taken as 1.
                        K is the number of tapers you use.
        fs (float): sampling rate
        f0 (float): center frequency of filiter

    Returns:
        dp_proj (nt, K): projection operator in [time, K] form
    '''

    if len(tapers) == 2:
        n = tapers[0]
        w = tapers[1]
        p = n*w
        k = math.floor(2*p-1)
        tapers = [n, p, k]

    if len(tapers) == 3:
        tapers[0] = round(tapers[0]*fs)
        tapers, v = dpsschk(tapers)

    # determine parameters and assign matrices
    N = tapers.shape[0]
    K = tapers.shape[1]
    pr_op = np.zeros((N,K),dtype = 'complex')

    f0 = 0
    shifter = np.exp(-2.*np.pi*1j*f0*np.arange(1,N+1)/fs)
    for k in range(K):
        pr_op[:,k] = shifter*tapers[:,k]
        
    return pr_op

def mtfilt(tapers, fs=1, f0=0):
    '''
    Generates a bandpass filter using the multitaper method

    Args:
        tapers(list):   tapers in [N, NW] or [N, P, K] form. If tapers in [N, NW] form, tapers is converted into [N, P, K] form 
                        P is computed by N*NW and K is given by math.floor(2*P-1)
                        (N*smapling rate) represents duration of data. 
                        NW is standardized half bandwidth corresponding to 2*NW = BW/f0 = BW*N*dt where dt is taken as 1.
                        K is the number of tapers you use.
        fs (float): sampling rate
        f0 (float): center frequency of filiter

    Returns:
        X (1, 2*N*fs): a bandpass filter
    '''    
    pr_op = dp_proj(tapers, fs, f0)
    N = pr_op.shape[0]
    X = np.zeros( (1,2*N), dtype = 'complex')
    pr = pr_op@pr_op.T

    for t in range(N):
        X[0, t:t+N] = X[0, t:t+N] + pr[:,N-1-t].T
    X = X/N 
    
    return X

def mtfilter(X, tapers, fs=1, f0=0, flag=False, complexflag=False):
    '''
    Bandpass-filter a time series data using the multitaper method

    Example:
        band = [-500, 500] # signals within band can pass
        N = 0.005 # N*sampling_rate is time window you analyze
        NW = (band[1]-band[0])/2
        T = 0.05
        fs = 25000
        nch = 1
        x_312hz = utils.generate_multichannel_test_signal(T, fs, nch, 312, self.a*1.5)
        x_600hz = utils.generate_multichannel_test_signal(T, fs, nch, self.freq[0], self.a*0.5)
        f0 = np.mean(band)
        tapers = [N, NW]
        x_mtfilter = precondition.mtfilter(x_312hz + x_600hz, tapers, fs=fs, f0=f0)
        plt.figure()
        plt.plot(x_312hz + x_600hz, label='Original signal (312 Hz + 600 Hz)')
        plt.plot(x_312hz, label='Original signal (312 Hz)')
        plt.plot(x_mtfilter, label='Multitaper-filtered signal')
        plt.xlim([0,500])
        plt.legend()

        .. image:: _images/mtfilter.png

    Args:
        X (nt, nch): time series array
        tapers(list):   tapers in [N, NW] or [N, P, K] form. If tapers in [N, NW] form, tapers is converted into [N, P, K] form 
                        P is computed by N*NW and K is given by math.floor(2*P-1)
                        (N*smapling rate) represents duration of data. 
                        NW is standardized half bandwidth corresponding to 2*NW = BW/f0 = BW*N*dt where dt is taken as 1.
                        K is the number of tapers you use.
        fs (float): sampling rate
        f0 (float): center frequency of filiter
        flag (bool): if flag == 0, output data is centered. Otherwise, output data is not centered.
        complexflag: if complexflag == 0, output data becomes real. Otherwise, output data becomes complex.

    Returns:
        Y (nt, nch): filtered time-series data
    '''   

    if len(tapers) == 2:
        n = tapers[0]
        w = tapers[1]
        p = n*w
        k = math.floor(2*p-1)
        tapers = [n, p, k]
    elif len(tapers) == 3:
        tapers[0] = tapers[0]*fs
        tapers = dpsschk(tapers)

    X = X.T
    filt = mtfilt(tapers, fs, f0)
    N = filt.shape[1]
    szX = X.shape

    if complexflag == 0:
        filt = filt.real

    if min(szX) > 1:
        Y = np.zeros(szX) + 1j*np.zeros(szX)
        for ii in range(szX[0]):
            tmp = np.convolve(X[ii,:], filt[0,:])
            if flag:
                Y[ii,:] = tmp[N:szX[1]+N]
            else:
                Y[ii,:] = tmp[round(N/2):szX[1]+round(N/2)]

    else:
        Y = np.convolve(X, filt[0,:])
        if flag:
            Y[ii,:] = Y[N:szX[1]+N]
        else:
            Y[ii,:] = Y[round(N/2):szX[1]+round(N/2)]

    if f0 > 0:
        Y = 2*Y
    
    Y = Y.T
    return Y


'''
Spike detection functions
'''

def calc_spike_threshold(spike_filt_data, high_threshold=True, rms_multiplier=3):
    '''
    Use the RMS of each channel to set a different threshold for each channel. Sadtler et al. 2014 set threshold to 3.0x RMS value for each channel, which is the default for this function.
    The threshold value will be calculated with the mean subtracted, then the mean for each signal will be added back to the threshold value.
    
    Args:
        spike_filt_data (nt, ...): Filtered time series data.
        high_threshold (bool): If the threshold should allow spikes to be detected above the threshold (True) or below the threshold (False). Defaults to true.
        rms_multiplier (float): Value to multiply the rms value of each time series by.

    Returns:
        threshold_values: Threshold values along the first axis. Output dimensions will be the same non-time dimensions as the input signal.
    
    '''
    mean_input_data = np.mean(spike_filt_data, axis=0)
    rms_values = analysis.calc_rms(spike_filt_data, remove_offset=True)

    if high_threshold:
        return (rms_multiplier*rms_values)+mean_input_data
    else:
        return mean_input_data-(rms_multiplier*rms_values)


def detect_spikes(spike_filt_data, samplerate, threshold, above_thresh=True,  wf_length=1000, tbefore_wf=200):
    '''
    This function calculates spike times based on threshold crossing of the input data and returns the waveforms if 'wf_length' is not None. 
    If the threshold desired is a negative value (i.e. extracellular recordings) set 'above_thresh' to False. 
    Data must exceed the threshold instead of equaling it. 

    Args:
        spike_filt_data (nt, nch): Time series neural data to detect spikes and extract waveforms from.
        samplerate (float): Sampling rate [Hz]
        threshold (nch): Threshold that input data must cross to indicate a spike for each channel. Must have same non time dimensions as spike_filt_data. 
        above_thresh (bool): If True, only spikes above the threshold will be detected. If false, only spikes below threshold will be detected. 
        wf_length (float): Length of waveforms to output :math:`[\mu s]`. Actual length will be rounded up. If set to 'None', waveforms will not be returned. Defaults to 1000 :math:`\mu s`
        tbefore_wf (float): Length of waveform to include before spike detection time :math:`[\mu s]`. Defaults to 200 :math:`\mu s`  
        
    Returns: 
        tuple: Tuple containing:
            | **spike_times (list of spike times):**  List of nspike length arrays with each list element corresponding to a channel. Spike times are in seconds.
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
        all_spike_times, _ = utils.detect_edges(data_above_thresh_mask[:,ich], samplerate, rising=True, falling=False)
        spike_times.append(all_spike_times)


        # Spike waveforms
        if wf_length is not None:
            wf_idx_length = math.ceil(samplerate*(wf_length*(1e-6)))
            nspikes = len(all_spike_times)

            temp_spike_waveforms = np.zeros((nspikes, wf_idx_length))
            temp_spike_waveforms[:] = np.nan

            for ispike in range(nspikes):
                starttime = all_spike_times[ispike]-(tbefore_wf*(1e-6))
                startidx = int(np.round(starttime*samplerate, decimals=0))
                stopidx = startidx + wf_idx_length

                if np.logical_and(stopidx < spike_filt_data.shape[0], startidx>=0): # Ensure there are enough data points to grab the waveform
                    temp_spike_waveforms[ispike, :] = spike_filt_data[startidx:stopidx,ich]
        
            spike_waveforms.append(temp_spike_waveforms)

    return spike_times, spike_waveforms

def filter_spike_times_fast(spike_times, refractory_period=100):
    '''
    This function takes spike times and filters them to remove spikes within the refractory period of the preceding spike. This function always assumes the first threshold crossing is a good spike.
    Note: This function will remove spikes that are in the refractory period of the preceding spike even if the the preceding spike is in the refractory period of its preceding spike even though that spike shouldn't be removed.
    For example if the refractory period is set to 100 :math:`\mu s` and there are spikes at 50 :math:`\mu s`, 125 :math:`\mu s`, and 200 :math:`\mu s`, the spikes at 125 :math:`\mu s` and 200 :math:`\mu s` will be removed even though only the 125 :math:`\mu s` spike should be removed.
    If the refractory period is small enough this shouldn't be a problem. For more thorough spike time filter use aopy.precondition.filter_spike_times although it takes ~50x longer.

    Args:
        spike_times (nspikes): 1D array of spike times.
        refractory_period (float): Length of time after a spike is detected before another spike can be detected :math:`[\mu s]`. Defaults to 100 :math:`\mu s`.

    Returns: 
        tuple: A tuple containing:
            | **filtered_spike_times (ngoodspikes):** Spikes times after preliminary filtering to remove spikes within the refractory period of the preceding spike.
            | **filtered_spike_idx (ngoodspikes):** Indices corresponding to accepted spike times
    '''
    filtered_spike_idx = np.where(np.diff(spike_times, prepend=0) > refractory_period*(1e-6))[0]

    # Assume first spike is always good. If it isn't included, add it back in
    if len(np.where(filtered_spike_idx==0)[0])==0:
        filtered_spike_idx = np.insert(filtered_spike_idx,0,0)

    filtered_spike_times = spike_times[filtered_spike_idx]

    return filtered_spike_times, filtered_spike_idx

def filter_spike_times(spike_times, refractory_period=100):
    '''
    This function takes spike times and filters them to remove spikes within the refractory period of the preceding spike. This function always assumes the first threshold crossing is a good spike.
    This function works by jumping to and recording the next spike time after the refractory period ends from the current spike.

    Args:
        spike_times (nspikes): 1D array of spike times.
        refractory_period (float): Length of time after a spike is detected before another spike can be detected :math:`[\mu s]`. Defaults to 100 :math:`\mu s` .

    Returns: 
        tuple: A tuple containing:
            | **filtered_spike_times (ngoodspikes):** Spikes times after preliminary filtering to remove spikes within the refractory period of the preceding spike.
            | **filtered_spike_idx (ngoodspikes):** Indices corresponding to accepted spike times
    '''

    nspikes = len(spike_times)
    filtered_spike_idx = np.zeros(nspikes, dtype=int)
    filtered_spike_idx[0] = spike_times[0] # Store first spike time
    
    ispike = 0
    counter = 0
  
    while ispike <= nspikes:
        
        # Store current spike time
        current_spike_time = spike_times[ispike]
        filtered_spike_idx[counter] = ispike

        # Get next acceptable spike time and idx
        remaining_spikes = np.where(spike_times > current_spike_time+(refractory_period*(1e-6)))[0]
        
        # If there is no next spike, exit loop, else store it.
        if len(remaining_spikes) == 0:
          break
        else:
          ispike = remaining_spikes[0]

        counter += 1

    # Remove trailing zeros
    filtered_spike_idx = np.trim_zeros(filtered_spike_idx, 'b')
    
    # Get filtered spike times
    filtered_spike_times = spike_times[filtered_spike_idx]

    return filtered_spike_times, filtered_spike_idx


def bin_spikes(data, fs, bin_width):
    '''
    Computes binned spikes [spikes/s]. The input data is the sampled thresholded data (0 or 1 data).
    Binned spikes are calculated at each bin whose width is determined by bin_width. 

    Example:
        >>> data = np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]])
        >>> data_T = data.T
        >>> fs = 100
        >>> binned_spikes = precondition.bin_spikes(data_T, fs, .05)
        >>> print(binned_spikes)
            [[200. 300.]
            [200. 200.]
            [300. 200.]
            [100. 100.]]

    Args:
        data (nt, nch): time series spike data with multiple channels.
        fs (float): sampling rate of data [Hz]
        bin_width (float): [s] Spikes are summed within 'bin_width' then devided by 'fs'

    Returns:
        binned_spikes (nbin, nch): binned spikes [spikes/s].
    '''

    dT = 1/fs
    bin_width_idx = round(bin_width*fs) # from [s] to index
    nbins = math.ceil(data.shape[0]/bin_width_idx) # the number of bins
    nch = data.shape[1]
    binned_spikes = np.zeros((nbins,nch))

    for idx in range(nbins):
        if idx == nbins - 1:
            binned_spikes[idx,:] = np.sum(data[idx * bin_width_idx :, :], axis = 0)
        else:
            binned_spikes[idx,:] = np.sum(data[idx * bin_width_idx : (idx + 1) * bin_width_idx, :], axis = 0)

    binned_spikes = binned_spikes/bin_width # convert from [spikes/bin] to [spikes/s]    
    return binned_spikes

def bin_spike_times(spike_times, time_before, time_after, bin_width):
    '''
    Computes binned spikes (spike rate) [spikes/s]. The input data are 1D spike times in seconds.
    Binned spikes are calculated at each bin whose width is determined by bin_width. 

    Example:
        >>> spike_times = np.array([0.0208, 0.0341, 0.0347, 0.0391, 0.0407])
        >>> spike_times = spike_times.T
        >>> time_before = 0
        >>> time_after = 0.05
        >>> bin_width = 0.01
        >>> binned_unit_spikes, time_bins = precondition.bin_spike_times(spike_times, time_before, time_after, bin_width)
        >>> print(binned_unit_spikes)
            [  0.   0. 100. 300. 100.]
        >>> print(time_bins)
            [0.005 0.015 0.025 0.035 0.045]

    Args:
        spike_times (nspikes): 1D array of spike times [s]
        time_before (float): start time to easimate spike rate [s]
        time_after (float): end time to estimate spike rate (Estimation includes endpoint)[s]
        bin_width (float): width of time-bin to use for estimating spike rate [s]

    Returns:
        binned_unit_spikes (nbin, nch): spike rate [spikes/s].
        time_bins : the center of the time-bin over which firing rate is estimated. [s]
    '''

    time_bins = np.arange(time_before, time_after+bin_width, bin_width) # contain endpoint

    binned_unit_spikes, _ = np.histogram(spike_times, bins=time_bins)
    binned_unit_spikes = binned_unit_spikes/bin_width # convert [spikes] to [spikes/s]

    time_bins = time_bins[0:-1] + np.diff(time_bins)/2 #change time_bins to be the center of the bin, not the edges.

    return binned_unit_spikes, time_bins

def downsample(data, old_samplerate, new_samplerate):
    '''
    Downsample by averaging. Computes a downsample factor based on old_samplerate/new_samplerate.
    Pads data to be a multiple of the downsample factor, then averages blocks into the new
    samples. 

    Args:
        data (nt, ...): timeseries data to be downsampled. Can be 1D or 2D.
        old_samplerate (float): the current sampling rate of the data
        new_samplerate (float): the desired sampling rate of the downsampled data
    '''
    assert new_samplerate < old_samplerate
    assert data.ndim < 3 # doesn't work for more than 2 dimensions

    old_samples = data.shape[0]
    downsample_factor = int(old_samplerate/new_samplerate)

    # Pad the data to a multiple of the downsample factor
    pad_size = math.ceil(float(old_samples)/downsample_factor)*downsample_factor - old_samples
    pad_shape = (pad_size,)
    if data.ndim > 1:
        pad_shape = np.concatenate(([pad_size], data.shape[1:]))
    data_padded = np.append(data, np.zeros(pad_shape)*np.NaN, axis=0)

    # Downsample using average
    if data.ndim > 1:
        downsampled = np.zeros((int(data_padded.shape[0]/downsample_factor), data.shape[1]), dtype=data.dtype)
        for idx in range(data.shape[1]):
            downsampled[:,idx] = np.nanmean(data_padded[:,idx].reshape(-1, downsample_factor), axis=1)
        return downsampled
    else:
        return np.nanmean(data_padded.reshape(-1, downsample_factor), axis=1)

def filter_lfp(broadband_data, broadband_samplerate, lfp_samplerate=1000., low_cut=500., buttord=4):
    '''
    Low-pass filter and return downsampled version of broadband data at default 1000 Hz. Default
    filter parameters taken from Stavisky, Kao, et al., 2015

    Args:
        broadband_data (nt, ...): raw headstage data, e.g.
        broadband_samplerate (float): sampling rate of the raw data
        lfp_samplerate (float, optional): sampling rate of the LFP data. Defaults to 1000.
        low_cut (float, optional): cutoff frequency for low-pass filter. Defaults to 500 Hz
        buttord (int, optional): order for butterworth low-pass filter. Defaults to 4.

    Returns:
        (nt', ...): lfp data
    '''
    b, a = butter(buttord, low_cut, btype='lowpass', fs=broadband_samplerate)
    filtered_data = filtfilt(b, a, broadband_data, axis=0)
    return downsample(filtered_data, broadband_samplerate, lfp_samplerate)

def filter_spikes(broadband_data, samplerate, low_pass=500, high_pass=7500, buttord=3):
    '''
    Band-pass filter broadband data at default 500-7500 Hz. Default filtering parameters taken
    from Stavisky, Kao, et al., 2015

    Args:
        broadband_data (nt, ...): raw headstage data, e.g.
        samplerate (float): sampling rate of the raw data
        low_pass (float, optional): low-cut frequency, default 500 Hz
        high_pass (float, optional): high-cut frequency, default 7500 Hz
        buttord (int, optional): order for butterworth band-pass filter. Default 3

    Returns:
        (nt, ...): spike filtered data
    '''
    window = [low_pass, high_pass]
    b, a = butter(buttord, window, btype='bandpass', fs=samplerate)
    return filtfilt(b, a, broadband_data, axis=0)