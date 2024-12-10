# precondition.py
# Code for cleaning and preparing neural data for users to interact with,
# for example: down-sampling, outlier detection, and initial filtering

import warnings
import math

from scipy import signal
from scipy.signal import butter, filtfilt, windows
import numpy as np

from .. import analysis
from .. import utils
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
        filter_type (str) : Type of filter. Accepts one of the four values - {`lowpass`, `highpass`, `bandpass`, `bandstop`}

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

def butterworth_filter_data(data, fs, cutoff_freqs=None, bands=None, order=None, filter_type='bandpass'):
    '''
    Apply a digital butterworth filter forward and backward to a timeseries signal.

    Args:
        data (nt, ...): neural data
        fs (int): sampling rate (in Hz)
        cutoff_freqs (float): list of cut-off frequencies (in Hz); only for 'high pass' or 'low pass' filter. Use bands for 'bandpass' filter
        bands (list): frequency bands should be a list of tuples representing ranges e.g., bands = [(0, 10), (10, 20), (130, 140)] for 0-10, 10-20, and 130-140 Hz
        order (int): Order of the butterworth filter. If no order is specified, the function will find the minimum order of filter required to maintain +3dB gain in the bandpass range.
        filter_type (str) : Type of filter. Accepts one of the four values - {'lowpass', 'highpass', 'bandpass', 'bandstop'}

    Returns:
        tuple: Tuple containing:
            | **filtered_data (list):** output bandpass filtered data in the form of a list. Each list item is filtered data in corresponding frequency band
            | **Wn (list):** frequency bands
    '''

    if filter_type in ['lowpass', 'highpass']:
        Wn = cutoff_freqs

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
            if type(wp) != int and type(wp) != float and len(wp) == 2:
                ws = [wp[0] - 0.1 , wp[1] + 1]
            else:
                ws = wp - 0.1
            order, _ = signal.buttord(wp, ws, 3, 40, False)

        # Filter this frequency or band
        b, a = butter(order, w, btype=filter_type, fs= fs)
        # filtered_data = lfilter(b,a,data)
        filtered_data.append(filtfilt(b, a, data, axis=0))
    return filtered_data, Wn

'''
Filter functions related to multitaper method
'''
def dpsschk(n, p, k):
    '''
    Computes the Discrete Prolate Spheroidal Sequences (DPSS) array based on input tapers

    Args:
        n (float): window length in number of samples
        p (float): standardized half bandwidth in hz
        k (int): number of DPSS tapers to use

    Returns:
        e ((n, k) array): K DPSS windows
        v ((k,) array): The concentration ratios for the K windows
    '''
    n = round(n)

    if k < 1:
        raise Exception('Error:  K must be greater than or equal to 1')
    elif k > 2*p-1:
        raise Exception('Error:  K must be less than 2*P-1')

    e, v = windows.dpss(n, p, k,return_ratios=True)
    e = e.T
    
    return e, v

def dp_proj(tapers, fs=1, f0=0):
    '''
    Generates a prolate projection operator

    Args:
        tapers ((n, k) array): data tapers
        fs (float): sampling rate
        f0 (float): center frequency of filiter

    Returns:
        dp_proj (nt, K): projection operator in [time, K] form
    '''
    N = tapers.shape[0]
    K = tapers.shape[1]
    pr_op = np.zeros((N,K),dtype = 'complex')

    shifter = np.exp(-2.*np.pi*1j*f0*np.arange(1,N+1)/fs)
    for k in range(K):
        pr_op[:,k] = shifter.conj()*tapers[:,k]
        
    return pr_op

def compute_bp_filter(tapers, fs=1, f0=0):
    '''
    Generates a bandpass filter using the multitaper method

    Args:
        tapers ((n, k) array): data tapers
        fs (float): sampling rate
        f0 (float): center frequency of filiter

    Returns:
        X (1, 2*N*fs): a bandpass filter
    '''    
    pr_op = dp_proj(tapers, fs, f0)
    N = pr_op.shape[0]
    X = np.zeros( (1,2*N), dtype = 'complex')
    pr = pr_op@pr_op.conj().T

    for t in range(N):
        X[0, t:t+N] = X[0, t:t+N] + pr[:,N-1-t].conj().T
    X = X/N 
    
    return X

def convert_taper_parameters(n, w):
    '''        
    Converts tapers in [n, w] form to [n, p, k] form. It's wise to print out 
    the number of tapers (k) to check that there will be enough frequency 
    resoltuion in the filter.
    P is computed by N*W and K is given by math.floor(2*P-1)

    Args:
        n (float): time window for analysis (in seconds)
        w (float): half bandwith of tapers (in hz)

    Returns:
        tuple: [n, p, k] taper parameters
    '''
    p = n*w
    k = math.floor(2*p-1)
    if k < 1:
        warnings.warn("K is less than 1, try increasing the taper length or the half-bandwidth")
    return n, p, k

def mt_lowpass_filter(data, lowcut, taper_len, fs, verbose=True, **kwargs):
    '''
    Wrapper around mtfilter that auto-generates [n, p, k] and f0 parameters 
    based on the lowpass of interest, length of the tapers, and the sampling 
    rate. See :func:`~aopy.precondition.mtfilter` for more details.

    Args:
        data (nt, nch): time series array
        lowcut (float): low-pass cutoff frequency
        taper_len (float): length of the tapers (in seconds) 
        fs (float): sampling rate of the data
        verbose (bool, optional): if True, print out the n, k parameters.
        kwargs (dict): additional keyword-arguments to pass to mtfilter

    Returns:
        (nt, nch): filtered time-series data
    '''
    n, p, k = convert_taper_parameters(taper_len, lowcut)
    f0 = 0
    if verbose:
        print(f"Using {k} tapers of length {n:.4f} s")
    return mtfilter(data, n, p, k, fs, f0, **kwargs)


def mt_bandpass_filter(data, band, taper_len, fs, verbose=True, **kwargs):
    '''
    Wrapper around mtfilter that auto-generates [n, p, k] and f0 parameters based on the band of interest,
    length of the tapers, and the sampling rate. See :func:`~aopy.precondition.mtfilter` for more details.

    Args:
        data (nt, nch): time series array
        band ((2,) tuple): low and high frequencies to band-pass
        taper_len (float): length of the tapers (in seconds) 
        fs (float): sampling rate of the data
        verbose (bool, optional): if True, print out the n, k parameters.
        kwargs (dict): additional keyword-arguments to pass to mtfilter

    Returns:
        (nt, nch): filtered time-series data

    '''
    n, p, k = convert_taper_parameters(taper_len, (band[1]-band[0])/2)
    f0 = np.mean(band)
    if verbose:
        print(f"Using {k} tapers of length {n:.4f} s")
    return mtfilter(data, n, p, k, fs, f0, **kwargs)


def mtfilter(data, n, p, k, fs=1, f0=0, center_output=True, complex_output=False, use_fft=True):
    '''
    Bandpass-filter a time series data using the multitaper method. Tapers are calculated using [n, p, k]
    parameters (see :func:`~aopy.precondition.convert_taper_parameters`). Be careful to set these parameters
    to control temporal and spectral properties of the filter. The parameters you pick will depend on the 
    time-scale of questions you're interested in. You get better estimates by smoothing across more time, 
    but at the cost of temporal resolution. For example, if responses that happen quickly, you'll want to keep 
    the taper length relatively small whenever possible. But that then blurs your frequency resolution, 
    so you need to play around with the bandwidth and taper length to find the lowest temporal resolution 
    you can get away with for a given frequency range goal.

    Example:
        ::

            band = [575, 625] # signals within band can pass
            N = 0.1
            NW = (band[1]-band[0])/2
            f0 = np.mean(band)
            n, p, k = convert_taper_parameters(N, NW)
            print(f"Using {k} tapers of half bandwidth {p:.2f}")

            T = 0.05
            fs = 25000
            x, t = utils.generate_test_signal(T, fs, freq=[600, 312, 2000], a=[5, 2, 0.5], noise=0.2)

            x_mtfilter = precondition.mtfilter(x, tapers, fs=fs, f0=f0)

        .. image:: _images/mtfilter.png

    Args:
        data (nt, nch): time series array
        n (float): window length in seconds
        p (float): standardized half bandwidth in hz
        k (int): number of DPSS tapers to use
        fs (float): sampling rate
        f0 (float, optional): center frequency of filiter. Default 0.
        center_output (bool, optional): if True, output data is centered. Default True.
        complexflag (bool, optional): if False, output data becomes real. Default False.
        use_fft (bool, optional): if True, use FFT to convolve data with filter, should be faster. Default True.

    Returns:
        (nt, nch): filtered time-series data
    '''   
    tapers, _ = dpsschk(n*fs, p, k)
        
    if data.ndim == 1:
        data = np.reshape(data,(-1,1))
        
    filt = compute_bp_filter(tapers, fs, f0)
    N = filt.shape[1]
    sz = data.shape

    if complex_output:
        Y = np.zeros(sz) + 1j*np.zeros(sz)
    else:
        filt = filt.real
        Y = np.zeros(sz)

    if use_fft:
        tmp = signal.fftconvolve(data, filt.T, axes=0)
        if center_output:
            Y = tmp[round(N/2):sz[0]+round(N/2), :]
        else:
            Y = tmp[N-1:sz[0]+N,:]
    else:
        for ii in range(sz[1]):
            tmp = np.convolve(data[:,ii], filt[0,:].T)
            if center_output:
                Y[:,ii] = tmp[round(N/2):sz[0]+round(N/2)]
            else:
                Y[:,ii] = tmp[N-1:sz[0]+N]

    if f0 > 0:
        Y = 2*Y
    
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
    if spike_filt_data.ndim == 1:
        spike_filt_data = spike_filt_data.reshape(-1,1)
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

def detect_spikes_chunk(spike_filt_data, samplerate, threshold, chunksize, above_thresh=True,  wf_length=1000, tbefore_wf=200):
    '''
    This function is based on detect_spikes and calculates spike times while dividing data into chunks to deal with memory issues.
    This may work for numpy memmap array

    Args:
        spike_filt_data (nt,nch): Time series neural data to detect spikes and extract waveforms from.
        samplerate (float): Sampling rate [Hz]
        threshold (float): Threshold that input data must cross to indicate a spike for each channel.
        chunksize (int): Chunk size to process data by chunk
        above_thresh (bool): If True, only spikes above the threshold will be detected. If false, only spikes below threshold will be detected. 
        wf_length (float): Length of waveforms to output :math:`[\mu s]`. Actual length will be rounded up. If set to 'None', waveforms will not be returned. Defaults to 1000 :math:`\mu s`
        tbefore_wf (float): Length of waveform to include before spike detection time :math:`[\mu s]`. Defaults to 200 :math:`\mu s`  
        
    Returns: 
        tuple: Tuple containing:
            | **spike_times (list of spike times):**  List of nspike length arrays with each list element corresponding to a channel. Spike times are in seconds.
            | **spike_waveforms (list of waveforms):** List of (nspike, nwf_pts) arrays with each list element corresponding to a channel. Returns NaN if there aren't enough data points to retreive a full waveform.
    '''
    if spike_filt_data.ndim == 1:
        spike_filt_data = spike_filt_data.reshape(-1,1)
        
    nt,nch = spike_filt_data.shape
    nchunk = math.ceil(nt/chunksize)
    chunktime = chunksize/samplerate
    
    # Calculate spike times
    n_samples = 0
    previous_thresh = np.ones(nch) == 1
    spike_times = []
    spike_waveforms =[]
    
    for ichunk in range(nchunk):
        if ichunk == nchunk:
            data = spike_filt_data[n_samples:,:]
        else:
            data = spike_filt_data[n_samples:n_samples+chunksize,:]

        if above_thresh:
            data_above_thresh_mask = data > threshold
        else:        
            data_above_thresh_mask = data < threshold

        # Spike times at each chunk (temp_spike_times: a list of array with different shape)

        for ich in range(nch):
            spike_times_chunk, _ = utils.detect_edges(data_above_thresh_mask[:,ich], samplerate, rising=True, falling=False)
            spike_times_chunk = spike_times_chunk + ichunk*chunktime # Take into account chunk time
            
            if ichunk == 0:
                spike_times.append(spike_times_chunk)
            else:
                if not previous_thresh[ich]: # Check if the last data of the previous chunk is below threshold
                    if data_above_thresh_mask[0,ich]:
                        spike_times_chunk = np.append(ichunk*chunktime, spike_times_chunk)   
                        
                spike_times[ich] = np.append(spike_times[ich],spike_times_chunk)
            
            # Check if data is below threshold at the last point of the chunk
            if data_above_thresh_mask[-1,ich]:
                previous_thresh[ich] = True
            else:
                previous_thresh[ich] = False # This is for next chunk
                
            # Spike waveforms at each chunk and channel
            if wf_length is not None:
                wf_idx_length = math.ceil(samplerate*(wf_length*(1e-6)))
                nspikes = len(spike_times_chunk)

                spike_waveforms_chunk = np.zeros((nspikes, wf_idx_length))
                spike_waveforms_chunk[:] = np.nan

                for ispike in range(nspikes):
                    starttime = spike_times_chunk[ispike]-(tbefore_wf*(1e-6))
                    startidx = int(np.round(starttime*samplerate, decimals=0))
                    stopidx = startidx + wf_idx_length

                    if np.logical_and(stopidx < nt, startidx>=0): # Ensure there are enough data points to grab the waveform
                        spike_waveforms_chunk[ispike, :] = spike_filt_data[startidx:stopidx,ich]
                
                if ichunk == 0:
                    spike_waveforms.append(spike_waveforms_chunk)
                else:
                    spike_waveforms[ich] = np.concatenate([spike_waveforms[ich],spike_waveforms_chunk],axis=0)
            
        n_samples += chunksize

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

def calc_ks_waveforms(raw_data, sample_rate, spike_times_unit, templates, channel_pos, waveforms_nch=10, time_before=1000., time_after=1000.):
    '''
    Calculate waveforms, waveform channels, and positions of units, using templates from kilosort
    This function does not account for drift correction
    
    Args:
        raw_data (nt,nch): time series neural data to detect spikes and extract waveforms from.
        sample_rate (float): sampling rate (Hz)
        spike_times_unit (dict): spike times for each unit identified by kilosort
        templates (n_unit, n_points, nch): templates that kilosort used to detect spikes
        channel_pos (nch, 2): channel positions
        waveforms_nch (int, optional): the number of channels with large amplitude of templates
        time_before (float, optional): time (us) to include before the start of each trial
        time_after (float, optional): time (us) to include after the start of each trial
    
    Returns:
        tuple: tuple containing:
            | **unit_waveforms (dict):** waveforms for each unit. The shape is (nspikes,  m_points, waveforms_nch)
            | **unit_waveforms_ch (n_unit, waveforms_nch):** large amplitude channels in templates
            | **unit_pos (dict):** channel positions of each unit
    '''
    
    time_before *= 1e-6
    time_after *= 1e-6
    nch = channel_pos.shape[0]
    duration = int((time_before + time_after)*sample_rate)
    
    unit_waveforms_ch = {}
    unit_waveforms = {}
    unit_pos = {}

    for iunit, unit in enumerate(spike_times_unit.keys()):
        # Look at high amplitude channels in templates
        amp_template_ch = np.zeros(nch)
        for ich in range(nch):
            amp_template_ch[ich] = np.max(templates[int(unit),:,ich])-np.min(templates[int(unit),:,ich]) # don't use iunit instead of int(unit)

        # Sort high amplitude channels and save channels and their positions
        large_amp_ch = np.argsort(amp_template_ch)[::-1][:waveforms_nch]
        unit_waveforms_ch[f'{unit}'] = large_amp_ch
        unit_pos[f'{unit}'] = channel_pos[large_amp_ch[0],:]

        # Get waveforms in high amplitude channels for each spike
        unit_times = spike_times_unit[f'{unit}']
        waveforms = np.zeros((unit_times.shape[0],duration,waveforms_nch))*np.nan
        for ispike, unit_time in enumerate(unit_times):
            start = int((unit_time - time_before)*sample_rate)
            end = start + duration

            if np.logical_and(end < raw_data.shape[0], start >= 0): # Ensure there are enough data points to grab the waveform
                for ich, ch in enumerate(large_amp_ch):
                    waveforms[ispike,:,ich] = raw_data[start:end,ch]
                    
        unit_waveforms[f'{unit}'] = waveforms
        
    return unit_waveforms, unit_waveforms_ch, unit_pos

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
    If the downsample factor is fractional, then first upsamples to the least common multiple of 
    the two sampling rates. Finally, pads data to be a multiple of the downsample factor and 
    averages blocks into the new samples. 

    .. image:: _images/downsample.png

    Args:
        data (nt, ...): timeseries data to be downsampled. Can be 1D or 2D.
        old_samplerate (int): the current sampling rate of the data
        new_samplerate (int): the desired sampling rate of the downsampled data
        
    Returns:
        (nt, ...) downsampled data
    '''
    assert new_samplerate < old_samplerate, "New sampling rate must be less than old sampling rate"
    assert int(old_samplerate) == old_samplerate, "Input samplerates must be integers"
    assert int(new_samplerate) == new_samplerate, "Input samplerates must be integers"

    # Check if the downsample factor will be an integer, otherwise we find a common divisor
    if old_samplerate % new_samplerate != 0:
        lcm = np.lcm(int(old_samplerate), int(new_samplerate)) # least common multiple
        print(f"Upsampling first to {lcm} Hz")
        upsampled = np.repeat(data, lcm/old_samplerate, axis=0)
        return downsample(upsampled, lcm, new_samplerate)
        
    old_samples = data.shape[0]
    downsample_factor = int(old_samplerate/new_samplerate)

    # Pad the data to a multiple of the downsample factor
    pad_size = math.ceil(float(old_samples)/downsample_factor)*downsample_factor - old_samples
    pad_shape = (pad_size,)
    if data.ndim > 1:
        pad_shape = np.concatenate(([pad_size], data.shape[1:]))
    data_padded = np.append(data, np.zeros(pad_shape)*np.NaN, axis=0)

    # Downsample using average
    if data.ndim == 1:
        return np.nanmean(data_padded.reshape(-1, downsample_factor), axis=1)
    elif data.ndim == 2:
        downsampled = np.zeros((int(data_padded.shape[0] / downsample_factor), *data.shape[1:]), dtype=data.dtype)
        for idx in range(data.shape[1]):
            downsampled[:, idx] = np.nanmean(data_padded[:, idx].reshape(-1, downsample_factor), axis=1)
        return downsampled
    elif data.ndim == 3:
        downsampled = np.zeros((int(data_padded.shape[0] / downsample_factor), *data.shape[1:]), dtype=data.dtype)
        for idx1 in range(data.shape[1]):
            for idx2 in range(data.shape[2]):
                downsampled[:, idx1, idx2] = np.nanmean(data_padded[:, idx1, idx2].reshape(-1, downsample_factor), axis=1)
        return downsampled

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
        tuple: tuple containing:
            | **lfp_data (nt', ...):** downsampled filtered lfp data
            | **samplerate (float):** sampling rate of the lfp data
    '''
    b, a = butter(buttord, low_cut, btype='lowpass', fs=broadband_samplerate)
    filtered_data = filtfilt(b, a, broadband_data, axis=0)
    return downsample(filtered_data, broadband_samplerate, lfp_samplerate), lfp_samplerate

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

def filter_kinematics(kinematic_data, samplerate, low_cut=15, buttord=4):
    '''
    Low-pass filter kinematics data to below 15 Hz by default. 
    Filter parameters taken from Bradberry, et al., 2009 
    https://doi.org/10.1016/j.neuroimage.2009.06.023 

    Args:
        kinematic_data (nt, ...): raw hand data, e.g.
        samplerate (float): sampling rate of the raw data
        low_cut (float, optional): cutoff frequency for low-pass filter. Defaults to 15 Hz
        buttord (int, optional): order for butterworth low-pass filter. Defaults to 4.

    Returns:
        tuple: tuple containing:
            | **filtere_data (nt, ...):** filtered kinematics data
            | **samplerate (float):** sampling rate of the kinematics data

    Examples:

        .. code-block:: python

            fs = 100
            x_single, t = utils.generate_test_signal(T=5, fs, 1, 5)
            x_noise, t = utils.generate_test_signal(T=5, fs=fs, freq=[1,3,30], a=[5, 2, 0.5], noise=0.2)
            x_filt, _ = precondition.filter_kinematics(x, fs, low_cut=15, buttord=4)
            fig, ax = plt.subplot_mosaic([['A', 'B'],['C', 'C']])
        
            ax['A'].plot(t, x_noise, label='Noisy signal')
            ax['A'].plot(t, x_filt, label='Filtered signal')
            ax['A'].set_xlabel('time (seconds)')
            
            x_filt_simple = filt_fun(x_single)
            ax['B'].plot(t, x_single, label=f'{freq[0]} Hz signal')
            ax['B'].plot(t, x_filt_simple, label='Filtered signal')
            ax['B'].set_xlabel('time (seconds)')

            f_noise, psd_noise = analysis.get_psd_welch(x_noise, fs)
            f_filt, psd_filt = analysis.get_psd_welch(x_filt, fs)
            ax['C'].semilogy(f_noise, psd_noise, label='Noisy signal')
            ax['C'].semilogy(f_filt, psd_filt, label='Filtered signal')
            ax['C'].set_xlabel('frequency (Hz)')
            ax['C'].set_ylabel('PSD')

        .. image:: _images/filter_kinematics.png
    '''
    b, a = butter(buttord, low_cut, btype='lowpass', fs=samplerate)
    filtered_data = filtfilt(b, a, kinematic_data, axis=0)
    return filtered_data, samplerate
