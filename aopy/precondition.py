# precondition.py
# code for preconditioning neural data

from scipy.signal import butter, lfilter

def bandpass_butterworth_params(cutoff_low, cutoff_high, fs, order = 4, filter_type = 'bandpass'):
    '''
    Args:
        cutoff_low (int): lower cut-off frequency (in Hz)
        cutoff_high (int): higher cutoff frequency (in Hz)
        fs (int): sampling rate (in Hz)
        order (int): Order of the butter worth filter
        filter_type (str) : Type of filter. Accepts one of the four values - {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns: bandpass filter parameters

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
        cutoff_low: lower cut-off frequency (in Hz)
        cutoff_high: higher cutoff frequency (in Hz)
        fs: sampling rate (in Hz)
        order: Order of the butterworth filter
        filter_type (str) : Type of filter. Accepts one of the four values - {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns:
        filtered_data: output bandpass filtered data

    '''

    b,a = butter(order, [cutoff_low, cutoff_high], btype=filter_type, fs= fs)
    filtered_data = lfilter(b,a,data)

    return filtered_data