# precondition.py
# code for preconditioning neural data
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, windows
import math
import numpy as np

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
        cutoff_low (int): lower cut-off frequency (in Hz)
        cutoff_high (int): higher cutoff frequency (in Hz)
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

def bandpass_multitaper_filter_data(data, cutoff_low, cuttoff_high, fs, window_len = 50, std_half_bw=3):
    '''

    Args:
        data (numpy array) : Time series neural data - n_timepoints x n_channels
        cutoff_low (int): lower cutoff frequency (in Hz)
        cuttoff_high (int): higher cutoff frequency (in Hz)
        fs (int): sampling rate (in Hz)
        window_len (int): W - choice of W decides how much to smooth in the multitaper filter
        std_half_bw (float): standardised half bandwidth - bandwidth o frequency smoothing in Hz

    Returns:


    '''

    nt = data.shape[0]  # calculate the number of points
    # nch = data.shape[1]  # calculate the number of channels
    n = nt // fs // 10 # array n equals array nt divided by 10 times sampling - default window size
    dn = np.true_divide(n, 10)
    k = math.floor(2*window_len * std_half_bw -1)
    # tapers, taper_weight = windows.dpss(window_len, std_half_bw, Kmax=k,  sym= True)
#
#     n_tapers = tapers.shape[0]
#     n_timepoints = tapers.shape[1]

#     fk = [0, np.true_divide(fs, 2.)]
#
#     dn = math.floor(dn * fs)
#     pad = 2 # padding factor for fft
#
#     np2 =  2 ** (math.ceil(math.log(n_timepoints+1,2)))
#     np2 = math.log(n_timepoints+1,2)
#     nf = np.maximum(256, pad * 2 ** nextPowerOfTwo(np2))
#     # temp = np.multiply(sampling, nf)
#     # fk = np.true_divide(fk,sampling)
#     nfk = np.floor(np.array(fk) * nf / fs)
#
#     nwin = np.floor(np.true_divide((nt - n_timepoints), dn))  # calculate the number of windows
#     nfr = np.int(np.diff(nfk)[0])
#     f = np.linspace(fk[0], fk[1], nfr)
#
#     for win in range(0, int(nwin)):
#         # Here the optimized spectral loop starts.
#         if contflag = 1:
#             tmp = signal.detrend(X[:,dn*win:(dn*win+N)]).T
#
#             if tmp.shape[1] > N :# machine precision work-around? added by alo for weird behavior 181000020
#                 tmp = tmp[1:N, :]
#
#         else:
#             if nch > 1:
#                 mX = X[:,dn*win:(dn*win+N)].mean(axis=0)
#
# #                     extendedArray = extendArrayWithCurrentData(mX, 0, nch, False)
#                 tmp = (X[:,dn*win:(dn*win+N)] - mX).T # N x nch
#             else:
#                 tmp = X[:,dn*win:(dn*win+N)].T
#
#
#         # this can all be done in a single pass. Don't for-loop.
#         lowerBound = int(nfk[0])
#         upperBound = int(nfk[1])
#         inputArray = np.einsum('ij,ki->ijk',tmp,dpss_tapers) # N x nch x k
#         Xk = np.fft.fft(inputArray,axis=0,n=int(nf))
#         Xk = Xk[lowerBound:upperBound,]
#         XkSquare = (Xk * np.conj(Xk)).real
#         spec[:,win,:] = XkSquare.mean(axis=-1).T

    slepian_seq = windows.dpss(window_len,std_half_bw, Kmax=k)

    slepian_data = []
    n_tapers = slepian_seq.shape[0]
    n_timepoints = slepian_seq.shape[1]
    n_win = np.floor(data.shape[0] / window_len)
    # nws = np.true_divide(n,10)
    # nws = math.floor(nws*fs)
    nws = 100

    for i in range(n_tapers):
        slep_dat = np.zeros_like(data)
        for win in range(int(n_win)):
            tmp_data = data[nws*win:nws*win+window_len]
            slep_dat[nws*win:nws*win+window_len] = tmp_data * slepian_seq[i, :]
        slepian_data.append(slep_dat)

    return slepian_seq, slepian_data



