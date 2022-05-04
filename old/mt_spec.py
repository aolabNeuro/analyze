import numpy as np
import scipy.signal as sps

# multitaper spectrogram estimator (handles missing data, i.e. NaN values
def mt_sgram(x,srate,win_t,over_t,bw,interp=False,mask=None,detrend=False):
    """mt_sgram

    compute multitaper spectrogram from a time series data array

    Args:
        x (channel x samples): data array
        srate (int): sample rate
        win_t (float): spectrogram window length
        over_t (float): spectrogram window overlap length
        bw (float): time half-bandwidth product (determines number of tapers)
        interp (bool, optional): interpolate np.nan values in x. Defaults to False.
        mask (bool array, optional): boolean array masking time points in x. Defaults to None.
        detrend (bool, optional): detrend data. Defaults to False.

    Returns:
        fxx (np.array): frequency values
        txx (np.array): time values
        Sxx (np.array): spectrogram estimate
    """

    # x - input data
    # srate - sampling rate of x
    # win_t - length of window (s)
    # over_t - size of window overlap (s)
    # bw - frequency resolution, i.e. bandwidth

    n_t = np.shape(x)[-1]
    t = srate*np.arange(n_t)

    # find, interpolate nan-values (replace in the output with nan)
#     nan_idx = np.any(np.isnan(x),axis=0)
    if interp:
        x = interp_multichannel(x)

    # compute parameters
    nw = bw*win_t/2 # time-half bandwidth product
    n_taper = int(max((np.floor(nw*2-1),1)))
    win_n = int(srate*win_t)
    over_n = int(srate*over_t)
    dpss_w = sps.windows.dpss(win_n,nw,Kmax=n_taper)

    # estimate mt spectrogram
    Sxx_m = []
    for k in range(n_taper):
        fxx,txx,Sxx_ = sps.spectrogram(x,srate,window=dpss_w[k,:],noverlap=over_n,detrend=detrend)
        Sxx_m.append(Sxx_)
    Sxx = np.mean(Sxx_m,axis=0)

    # align sgram time bins with bad times, overwrite values with NaN
    if np.any(mask):
        n_bin = np.shape(txx)[0]
        txx_edge = np.append(txx - win_t/2,txx[-1]+win_t/2)
        bad_txx = np.zeros(n_bin)
        for k in range(n_bin):
            t_in_bin = np.logical_and(t>txx_edge[k],t<txx_edge[k+1])
            bad_txx[k] = np.any(np.logical_and(t_in_bin,mask))
        bad_txx = bad_txx > 0
        Sxx[...,bad_txx] = np.nan

    return fxx, txx, Sxx


def interp_multichannel(x):
    """interp_multichannel

    interpolates nan segments in multichannel data

    Args:
        x (np.array): multichannel data array containing nan values

    Returns:
        x_interp (np.array): data array with interpolated nan values
    """
    nan_idx = np.isnan(x)
    ok_idx = ~nan_idx
    xp = ok_idx.ravel().nonzero()[0]
    fp = x[ok_idx]
    idx = nan_idx.ravel().nonzero()[0]
    x[nan_idx] = np.interp(idx,xp,fp)

    return x
