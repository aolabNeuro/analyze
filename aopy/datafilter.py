# datafilter.py

# submodule containing data conditioning methods for ECoG

import numpy as np
import numpy.linalg as npla
import scipy.signal as sps
import sys

# python implementation of badChannelDetection.m - see which channels are too noisy
def bad_channel_detection( data, srate, lf_c=100, sg_win_t=8, sg_over_t=4, sg_bw = 0.5 ):
    print("Running bad channel assessment:")
    (num_ch,num_samp) = np.shape(data)

    # compute low-freq PSD estimate
    [fxx,txx,Sxx] = mt_sgram(data,srate,sg_win_t,sg_over_t,sg_bw)
    low_freq_mask = fxx < lf_c
    Sxx_low = Sxx[:,low_freq_mask,:]
    Sxx_low_psd = np.mean(Sxx_low,axis=2)

    psd_var = np.var(Sxx_low_psd,axis=1)
    norm_psd_var = psd_var/npla.norm(psd_var)
    low_var_θ = np.mean(norm_psd_var)/3
    bad_ch_mask = norm_psd_var <= low_var_θ

    return bad_ch_mask


# python implementation of highFreqTimeDetection.m - looks for spectral signatures of junk data
def high_freq_data_detection( data, srate, bad_channels=None, lf_c=100):
    print("Running high frequency noise detection: lfc @ {0}".format(lf_c))
    [num_ch,num_samp] = np.shape(data)
    bad_data_mask_all_ch = np.zeros((num_ch,num_samp))
    data_t = np.arange(num_samp)/srate
    if not bad_channels:
        bad_channels = np.zeros(num_ch)

    # mt sgram parameters
    sg_win_t = 8 # (s)
    sg_over_t = sg_win_t // 2 # (s)
    sg_bw = 0.5 # (Hz)

    # estimate hf influence, channel-wise
    for ch_i in np.arange(num_ch)[np.logical_not(bad_channels)]:
        print_progress_bar(ch_i,num_ch)
        fxx,txx,Sxx = mt_sgram(data[ch_i,:],srate,sg_win_t,sg_over_t,sg_bw) # Sxx: [num_ch]x[num_freq]x[num_t]
        num_freq, = np.shape(fxx)
        num_t, = np.shape(txx)
        Sxx_mean = np.mean(Sxx,axis=1) # average across all windows, i.e. numch x num_f periodogram

        # get low-freq, high-freq data
        low_f_mask = fxx < lf_c # Hz
        high_f_mask = np.logical_not(low_f_mask)
        low_f_mean = np.mean(Sxx_mean[low_f_mask],axis=0)
        low_f_std = np.std(Sxx_mean[low_f_mask],axis=0)
        high_f_mean = np.mean(Sxx_mean[high_f_mask],axis=0)
        high_f_std = np.std(Sxx_mean[high_f_mask],axis=0)

        # set thresholds for high, low freq. data
        low_θ = low_f_mean - 3*low_f_std
        high_θ = high_f_mean + 3*high_f_std

        for t_i, t_center in enumerate(txx):
            low_f_mean_ = np.mean(Sxx[low_f_mask,t_i])
            high_f_mean_ = np.mean(Sxx[high_f_mask,t_i])
            if low_f_mean_ < low_θ or high_f_mean_ > high_θ:
                # get indeces for the given sgram window and set them to "bad:True"
                t_bad_mask = np.logical_and(data_t > t_center - sg_win_t/2, data_t < t_center + sg_win_t/2)
                bad_data_mask_all_ch[ch_i,t_bad_mask] = True

#     bad_ch_θ = 0
#     bad_data_mask = np.sum(bad_data_mask_all_ch,axis=0) > bad_ch_θ
    bad_data_mask = np.any(bad_data_mask_all_ch,axis=0)

    return bad_data_mask, bad_data_mask_all_ch


# py version of noiseByHistogram.m - get upper and lower signal value bounds from a histogram
def histogram_defined_noise_levels( data, nbin=20 ):
    # remove data in outer bins of the histogram calculation
    hist, bin_edge = np.histogram(data,bins=nbin)
    low_edge, high_edge = bin_edge[1], bin_edge[-2]
    no_edge_mask = np.all([(data > low_edge), (data < high_edge)],axis = 0)
    data_no_edge = data[no_edge_mask]
    # compute gaussian 99% CI estimate from trimmed data
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_CI_lower, data_CI_higher = data_mean - 3*data_std, data_mean + 3*data_std
    # return min/max values from whole dataset or the edge values, whichever is lower
    noise_lower = low_edge if low_edge < data_CI_lower else min(data)
    noise_upper = high_edge if high_edge > data_CI_higher else max(data)

    return (noise_lower, noise_upper)


# multitaper spectrogram estimator (handles missing data, i.e. NaN values
def mt_sgram(x,srate,win_t,over_t,bw,interp=False,mask=None,detrend=False):
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


# py version of saturatedTimeDetection.m - get indeces of saturated data segments
def saturated_data_detection( data, srate, bad_channels=None, adapt_tol=1e-8 ,
                              win_n=20 ):
    print("Running saturated data segment detection:")
    num_ch, num_samp = np.shape(data)
    if not bad_channels:
        bad_channels = np.zeros(num_ch)
    bad_all_ch_mask = np.zeros((num_ch,num_samp))
    data_rect = np.abs(data)
    mask = [bool(not x) for x in bad_channels]
    for ch_i in np.arange(num_ch)[mask]:
        print_progress_bar(ch_i,num_ch)
        ch_data = data_rect[ch_i,:]
        θ1 = 50 # initialize threshold value
        θ0 = 0
        h, valc = np.histogram(ch_data,int(np.max(ch_data)))
        val = (valc[1:] + valc[:-1])/2 # computes the midpoints of each bin, valc are the edges
        val = np.floor(val)
        prob_val = h/np.shape(h)[0]

        # estimate midpoint between bimodal distribution for a theshold value
        while np.abs(θ1 - θ0) > adapt_tol:
            θ0 = θ1
            sub_θ_val_mask = val <= θ1
            sup_θ_val_mask = val > θ1
            sub_θ_val_mean = np.sum(np.multiply(val[sub_θ_val_mask],prob_val[sub_θ_val_mask]))/np.sum(prob_val[sub_θ_val_mask])
            sup_θ_val_mean = np.sum(np.multiply(val[np.logical_not(sup_θ_val_mask)],prob_val[np.logical_not(sup_θ_val_mask)]))/np.sum(prob_val[sup_θ_val_mask])
            θ1 = (sub_θ_val_mean + sup_θ_val_mean)/2

        # filter signal, boxcar window
        b_filt = np.ones(win_n)/win_n
        a_filt = 1
        ch_data_filt = sps.lfilter(b_filt,a_filt,ch_data)
        ch_data_filt_sup_θ_mask = ch_data_filt > θ1

        # get histogram-derived noise limits
        n_low, n_high = histogram_defined_noise_levels(ch_data)
        ch_data_low_mask = ch_data < n_low
        ch_data_high_mask = ch_data > n_high
        ch_data_filt_low_mask = np.logical_and(ch_data_filt_sup_θ_mask,ch_data_low_mask)
        ch_data_filt_high_mask = np.logical_and(ch_data_filt_sup_θ_mask,ch_data_high_mask)
        bad_all_ch_mask[ch_i,:] = np.logical_or(ch_data_filt_low_mask,ch_data_filt_high_mask)

        # clear out straggler values
        # I will hold off on implementing this until
#         out_of_range_samp_mask = np.logical_or(ch_data < n_low, ch_data > n_high)

#         for samp_i in np.arange(samp_i)[np.logical_and(out_of_range_samp_mask,np.logical_not(bad_all_ch_mask[i,:]))]:
#             if np.abs(ch_data[samp_i]) >= θ1 and
#             if samp_i < num_samp - srate*45:

#             else:

    num_bad = np.sum(bad_all_ch_mask,axis=0)
    sat_data_mask = num_bad > num_ch/2

    return sat_data_mask, bad_all_ch_mask

# 1-d interpolation of missing values (NaN) in multichannel data (unwraps, interpolates over NaN, fills in.)
def interp_multichannel(x):
    nan_idx = np.isnan(x)
    ok_idx = ~nan_idx
    xp = ok_idx.ravel().nonzero()[0]
    fp = x[ok_idx]
    idx = nan_idx.ravel().nonzero()[0]
    x[nan_idx] = np.interp(idx,xp,fp)

    return x

# simple progressbar, not tied to the iterator
def print_progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
