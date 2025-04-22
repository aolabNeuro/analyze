# latency.py
# Functions for calculating latency and selectivity of neural responses.
# 
# AccLLR calculations are based on the paper:
# Banerjee A, Dean HL, Pesaran B. A likelihood method for computing selection times 
# in spiking and local field potential activity. J Neurophysiol. 2010 Dec;104(6):3705-20. 
# doi: 10.1152/jn.00036.2010. Epub 2010 Sep 8. https://pubmed.ncbi.nlm.nih.gov/20884767/

import warnings

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm
import multiprocessing as mp
from matplotlib import pyplot as plt

from .. import visualization
from .. import utils
from . import base

def detect_erp_response(nullcond, altcond, samplerate, num_sd=3, debug=False):
    '''
    Calculates the latency with which an ERP becomes maximized. Also checks for significant responses 
    using a threshold of num_sd standard deviations above the mean of a null condition (e.g. baseline) 
    response across time. Can be used on single trial or trial-averaged ERP data.
    
    Args:
        nullcond (nt, nch, ntr): responses for null condition trials
        altcond (nt, nch, ntr): responses for alternative condition trials
        samplerate (float): sampling rate of the data
        num_sd (float, optional): number of standard deviations to use as a threshold for significant response. Default 3.
        debug (bool, optional): if True, plot the data and computed thresholds. Default False.

    Returns:
        (nch, ntr): array of latencies in seconds for each channel. NaN indicates no response above threshold.
    
    Examples:

        Make a null baseline and an alternate condition response that are both uniform random noise. Then
        add a linearly increasing signal to one channel in the alternate condition response. The other channel
        will have no signal.

        .. code-block:: python
        
            fs = 100
            nt = fs * 2
            nch = 2
            ntr = 10
            null_data = np.random.uniform(-1, 1, (nt, nch, ntr))
            alt_data = np.random.uniform(-1, 1, (nt, nch, ntr)) 
            alt_data[:,1,:] += np.tile(np.expand_dims(np.arange(nt)/10, 1), (1,ntr))

            latency = aopy.analysis.latency.detect_erp_response(null_data, alt_data, fs, 3, debug=True)
            self.assertEqual(latency.shape, (nch, ntr))

        .. image:: _images/detect_erp_response.png
    
    ''' 
    mean = np.mean(nullcond, axis=0)
    std = np.std(nullcond, axis=0)
    significant = np.abs(altcond - mean) > num_sd * std
    idx_latency = np.argmax(significant, axis=0)
    idx_latency = np.where(np.any(significant, axis=0), idx_latency.astype(float), np.nan)
    
    if debug:
        y_min = np.min(np.concatenate([nullcond, altcond]))
        y_max = np.max(np.concatenate([nullcond, altcond]))

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.hlines([mean + (num_sd * std)], 0, 1, transform=plt.gca().get_yaxis_transform(), color='r', linestyles='dashed')
        visualization.plot_timeseries(nullcond, samplerate, color='k', alpha=0.5)
        plt.ylim(y_min, y_max)
        plt.title('nullcond')

        plt.subplot(2,2,2)
        plt.hlines([mean + (num_sd * std)], 0, 1, transform=plt.gca().get_yaxis_transform(), color='r', linestyles='dashed')
        visualization.plot_timeseries(altcond[:,~np.isnan(idx_latency)], samplerate, color='m', alpha=0.5)
        plt.xlabel('')
        plt.xticks([])
        plt.ylim(y_min, y_max)
        plt.title('altcond detected')
            
        plt.subplot(2,2,4)
        plt.hlines([mean + (num_sd * std)], 0, 1, transform=plt.gca().get_yaxis_transform(), color='r', linestyles='dashed')
        visualization.plot_timeseries(altcond[:,np.isnan(idx_latency)], samplerate, color='c', alpha=0.5)
        plt.ylim(y_min, y_max)
        plt.title('altcond undetected')

    return idx_latency/samplerate

def detect_itpc_response(im_nullcond, im_altcond, samplerate, num_sd=3, debug=False):
    '''
    Calculates the latency with which itpc becomes maximized for the given analytical signals. Also
    checks for significant responses using a threshold of num_sd standard deviations above the mean
    of a null condition (e.g. baseline) response.
    
    Args:
        im_nullcond (nt, nch, ntrial): analytical signals for null condition trials
        im_altcond (nt, nch, ntrial): analytical signals for alternative condition trials
        samplerate (float): sampling rate of the data
        num_sd (float, optional): number of standard deviations to use as a threshold for significant response. Default 3.
        debug (bool, optional): if True, plot the itpc values and computed thresholds. Default False.

    Returns:
        nch: array of latencies in seconds for each channel. NaN indicates no response above threshold.
    
    Examples:

        Create a null baseline and an alternate condition response that are both uniform random noise. Then
        add a sinusoidal signal to the alternate condition response. On one channel the signal is amplitude
        1 (smaller than noise) and on the other channel it is amplitude 10 (bigger than noise).
        
        
        .. code-block:: python
        
            fs = 100
            nt = fs * 2
            nch = 2
            ntr = 10
            null_data = np.random.uniform(-1, 1, (nt, nch, ntr))
            alt_data = np.random.uniform(-1, 1, (nt, nch, ntr))
            alt_data[:,0,:] += np.tile(np.expand_dims(np.sin(np.arange(nt)*np.pi/10), 1), (1,ntr))
            alt_data[:,1,:] += np.tile(np.expand_dims(10*np.sin(np.arange(nt)*np.pi/10), 1), (1,ntr))

            im_nullcond = signal.hilbert(null_data, axis=0)
            im_altcond = signal.hilbert(alt_data, axis=0)

            latency = aopy.analysis.latency.detect_itpc_response(im_nullcond, im_altcond, fs, 5, debug=True)

        .. image:: _images/detect_itpc_response.png
    ''' 
    itpc = base.calc_itpc(im_altcond)
    chance_itpc = base.calc_itpc(im_nullcond)
    mean = np.mean(chance_itpc, axis=0)
    std = np.std(chance_itpc, axis=0)
    significant = np.abs(itpc - mean) > num_sd * std
    idx_latency = np.argmax(significant, axis=0)
    idx_latency = np.where(np.any(significant, axis=0), idx_latency.astype(float), np.nan)
    
    if debug:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.hlines([mean + (num_sd * std)], 0, 1, transform=plt.gca().get_yaxis_transform(), color='r', linestyles='dashed')
        visualization.plot_timeseries(chance_itpc, samplerate, color='k', alpha=0.5)
        plt.ylim(0,1)
        plt.ylabel('ITPC')
        plt.title('nullcond')

        plt.subplot(2,2,2)
        plt.hlines([mean + (num_sd * std)], 0, 1, transform=plt.gca().get_yaxis_transform(), color='r', linestyles='dashed')
        visualization.plot_timeseries(itpc[:,~np.isnan(idx_latency)], samplerate, color='m', alpha=0.5)
        plt.xlabel('')
        plt.xticks([])
        plt.ylim(0,1)
        plt.ylabel('ITPC')
        plt.title('altcond detected')
            
        plt.subplot(2,2,4)
        plt.hlines([mean + (num_sd * std)], 0, 1, transform=plt.gca().get_yaxis_transform(), color='r', linestyles='dashed')
        visualization.plot_timeseries(itpc[:,np.isnan(idx_latency)], samplerate, color='c', alpha=0.5)
        plt.ylim(0,1)
        plt.ylabel('ITPC')
        plt.title('altcond undetected')

    return idx_latency/samplerate

def calc_llr_gauss(lfp, lfp_mean_1, lfp_mean_2, lfp_sigma_1, lfp_sigma_2):
    '''
    Calculate log likelihood ratio of lfp data belonging to one of two 
    gaussian models with no history.
    
    Args:
        lfp ((nt,) array): single trial lfp data
        lfp_mean_1 ((nt,) array): mean across trials of lfp data from condition 1
        lfp_mean_2 ((nt,) array): mean across trials of lfp data from condition 2
        lfp_sigma_1 (float): variance of lfp data from condition 1
        lfp_sigma_2 (float): variance of lfp data from condition 2
    
    Returns:
        (nt,) array: log likelihood ratio 
    
    '''
    res_1 = lfp - lfp_mean_1
    res_2 = lfp - lfp_mean_2

    llr = (np.log(lfp_sigma_2) 
          - np.log(lfp_sigma_1)
          + res_2**2/(2*lfp_sigma_2**2)
          - res_1**2./(2*lfp_sigma_1**2))
    
    return llr
   
def calc_accllr_lfp(lfp_altcond, lfp_nullcond, lfp_altcond_lowpass, lfp_nullcond_lowpass, common_variance=True):
    '''
    Accumulated log likelihood for single channel LFP data.
    
    Note: 
        As quoted from Banerjee et al. (2010), the common_variance parameter should
        be set to True:
        "Since AccLLR changes for LFP activity are scaled by the noise variance for 
        each condition, if the noise variances for each condition differ, the AccLLRs 
        scale differently and this leads to the need for different upper and lower 
        bounds at the discrimination stage. To allow the simplicity of using the same 
        bound for both conditions (see the following subsection), we assume that the 
        variance of activity under each condition is the same."
        
    Note:
        Unsure whether the lowpass filtered version of the lfp should be used in 
        place of the lfp entirely. That is how the Pesaran lab code works, however
        it doesn't seem to make sense to throw away the raw lfp, so here I use
        the lowpass filtered version for the mean estimate across trials but use 
        the raw lfp for single trials.
        
    Note:
        Finally, there is a difference between how the cumulative sum is calculated
        here versus in the Pesaran lab code. Here we include the first value of
        the LLR as the first value of the accLLR. However, the Pesaran lab truncates
        the accLLR so that it starts at the second sum.
    
    Args:
        lfp_altcond ((nt, ntrial) array): lfp data for alternative condition trials
        lfp_nullcond ((nt, ntrial) array): lfp data for null condition trials. The number of null condition
            trials must be the same as the number of alternative condition trials.
        lfp_altcond_lowpass ((nt, ntrial) array): low-pass filtered copy of alternative condition trials. If
            desired, just pass the lfp_altcond again to avoid using a low-pass filtered version for model building.
        lfp_nullcond_lowpass ((nt, ntrial) array): low-pass filtered copy of null condition trials
        common_variance (bool, optional): calculate a shared variance of event and null lfp (see notes). Default True.
    
    Returns:
        tuple: tuple containing:
            | **accllr_altcond ((nt, ntrial) array):** accumulated log-likelihood for alterative condition trials
            | **accllr_nullcond ((nt, ntrial) array):** accumulated log-likelihood for null condition trials
    '''    
    assert lfp_altcond.shape == lfp_nullcond.shape

    nt = lfp_altcond.shape[0]
    n_trial = lfp_altcond.shape[1]
    
    lfp_mean_altcond = np.mean(lfp_altcond_lowpass, axis=1)
    lfp_mean_nullcond = np.mean(lfp_nullcond_lowpass, axis=1)
        
    lfp_sigma_altcond = np.std(lfp_altcond.T - lfp_mean_altcond, ddof=1) # ddof=1 is the behavior of MATLAB
    lfp_sigma_nullcond = np.std(lfp_nullcond.T - lfp_mean_nullcond, ddof=1)
    lfp_sigma_both = (lfp_sigma_altcond + lfp_sigma_nullcond)/2
        
    accllr_altcond = np.zeros((nt, n_trial))
    accllr_nullcond = np.zeros((nt, n_trial))
    
    for idx_trial in range(n_trial):
        
        loo_trials = np.ones((n_trial,), dtype='bool')
        loo_trials[idx_trial] = 0
        
        lfp_mean_altcond_loo = np.mean(lfp_altcond_lowpass[:,loo_trials], axis=1)
        lfp_mean_nullcond_loo = np.mean(lfp_nullcond_lowpass[:,loo_trials], axis=1)
        
        if common_variance:
            llr_altcond = calc_llr_gauss(lfp_altcond[:,idx_trial], 
                                         lfp_mean_altcond_loo, lfp_mean_nullcond, 
                                         lfp_sigma_both, lfp_sigma_both)
            llr_nullcond = calc_llr_gauss(lfp_nullcond[:,idx_trial], lfp_mean_altcond, 
                                          lfp_mean_nullcond_loo, 
                                          lfp_sigma_both, lfp_sigma_both)
        else:
            llr_altcond = calc_llr_gauss(lfp_altcond[:,idx_trial], 
                                         lfp_mean_altcond_loo, lfp_mean_nullcond, 
                                         lfp_sigma_altcond, lfp_sigma_nullcond)
            llr_nullcond = calc_llr_gauss(lfp_nullcond[:,idx_trial], lfp_mean_altcond, 
                                          lfp_mean_nullcond_loo,
                                          lfp_sigma_altcond, lfp_sigma_nullcond)

        accllr_altcond[:, idx_trial] = np.nancumsum(llr_altcond)
        accllr_nullcond[:, idx_trial] = np.nancumsum(llr_nullcond)
            
    return accllr_altcond, accllr_nullcond

# Try using ROC approach from Qiao et al 2020
def detect_accllr(accllr, upper, lower):
    '''
    Calculate the probability of upper, lower, and unknown level detections. This
    version is present only for readability, since it is very slow. See 
    :func:`~aopy.analysis.accllr.detect_accllr_fast`.
    
    Args:
        accllr (nt, ntrial): the accllr timeseries to test across trials
        upper (float): upper level
        lower (float): lower level
        
    Returns:
        tuple: tuple containing:
            | **p (3,):** probability of upper, lower, and unknown level detections
            | **selection_idx (ntrial):** index at which accllr crosses upper threshold 
                (or nan if missed) for each trial
    '''
    ntrial = accllr.shape[1]
    selection_idx = np.zeros((ntrial,))
    n_upper = 0
    n_lower = 0
    n_unknown = 0
    
    for tr_idx in range(ntrial):
        lower_hit = accllr[:,tr_idx] < lower
        upper_hit = accllr[:,tr_idx] > upper
        if np.any(lower_hit) and np.any(upper_hit):
            
            # both thresholds hit, count whichever was first
            if np.where(lower_hit)[0][0] < np.where(upper_hit)[0][0]:
                n_lower += 1
                selection_idx[tr_idx] = np.where(lower_hit)[0][0]
            else:
                n_upper += 1
                selection_idx[tr_idx] = np.where(upper_hit)[0][0]
            
        elif np.any(lower_hit):
            
            # lower threshold hit
            n_lower += 1
            selection_idx[tr_idx] = np.nan
            
        elif np.any(upper_hit):
            
            # upper threshold hit
            n_upper += 1
            selection_idx[tr_idx] = np.where(upper_hit)[0][0]
            
        else:
            
            # no hit
            n_unknown += 1
            selection_idx[tr_idx] = np.nan

    
    p = np.array([
        n_upper/ntrial,
        n_lower/ntrial,
        n_unknown/ntrial
    ])
    
    return p, selection_idx

# Write a faster version of the detection algorithm
def detect_accllr_fast(accllr, upper, lower):
    '''
    Calculate the probability of upper, lower, and unknown level detections. This
    faster algorithm avoids looping over every single trial, greatly speeding up
    computations of accllr. 
    
    Args:
        accllr (nt, ntrial): the accllr timeseries to test across trials
        upper (float): upper level
        lower (float): lower level
        
    Returns:
        tuple: tuple containing:
            | **p (3,):** probability of upper, lower, and unknown level detections
            | **selection_idx (ntrial):** index at which accllr crosses upper threshold 
                (or nan if missed) for each trial
    '''
    nt = accllr.shape[0]
    n_trial = accllr.shape[1]
    
    # Find the first value above the threshold for all trials at once
    lower_hit = accllr < lower
    upper_hit = accllr > upper
    lower_hit_st = utils.first_nonzero(lower_hit, axis=0, all_zeros_val=nt+1) # if no hit, then set nt+1
    upper_hit_st = utils.first_nonzero(upper_hit, axis=0, all_zeros_val=nt+1)

    # Break lower-upper ties
    lower_first = lower_hit_st < upper_hit_st
    st = np.where(lower_first, np.nan, upper_hit_st)
    
    unknown = st == nt+1 # no threshold was hit
    st[st == nt+1] = np.nan

    # Sum the counts to calculate probabilites
    n_lower = np.count_nonzero(lower_first & ~unknown)
    n_unknown = np.count_nonzero(unknown)
    n_upper = n_trial - n_lower - n_unknown
    p = np.array([
        n_upper/n_trial,
        n_lower/n_trial,
        n_unknown/n_trial
    ])
    
    return p, st

def calc_accllr_performance(accllr_altcond, accllr_nullcond, nlevels=200):
    '''
    Calculate the probabilities and selection times of accllr trials for a number of
    different levels, evenly spaced between 0 and the max value of the accllr series. 
    
    Args:
        event_accllr (nt, ntrial): accllr timeseries for alternative condition trials
        null_accllr (nt, ntrial): accllr timeseries for null condition trials
        nlevels (int, optional): number of levels to calculate. Defaults to 200.
        
    Returns:
        tuple: tuple containing:
            | **p_altcond (nlevels,3):** probability of upper, lower, and unknown level detections
                for the alternative condition at each detection level
            | **p_nullcond (nlevels,3):** probability of upper, lower, and unknown level detections
                for the null condition at each detection level
            | **selection_idx (nlevels):** index at which accllr crosses upper threshold 
                (or nan if missed) for each trial, averaged across trials
            | **levels (nlevels):** levels used for calculation of probabilities
    '''
    max_accllr = np.nanmax([accllr_altcond, accllr_nullcond]) # ignore nan
    levels = np.linspace(max_accllr/nlevels,max_accllr,nlevels)

    p_altcond = np.ones((nlevels,3))
    p_nullcond = np.ones((nlevels,3))
    selection_idx = []
    for idx_level, level in enumerate(levels):
        p_altcond[idx_level,:], selection_idx_altcond = detect_accllr_fast(accllr_altcond, level, -level)
        p_nullcond[idx_level,:], _ = detect_accllr_fast(accllr_nullcond, level, -level)
        if np.all(selection_idx_altcond!=selection_idx_altcond):
            selection_idx.append(np.nan)
        else:
            selection_idx.append(np.nanmean(selection_idx_altcond))
        
    return p_altcond, p_nullcond, selection_idx, levels

def choose_best_level(p_altcond, p_nullcond, levels):
    '''
    Given a list of probabilities for upper, lower, and unknown detections at 
    various levels, select the level with the highest difference in correct 
    detection and incorrect detection.

    Args:
        p_altcond ((ntrial, 3) array): [upper, lower, unknown] detections for 
            trials from the alternative condition
        p_nullcond ((ntrial, 3) array): [upper, lower, unknown] detections for 
            trials from the null condition
        levels (nlevels): levels used for calculation of probabilities

    Returns:
        float: the best level to use
    '''
    p_correct_detect = (p_altcond[:,0]+p_nullcond[:,1])/2
    p_incorrect_detect = (p_nullcond[:,0]+p_altcond[:,1])/2
    p_diff = p_correct_detect - p_incorrect_detect
    p_max_idx = np.argmax(p_diff)
    p_max = p_diff[p_max_idx]
    level = levels[p_max_idx]
    return level

def compute_midrank(x):
    '''
    Computes midranks.
    
    Adapted from: https://github.com/yandexdataschool/roc_comparison

    Args:
       x (npt): data to compute midrank over
    
    Returns:
       (npt): array of midranks
    '''
    x_sorting = np.argsort(x)
    x_sorted = x[x_sorting]
    npt = len(x)
    mid_sorted = np.zeros(npt, dtype=float)
    i = 0
    while i < npt:
        j = i
        while j < npt and x_sorted[j] == x_sorted[i]:
            j += 1
        mid_sorted[i:j] = 0.5*(i + j - 1)
        i = j
    midrank = np.empty(npt, dtype=float)
    midrank[x_sorting] = mid_sorted + 1
    return midrank

def calc_delong_roc_variance(ground_truth, predictions):
    '''
    Computes ROC AUC variance for a single set of predictions using the
    fast version of DeLong's method for computing the covariance of
    unadjusted AUC.

    Adapted from https://github.com/Netflix/vmaf/

    Args:
        ground_truth (nt): array of 0 and 1 ground truth classes
        predictions (nt): array of floats of the probability of being class 1
       
    Returns:
        tuple: tuple containing:
            | **auc (float):** area under the curve after ROC analysis
            | **cov (float):** variance of the predicted auc
 
    Reference:
        @article{sun2014fast,
            title={Fast Implementation of DeLong's Algorithm for
                  Comparing the Areas Under Correlated Receiver Oerating
                  Characteristic Curves},
            author={Xu Sun and Weichao Xu},
            journal={IEEE Signal Processing Letters},
            volume={21},
            number={11},
            pages={1389--1393},
            year={2014},
            publisher={IEEE}
        }
    '''
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    predictions_sorted_transposed = predictions[np.newaxis, order]

    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def calc_accllr_roc(accllr_altcond, accllr_nullcond):
    '''
    ROC analysis on accllr data. Groups the data according to class (alt or null) and computes
    area under the curve (AUC) using the DeLong method.
    
    Args:
        accllr_altcond ((ntrial,) array): a single accumulated llr across trials for alternative
            condition trials. Usually we pick the end of the accumulation window as the timepoint 
            of interest.
        accllr_nullcond((ntrial,) array): a single accumulated llr across trials for null
            condition trials.

    Returns:
        tuple: tuple containing:
            | **auc (float):** area under the curve
            | **se (float):** error across trials of the auc
    '''
    group = np.concatenate([np.ones((len(accllr_altcond),)), np.zeros((len(accllr_nullcond),))])
    data = np.concatenate([accllr_altcond, accllr_nullcond])

    auc, var = calc_delong_roc_variance(group, data)
    return auc, np.sqrt(var)

def calc_accllr_st_single_ch(data_altcond_ch, data_nullcond_ch, lowpass_altcond_ch, lowpass_nullcond_ch,
                       modality, bin_width, nlevels, verbose_out=False):
    '''
    Calculate accllr selection time for a single channel

    Args:
        data_altcond_ch ((nt, ntrial)): lfp data for trials from the alternative condition
        data_nullcond_ch ((nt, ntrial)): lfp data for trials from the null condition
        lowpass_altcond_ch ((nt, ntrial) array): low-pass filtered copy of alternative condition trials. If
            desired, just pass the lfp_altcond again to avoid using a low-pass filtered version for model building.
        lowpass_nullcond_ch ((nt, ntrial) array): low-pass filtered copy of null condition trials
        modality (str): type of data being inputted ("lfp", "spikes", etc.)
        bin_width (float): bin width of input activity, or 1./samplerate for lfp data
        nlevels (int): number of levels at which to test accllr performance
        verbose_out (bool, optional): Include probabilities of detection and accllr in the output. Defaults to False.

    Returns:
        tuple: tuple containing:
            | **selection_time (nch, ntrials):** time (in s) at which each trial 
            | **roc_auc (nch,):** area under the curve from receiver operating characteristic analysis
            | **roc_se (nch,):** error across trials of the auc analysis
    '''
    # Calculate accllr for null and altcond
    if modality == 'lfp':
        accllr_altcond, accllr_nullcond = calc_accllr_lfp(data_altcond_ch, data_nullcond_ch, 
                                                          lowpass_altcond_ch, lowpass_nullcond_ch)
    else:
        raise ValueError("AccLLR currently only supports LFP")
        
    # Calculate appropriate levels to select accLLR decisions
    p_altcond, p_nullcond, _, levels = calc_accllr_performance(accllr_altcond, accllr_nullcond, nlevels)
    level = choose_best_level(p_altcond, p_nullcond, levels)

    # Calculate p values and selection times using the level with highest significance
    p_altcond, selection_idx_altcond = detect_accllr_fast(accllr_altcond, level,-level)
    selection_time_altcond = (selection_idx_altcond)*bin_width

    # Calculate ROC at the end of AccLLR accmulation time
    roc_auc, roc_se = calc_accllr_roc(accllr_altcond[-1,:], accllr_nullcond[-1,:])

    if verbose_out:
        p_nullcond, selection_idx_nullcond = detect_accllr_fast(-accllr_nullcond, level, -level)
        return (accllr_altcond, accllr_nullcond, p_altcond, p_nullcond, 
                selection_time_altcond, roc_auc, roc_se)

    return selection_time_altcond, roc_auc, roc_se

def _calc_accllr_st_worker(data_altcond, data_nullcond, lowpass_altcond, lowpass_nullcond,
                           modality, bin_width, nlevels, parallel):
    '''
    Worker function for calc_accllr_st(). 

    Args:
        data_altcond ((nt, nch, ntrial)): lfp channel data for trials from the alternative condition
        data_nullcond ((nt, nch, ntrial)): lfp channel data for trials from the null condition
        lowpass_altcond ((nt, nch, ntrial) array): low-pass filtered copy of alternative condition trials. If
            desired, just pass the lfp_altcond again to avoid using a low-pass filtered version for model building.
        lowpass_nullcond ((nt, nch, ntrial) array): low-pass filtered copy of null condition trials
        modality (str): type of data being inputted ("lfp", "spikes", etc.)
        bin_width (float): bin width of input activity, or 1./samplerate for lfp data
        nlevels (int): number of levels at which to test accllr performance. 
        parallel (bool or mp.pool.Pool): whether to use parallel processing. Can optionally be a pool object
            to use an existing pool. If True, a new pool is created with the number of CPUs available. If False,
            computation is done serially.

    Returns:
        tuple: tuple containing:
            | **selection_time (nch, ntrials):** time (in s) at which each trial 
            | **roc_auc (nch,):** area under the curve from receiver operating characteristic analysis
            | **roc_se (nch,):** error across trials of the auc analysis
    '''
    nt = data_altcond.shape[0]
    nch = data_altcond.shape[1]
    ntrials = data_altcond.shape[2]
    
    # Create a parallel pool if requested
    pool = None
    if parallel is True: # create a parallel pool
        pool = mp.Pool(min(mp.cpu_count(), nch))
    elif type(parallel) is mp.pool.Pool: # use an existing pool
        pool = parallel

    # Run accllr on the test datasets    
    if pool is not None:
        
        # call apply_async() without callback
        result_objects = [pool.apply_async(calc_accllr_st_single_ch, 
                          args=(data_altcond[:,ich,:], data_nullcond[:,ich,:], 
                                lowpass_altcond[:,ich,:], lowpass_nullcond[:,ich,:],
                                modality, bin_width, nlevels)) 
                          for ich in range(nch)]

        # result_objects is a list of pool.ApplyResult objects
        results = [r.get() for r in result_objects]
        selection_time_altcond, roc_auc, roc_se = zip(*results)
        selection_time_altcond = np.array(selection_time_altcond)
        roc_auc = np.squeeze(roc_auc)
        roc_se = np.squeeze(roc_se)
        
        if parallel is True:
            pool.close()

    else:
        selection_time_altcond = np.zeros((nch, ntrials))*np.nan
        roc_auc = np.zeros((nch,))*np.nan
        roc_se = np.zeros((nch,))*np.nan
        for ich in tqdm(range(nch), desc="AccLLR wrapper", leave=False):
            (selection_time_altcond[ich,:], roc_auc[ich], 
             roc_se[ich]) = calc_accllr_st_single_ch(data_altcond[:,ich,:], data_nullcond[:,ich,:], 
                                                     lowpass_altcond[:,ich,:], lowpass_nullcond[:,ich,:],
                                                     modality, bin_width, nlevels)
                
    return selection_time_altcond, roc_auc, roc_se
            
def calc_accllr_st(data_altcond, data_nullcond, lowpass_altcond, lowpass_nullcond, 
                   modality, bin_width, nlevels=None, match_selectivity=False, 
                   match_ch=None, noise_sd_step=1, parallel=True):
    '''
    Calculate accllr selection time for a multiple channels with optional selectivity
    matching. Selectivity is defined by the area under the ROC curve. To match 
    selectivity, we add gaussian noise to channel until selectivity is constant 
    across all channels.

    Based on the paper:

    Banerjee A, Dean HL, Pesaran B. A likelihood method for computing selection times 
    in spiking and local field potential activity. J Neurophysiol. 2010 Dec;104(6):3705-20. 
    doi: 10.1152/jn.00036.2010. Epub 2010 Sep 8. https://pubmed.ncbi.nlm.nih.gov/20884767/

    Note: 
        No noise is added to the models built on the lowpass versions of data. It is unclear how
        this step was performed in the original paper.

    Examples:

        Below are powers in 50-250hz band of three channels of laser-evoked response in motor cortex, 
        aligned to laser onset across 350 trials, 50 ms before and after. We first apply AccLLR without
        selectivity matching:

        .. code-block:: python

            st, roc_auc, roc_se, roc_p_fdrc = accllr.calc_accllr_st(altcond, nullcond, altcond, 
                                                                    nullcond, 'lfp', 1./samplerate)
            accllr_mean = np.nanmean(st, axis=1)

        .. image:: _images/accllr_test_data.png

        The dotted lines are the estimated selection times returned by accllr. Note that the bigger
        responses have faster selection times -- to test whether the estimates are biased by the 
        larger peaks being easier to identify, we can add noise until they match the selectivity of
        the smaller peak. Selectivity is defined by ROC analysis -- our ability to discriminate 
        between laser event trials and null trials.
    
        .. image:: _images/accllr_test_data_match_selectivity.png

        Note that the selection times for the two larger peaks have shifted slightly to the right,
        but are still earlier than the smaller peak, despite having the same (or lower) selectivity.
        Thus, we can be confident that the bigger peaks appear faster than the smaller one.

    Args:
        data_altcond ((nt, nch, ntrial)): lfp channel data for trials from the alternative condition
        data_nullcond ((nt, nch, ntrial)): lfp channel data for trials from the null condition
        lowpass_altcond ((nt, nch, ntrial) array): low-pass filtered copy of alternative condition trials. If
            desired, just pass the lfp_altcond again to avoid using a low-pass filtered version for model building.
        lowpass_nullcond ((nt, nch, ntrial) array): low-pass filtered copy of null condition trials
        modality (str): type of data being inputted ("lfp", "spikes", etc.)
        bin_width (float): bin width of input activity, or 1./samplerate for lfp data
        nlevels (int): number of levels at which to test accllr performance
        match_selectivity (bool, optional): if True, add noise to input data to match selectivity across channels.
            Default False.
        match_ch ((nch,) bool array or None, optional): if set, limit selectivity matching to only these 
            channels. Default None.
        noise_sd_step (float, optional): standard deviation step size to take when adding noise to ch data. 
            Default 1.
        parallel (bool or mp.pool.Pool): whether to use parallel processing. Can optionally be a pool object
            to use an existing pool. If True, a new pool is created with the number of CPUs available. If False,
            computation is done serially across channels.
            
    Returns:
        tuple: tuple containing:
            | **selection_time (nch, ntrials):** time (in s) at which each trial 
            | **roc_auc (nch,):** area under the curve from receiver operating characteristic analysis
            | **roc_se (nch,):** error across trials of the auc analysis
            | **roc_p_fdrc (nch,):** p-value for each channel after false-discovery-rate correction
    '''
    # Calculate an initial analysis
    nt = data_altcond.shape[0]
    nch = data_altcond.shape[1]
    ntrials = data_altcond.shape[2]
    if nlevels is None:
        nlevels = nt
    (selection_time_altcond, 
     roc_auc, roc_se) = _calc_accllr_st_worker(data_altcond, data_nullcond, 
                                               lowpass_altcond, lowpass_nullcond, modality, 
                                               bin_width, nlevels, parallel)
    
    # Optionally perform selectivity matching
    if match_ch is None:
        match_ch = np.ones((nch,), dtype='bool')
    if match_selectivity and not np.any(match_ch):
        warnings.warn("No channels selected. Not attempting selectivity matching.")
        match_selectivity = False
    if match_selectivity:

        # Find selective channel with the lowest ROC AUC
        match_ch_prob = np.min(roc_auc[match_ch])
        print(f"Matching selectivity to {match_ch_prob:0.4f}. Largest selectivity is {np.max(roc_auc[match_ch]):0.4f}")
        match_ch_idx = roc_auc <= match_ch_prob

        # Use the ROC AUC to match selectivity across channels   
        pbar = tqdm(total=nch-np.sum(match_ch_idx), desc="Matching selectivity")
        noise_sd = noise_sd_step
        while np.sum(match_ch_idx) < nch:
            
            # Add noise to non-matched channels
            unmatch_chs_idx = np.arange(nch) # unmatched channel idx
            unmatch_chs_idx = unmatch_chs_idx[~match_ch_idx]

            noise = np.random.normal(0,noise_sd, size=(nt,len(unmatch_chs_idx), ntrials))
            data_altcond[:,unmatch_chs_idx,:] += noise
            data_nullcond[:,unmatch_chs_idx,:] += noise

            # Re-calculate roc 
            (selection_time_altcond[unmatch_chs_idx,:], roc_auc[unmatch_chs_idx], 
            roc_se[unmatch_chs_idx]) = \
                _calc_accllr_st_worker(data_altcond[:,unmatch_chs_idx,:], data_nullcond[:,unmatch_chs_idx,:],
                                       lowpass_altcond[:,unmatch_chs_idx,:], lowpass_nullcond[:,unmatch_chs_idx,:],
                                       modality, bin_width, nlevels, parallel)

            match_ch_idx = roc_auc <= match_ch_prob
            noise_sd += noise_sd_step
            pbar.update(np.sum(match_ch_idx[unmatch_chs_idx]))
            
        pbar.close()

    # Finally calculate an FDR-corrected p-value across all channels
    z = (roc_auc-0.5)/roc_se # null hypothesis auc=0.5 (chance)
    p_uncorrected = norm.sf(z) # one-sided test of auc>0.5
    if nch > 1:
        rej, roc_p_fdrc = fdrcorrection(p_uncorrected, alpha=0.05)
    else:
        roc_p_fdrc = np.array((np.nan,))
    
    return selection_time_altcond, roc_auc, roc_se, roc_p_fdrc


def prepare_erp(erp, erp_lowpass, samplerate, time_before, time_after, 
                window_nullcond, window_altcond):
    '''
    Prepare data for accllr. Given event-related potentials, organize alternative
    and null condition data and subtract the mean baseline from the null condition.

    Args:
        erp ((nt, nch, ntr) array): trial-aligned data
        erp_lowpass ((nt, nch, ntr) array): trial-aligned data lowpass filtered
        samplerate (float): sampling rate of the erps
        time_before (float): time before event in the erp (in seconds)
        time_after (float): time after event in the erp (in seconds)
        window_nullcond ((2,) tuple of float): desired (start, end) of nullcond (in seconds)
        window_altcond ((2,) tuple of float): desired (start, end) of altcond (in seconds)

    Returns:
        tuple: tuple containing:
            | **data_altcond ((nt_before_new, nch, ntr) array):** alternative condition data
            | **data_nullcond ((nt_before_new, nch, ntr) array):** null condition data
            | **lowpass_altcond ((nt_before_new, nch, ntr) array):** alternative condition low-passed data
            | **lowpass_nullcond ((nt_before_new, nch, ntr) array):** null condition low-passed data

    Example:

        .. code-block:: python

            npts = 100
            nch = 50
            ntrials = 30
            align_idx = 50
            onset_idx = 40
            samplerate = 100
            time_before = align_idx/samplerate
            time_after = time_before
            window_nullcond = (-0.4, -0.1)
            window_altcond = (-0.1, 0.3)
            data = np.ones(npts)*10
            data[onset_idx:] = 10 + np.arange(npts-onset_idx)
            data = np.repeat(np.tile(data, (nch,1)).T[:,:,None], ntrials, axis=2)

            data_altcond, data_nullcond, lowpass_altcond, lowpass_nullcond = accllr.prepare_erp(
                data, data, samplerate, time_before, time_after, window_nullcond, window_altcond,
            )

        .. image:: _images/prepare_erp_for_accllr.png


    '''
    assert len(window_nullcond) == 2 and window_nullcond[1] > window_nullcond[0]
    assert len(window_altcond) == 2 and window_altcond[1] > window_altcond[0]
    assert window_nullcond[0] >= -time_before
    assert window_altcond[1] <= time_after
    
    # Find start and end indices
    altcond_start = int((time_before+window_altcond[0])*samplerate)-1
    altcond_dur = window_altcond[1] - window_altcond[0]
    altcond_end = altcond_start + int(altcond_dur*samplerate)
    nullcond_start = int((time_before+window_nullcond[0])*samplerate)
    nullcond_dur = window_nullcond[1] - window_nullcond[0]
    nullcond_end = nullcond_start+int(nullcond_dur*samplerate)
    
    # Extract data
    data_altcond = erp[altcond_start:altcond_end,:,:].copy()
    data_nullcond = erp[nullcond_start:nullcond_end,:,:].copy()
    lowpass_altcond = erp_lowpass[altcond_start:altcond_end,:,:].copy()
    lowpass_nullcond = erp_lowpass[nullcond_start:nullcond_end,:,:].copy()
    
    # Make each trial zero-mean for both stim and baseline
    baseline = np.mean(data_nullcond, axis=0)
    data_altcond -= baseline
    data_nullcond -= baseline
    lowpass_baseline = np.mean(lowpass_nullcond, axis=0)
    lowpass_altcond -= lowpass_baseline
    lowpass_nullcond -= lowpass_baseline

    return data_altcond, data_nullcond, lowpass_altcond, lowpass_nullcond
