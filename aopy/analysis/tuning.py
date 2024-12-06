# tuning.py
#
# Code related to tuning analysis, e.g. modulation depth, specificity, curve fitting, etc.

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f_oneway


'''
Curve fitting
'''
# These functions are for curve fitting and getting modulation depth and preferred direction from firing rates
def curve_fitting_func(target_theta, b1, b2, b3):
    '''

    Args:
        target_theta (float): center out task target direction index [degrees]
        b1, b2, b3 (float): parameters used for curve fitting

    .. math::
    
        b1 * cos(\\theta) + b2 * sin(\\theta) + b3

    Returns:
        float: Evaluation of the fitting function for a given target.

    '''
    return b1 * np.cos(np.deg2rad(target_theta)) + b2 * np.sin(np.deg2rad(target_theta)) + b3


def get_modulation_depth(b1, b2):
    '''
    Calculates modulation depth from curve fitting parameters as follows:
    
    .. math::
    
        \\sqrt{b_1^2+b_2^2}

    Returns:
        float: Modulation depth (amplitude) of the curve fit
    '''
    return np.sqrt((b1 ** 2) + (b2 ** 2))


def get_preferred_direction(b1, b2):
    '''
    Calculates preferred direction from curve fitting parameters as follows:
    
    .. math:: 
        
        arctan(\\frac{b_1^2}{b_2^2})
    
    Returns:
        float: Preferred direction of the curve fit in radians
    '''
    b1sign = np.sign(b1)
    b2sign = np.sign(b2)
    temp_pd = np.arctan2(b2sign*b2**2, b1sign*b1**2)
    if temp_pd < 0:
      pd = (2*np.pi)+temp_pd
    else:
      pd = temp_pd
    return pd


def get_mean_fr_per_condition(data, condition_labels, return_significance=False):
    '''
    This function computes the average activity for each feature and trial. 

    Args:
        data (ntime, nch, ntrials): Trial aligned neural data
        condition_labels (ntrials): condition label for each trial
        return_significance (bool): Uses the one-way ANOVA test to compute a p-value for each channel/unit

    Returns:
        tuple: Tuple containing:
            | **means_d: (nch, nconditions)** = mean firing rate per neuron per target direction
            | **stds_d: (nch, nconditions)** standard deviation from mean firing rate per neuron
            | **pvalue: (nch)** significance of modulation
    '''
    means_d = []
    stds_d = []
    unique_condition_labels = np.unique(condition_labels)
    nconditions = len(unique_condition_labels)

    [means_d.append(np.mean(data[:,:,condition_labels==unique_condition_labels[icond]], axis=(0,2))) for icond in range(nconditions)]
    [stds_d.append(np.std(data[:,:,condition_labels==unique_condition_labels[icond]], axis=(0,2))) for icond in range(nconditions)]    

    if return_significance:
        cond_means = [] 
        [cond_means.append(np.mean(data[:,:,condition_labels==unique_condition_labels[icond]], axis=0)) for icond in range(nconditions)]
        _, pvalue = f_oneway(*cond_means, axis=1)
        return np.array(means_d).T, np.array(stds_d).T, pvalue
    else:
        return np.array(means_d).T, np.array(stds_d).T
    

def run_tuningcurve_fit(mean_fr, targets, fit_with_nans=False, min_data_pts=3):
    '''
    This function calculates the tuning parameters from center out task neural data.
    It fits a sinusoidal tuning curve to the mean firing rates for each unit.
    Uses approach outlined in Orsborn 2014/Georgopolous 1986.
    Note: To orient PDs with the quadrant of best fit, this function samples the target location
    with high resolution between 0-360 degrees.

    Args:
        mean_fr (nunits, ntargets): The average firing rate for each unit for each target.
        targets (ntargets): Targets locations to fit to [Degrees]. Corresponds to order of targets in 'mean_fr' (Xaxis in the fit). Targets should be monotonically increasing.
        fit_with_nans (bool): Optional. Control if the curve fitting should be performed for a unit in the presence of nans. If true curve fitting will be run on non-nan values but will return nan is less than 3 non-nan values. If false, any unit that contains a nan will have the corresponding outputs set to nan.
        min_data_pts (int): Optional. 

    Returns:
        tuple: Tuple containing:
            | **fit_params (3, nunits):** Curve fitting parameters for each unit
            | **modulation depth (ntargets, nunits):** Modulation depth of each unit
            | **preferred direction (ntargets, nunits):** preferred direction of each unit [rad]
    '''
    nunits = np.shape(mean_fr)[0]
    ntargets = len(targets)

    fit_params = np.empty((nunits,3))*np.nan
    md = np.empty((nunits))*np.nan
    pd = np.empty((nunits))*np.nan

    for iunit in range(nunits):
        # If there is a nan in the values of interest skip curve fitting, otherwise fit
        if ~np.isnan(mean_fr[iunit,:]).any():
            params, _ = curve_fit(curve_fitting_func, targets, mean_fr[iunit,:])
            fit_params[iunit,:] = params

            md[iunit] = get_modulation_depth(params[0], params[1])
            pd[iunit] = get_preferred_direction(params[0], params[1])

        # If this doesn't work, check if fit_with_nans is true. It it is remove nans and fit
        elif fit_with_nans:
            nonnanidx = ~np.isnan(mean_fr[iunit,:])
            if np.sum(nonnanidx) >= min_data_pts: # If there are enough data points run curve fitting, else return nan
                params, _ = curve_fit(curve_fitting_func, targets[nonnanidx], mean_fr[iunit,nonnanidx])
                fit_params[iunit,:] = params

                md[iunit] = get_modulation_depth(params[0], params[1])
                pd[iunit] = get_preferred_direction(params[0], params[1])
            else:
                md[iunit] = np.nan
                pd[iunit] = np.nan

    return fit_params, md, pd

def calc_dprime(*dist):
    '''
    d-prime or the sensitivity index is the peak-to-peak difference of means of the signals 
    across categories divided by their pooled (mean) standard deviation. The formula is as follows:
    
    .. math::

        d' = \\frac{µ_{max} - µ_{min}}{\sqrt{\sum_{i=0}^{n-1} (p_i)\sigma_i^2}}

    where $µ_{max} - µ_{min}$ is the peak-to-peak distance across category means, $p_i$ is the proportion 
    of trials in the i-th category, and $\sigma_i^2$ is the standard deviation of the i-th category.

    Args:
        *dist (ntr, nch): distribution of the data for each category. d-prime is calculated along the first axis.

    Returns:
        (nch): d-prime value for each channel or unit.
    
    Examples:
    
        $d'$ is essentially a signal-to-noise ratio. In the simple case of two distributions, the numerator 
        is the distance between the two means while the denominator is the average noise within each distribution. 
        If the distributions are normal and of equal variance then the $d'$ value becomes the z-score of the 
        difference between the two means.

        .. code-block:: python
        
            noise_dist = np.array([[0, 1], [0, 1], [0, 1]]).T
            signal_dist = np.array([[0.5, 1.5], [1, 2], [1.5, 2.5]]).T
            dprime = aopy.analysis.calc_dprime(noise_dist, signal_dist)
            print(dprime)
            >>> np.array([1, 2, 3])

        If there are more than two classes, d-prime finds the maximum signal-to-noise ratio, as illustrated in
        this pictorial from Williams, J. J. (2013). The numerator (peak-to-peak distance between distributions) 
        approximates the "signal" while the denominator (pooled standard deviation of each distribution)
        approximates the "noise".

        .. image:: _images/dprime.png

        Another simple example with 3 distributions:

        .. code-block:: python

            dist_1 = np.array([0, 1])
            dist_2 = np.array([1, 2])
            dist_3 = np.array([2, 3])
            dprime = aopy.analysis.calc_dprime(dist_1, dist_2, dist_3)
            print(dprime)
            >>> 4.
    '''
    means = [np.mean(d, axis=0) for d in dist]
    try:
        np.shape(means)
    except:
        raise ValueError('Input distributions must all have the same number of channels.')
    peak_to_peak_dist = np.max(means, axis=0) - np.min(means, axis=0)
    n_trials = np.sum([len(d) for d in dist])
    pooled_std = np.sum([len(d)*np.std(d, axis=0)/n_trials for d in dist], axis=0)
    return peak_to_peak_dist / pooled_std
