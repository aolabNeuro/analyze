# tuning.py
#
# Code related to tuning analysis, e.g. modulation depth, specificity, curve fitting, etc.

import numpy as np
from scipy.optimize import curve_fit


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


def get_mean_fr_per_direction(data, target_dir):
    '''

    Args:
        data (3D array): Neural data in the form of a 3D array neurons X trials x timebins
        target_dir (1D array): target direction

    Returns:
        tuple: Tuple containing:
            | **means_d:** = mean firing rate per neuron per target direction
            | **stds_d:** standard deviation from mean firing rate per neuron
    '''
    means_d = []
    stds_d = []

    for i in range(1, 9):
        means_d.append(np.mean(data[target_dir == i], axis=(0, 2)))
        stds_d.append(np.std(data[target_dir == i], axis=(0, 2)))

    return means_d, stds_d

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
