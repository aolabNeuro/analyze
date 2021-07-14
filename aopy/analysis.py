"""
Code for basic neural data analysis
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, FactorAnalysis
from scipy.optimize import curve_fit
from sklearn import model_selection
from scipy import interpolate
import warnings


def factor_analysis_dimensionality_score(data_in, dimensions, nfold, maxiter=1000, verbose=False):
    '''
    Estimate the latent dimensionality of an input dataset by appling cross validated 
    factor analysis (FA) to input data and returning the maximum likelihood values. 
    
    Args:
        data_in (nt, nch): Time series data in
        dimensions (ndim): 1D Array of dimensions to compute FA for 
        nfold (int): Number of cross validation folds to compute. Must be >= 1
        maxiter (int): Maximum number of FA iterations to compute if there is no convergence. Defaults to 1000.
        verbose (bool): Display % of dimensions completed. Defaults to False

    Returns:
        tuple: tuple containing:

        (ndim, nfold): Array of MLE FA score for each dimension for each fold
        
        (ndim, nfold): How many iterations of FA were required to converge for each fold
    '''

    # Initialize arrays
    log_likelihood_score = np.zeros((np.max(np.shape(dimensions)), nfold))
    iterations_required = np.zeros((np.max(np.shape(dimensions)), nfold))

    if verbose == True:
        print('Cross validating and fitting ...')

    # Compute the maximum likelihood score for each dimension using factor analysis    
    for dim_idx in range(len(dimensions)):
        fold_idx = 0

        # Handle the case without cross validation.
        if nfold == 1:
            fa = FactorAnalysis(n_components=dimensions[dim_idx], max_iter=maxiter)
            fafit = fa.fit(data_in.T)
            log_likelihood_score[dim_idx, fold_idx] = fafit.score(data_in.T)
            iterations_required[dim_idx, fold_idx] = fafit.n_iter_
            warnings.warn("Without cross validation the highest dimensional model will always fit best.")

        # Every other case with cross validation
        else:
            for trainidx, testidx in model_selection.KFold(n_splits=nfold).split(data_in.T):
                fa = FactorAnalysis(n_components=dimensions[dim_idx], max_iter=maxiter)
                fafit = fa.fit(data_in[:, trainidx].T)
                log_likelihood_score[dim_idx, fold_idx] = fafit.score(data_in[:, testidx].T)
                iterations_required[dim_idx, fold_idx] = fafit.n_iter_
                fold_idx += 1

        if verbose == True:
            print(str((100 * (dim_idx + 1)) // len(dimensions)) + "% Complete")

    return log_likelihood_score, iterations_required


'''
Curve fitting
'''


# These functions are for curve fitting and getting modulation depth and preferred direction from firing rates
def func(target, b1, b2, b3):
    '''

    Args:
        target (int) center out task target index ( takes values from 0 to 7)
        b1, b2, b3 : parameters used for curve fitting

    .. math::
    
        b1 * cos(\\theta) + b2 * sin(\\theta) + b3

    Returns: result from above equation

    '''
    theta = 45 * (target - 1)
    return b1 * np.cos(np.deg2rad(theta)) + b2 * np.sin(np.deg2rad(theta)) + b3


def get_modulation_depth(b1, b2):
    '''
    Calculates modulation depth from curve fitting parameters as follows:
    
    .. math::
    
        \\sqrt{b_1^2+b_2^2}

    '''
    return np.sqrt((b1 ** 2) + (b2 ** 2))


def get_preferred_direction(b1, b2):
    '''
    Calculates preferred direction from curve fitting parameters as follows:
    
    .. math:: 
        
        arctan(\\frac{b_1^2}{b_2^2})
        
    '''
    return np.arctan2(b2 ** 2, b1 ** 2)


def get_mean_fr_per_direction(data, target_dir):
    '''

    Args:
        data (3D array): Neural data in the form of a 3D array neurons X trials x timebins
        target_dir (1D array): target direction

    Returns:
        means_d = mean firing rate per neuron per target direction
        stds_d = standard deviation from mean firing rate per neuron
    '''
    means_d = []
    stds_d = []

    for i in range(1, 9):
        means_d.append(np.mean(data[target_dir == i], axis=(0, 2)))
        stds_d.append(np.std(data[target_dir == i], axis=(0, 2)))

    return means_d, stds_d


def plot_mean_fr_per_target_direction(means_d, neuron_id, ax, color, this_alpha, this_label):
    '''
    generate a plot of mean firing rate per target direction

    '''
    sns.set_context('talk')
    ax.plot(np.array(means_d)[:, neuron_id], c=color, alpha=this_alpha, label=this_label)

    ax.legend()
    ax.set_xlabel("Target", fontsize=16)
    ax.set_ylabel("Spike Rate (Hz)", fontsize=16)
    plt.tight_layout()


def run_curvefitting(means, make_plot=True, fig_title='Tuning Curve', n_subplot_rows=None, n_subplot_cols= None):
    '''
    Args:
        means (2D array) : Mean firing rate [n_targets x n_neurons]
        make_plot (bool) : Generates plot with curve fitting and mean firing rate for n_neuons
        Fig title (str) : Figure title
        n_rows (int) : No of rows in subplot
        n_cols (int) : No. of cols in subplot

    Returns:
        params_day (Numpy array) : Curve fitting parameters
        modulation depth (Numpy array) : Modulation depth of neuron
        preferred direction (Numpy array) : preferred direction of neurons
    '''
    params_day = []
    mod_depth = []
    pd = []

    if make_plot:
        # sns.set_context('paper')
        plt.figure(figsize=(20, 10))

    for this_neuron in range(np.shape(means)[1]):
        xdata = np.arange(1, 9)
        ydata = np.array(means)[:, this_neuron]
    # print(ydata)

        params, params_cov = curve_fit(func, xdata, ydata)
        # print(params)
        params_day.append(params)

        mod_depth.append(get_modulation_depth(params[0], params[1]))
        pd.append(get_preferred_direction(params[0], params[1]))

        if make_plot:
            plt.subplot(n_subplot_rows, n_subplot_cols, this_neuron + 1)
            plt.plot(xdata, ydata, 'b-', label='data')
            plt.plot(xdata, func(xdata, params[0], params[1], params[2]), 'b--', label='fit')
            plt.suptitle(fig_title, y=1.01)
            plt.ylim(0, 600)
            plt.xticks(np.arange(1, 8, 2))
    return np.array(params_day), np.array(mod_depth), np.array(pd)


def find_trough_peak_idx(unit_data):
    '''
    This function calculates the trough-to-peak time at the index level (0th order) by finding the minimum value of
    the waveform, and identifying that as the trough index. To calculate the peak index, this function finds the 
    index corresponding to the first negative derivative of the waveform. If there is no next negative derivative of
    the waveform, this function returns the last index as the peak time.
    
    Args:
        unit_data (nt, nch): Array of waveforms (Can be a 1D array with dimension nt)

    Returns:
        tuple: A tuple containing
        troughidx (nch): Array of indices corresponding to the trough time for each channel

        peakidx (nch): Array of indices corresponding ot the peak time for each channel. 
    '''
    # Handle condition where the input data is a 1D array
    if len(unit_data.shape) == 1:
        troughidx = np.argmin(unit_data)
        
        wfdecreaseidx = np.where(np.diff(unit_data[troughidx:])<0)
        
        if np.size(wfdecreaseidx) == 0:
            peakidx = len(unit_data)-1
        else:
            peakidx = np.min(wfdecreaseidx) + troughidx

    # Handle 2D input data array  
    else:
        troughidx = np.argmin(unit_data, axis = 0)
        peakidx = np.empty(troughidx.shape)

        for trialidx in range(len(peakidx)):
            
            wfdecreaseidx = np.where(np.diff(unit_data[troughidx[trialidx]:,trialidx])<0)

            # Handle the condition where there is no negative derivative.
            if np.size(wfdecreaseidx) == 0:
                peakidx[trialidx] = len(unit_data[:,trialidx])-1
            else:
                peakidx[trialidx] = np.min(wfdecreaseidx) + troughidx[trialidx]
        
    return troughidx, peakidx

def interpolate_extremum_poly2(extremum_idx, data, extrap_peaks=False):
    '''
    This function finds the extremum approximation around an index by fitting a second order polynomial (using a lagrange polynomial) to
    the index input, the point before, and the point after it. In the case where the input index is either 
    at the end or the beginning of the data array, the function can either fit the data using the closest 3
    data points and return the extrapolated peak value or just return the input index. This extrapolation
    functionality is controlled with the 'extrap_peaks' input variable. Note: the extrapolation function may choose an 
    index within the input data length if chosen points result in a polynomial with an extremum at that point.

    Args:
        extremum_idx (int): Current extremum index
        data (n): data used to interpolate (or extrapolate) with
        extrap_peaks (bool): If the extremum_idx is at the start or end of the data, indicate if the closest 3 points
                                should be used to extrapolate a peak index.
        
    Returns:
        tuple: A tuple containing 
        
        extremum_time (float): Interpolated (or extrapolated) peak time
        
        extremum_value (float): Approximated peak value.
        
        f (np.poly): Polynomial used to calculate peak time
    '''

    # Handle condition where the peak is at the beginning of a dataset
    if extremum_idx == 0:
        edge_idx = True
        xpts = np.arange((extremum_idx), extremum_idx+3, 1)
        ypts = data[extremum_idx:extremum_idx+3]
    
    # Handle condition where the peak is at the end of a dataset
    elif extremum_idx == len(data)-1:
        edge_idx = True
        xpts = np.arange((extremum_idx-2), extremum_idx+1, 1)
        ypts = data[extremum_idx-2:extremum_idx+1]
        
    # Condition where the peak is in the middle of the dataset
    else:
        edge_idx = False
        xpts = np.arange((extremum_idx-1), extremum_idx+2, 1)
        ypts = data[extremum_idx-1:extremum_idx+2]
    
    f = interpolate.lagrange(xpts, ypts)
    extremum_time = -f[1]/(2*f[2])
    extremum_value = (f[2]*(extremum_time**2)) + (f[1]*extremum_time) + f[0]
    
    # If end points should not be extrapolated from...
    if extrap_peaks==False and edge_idx:
        extremum_time = extremum_idx
        extremum_value = data[extremum_time]

    return extremum_time, extremum_value, f

def get_unit_spiking_mean_variance(spiking_data):
    '''
    This function calculates the mean spiking count and the spiking count variance in spiking data across 
    trials for each unit. 

    Args:
        spiking_data (ntime, nunits, ntr): Input spiking data

    Returns:
        Tuple:  A tuple containing
            unit_mean: The mean spike counts for each unit across the input time
            unit_variance: The spike count variance for each unit across the input time
    '''

    counts = np.sum(spiking_data, axis=1) # Counts has the shape (nunits, ntr)
    unit_mean = np.mean(counts, axis=1) # Averge the counts for each unit across all trials
    unit_variance = np.var(counts, axis=1) # Calculate the count variance for each unit across all trials

    return unit_mean, unit_variance

def get_pca_dimensions(data, max_dims=None, VAF=0.9):
    """
    Use PCA to estimate the dimensionality required to account for the variance in the given data
    
    Args:
        data (nt, nch): time series data
        max_dims (int): (default None) the maximum number of dimensions
                        if left unset, will equal the dimensions (number of columns) in the dataset
        VAF (float): (default 0.9) variance accounted for (VAF)

    Returns: 
        dimensions (list): list of principal components accounting for the variance
        explained_variance (list): variance accounted for by each principal component
        num_dims (int): number of principal components required to account for variance
    """
    if max_dims is None:
            max_dims = np.shape(data)[1]

    num_dims = None
    explained_variance = []
    dimensions = range(1, max_dims + 1)

    pca = PCA()
    pca.fit(data)

    for dims in dimensions:

        explained_here = sum(pca.explained_variance_ratio_[0:dims])
        explained_variance.append(explained_here)

        # If we have not yet reached the threshold, check if we have now reached the threshold
        if num_dims is None and explained_here > VAF:
            num_dims = dims

    return dimensions, explained_variance, num_dims

