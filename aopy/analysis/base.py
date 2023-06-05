# Code for neural data analysis; functions here should return interpretable results such as
# firing rates, success rates, direction tuning, etc.

import numpy as np
from matplotlib import pyplot as plt
from ..precondition.base import dpsschk

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression

import scipy
from scipy import stats, signal
from scipy import interpolate
from scipy.fft import fft

import warnings
import nitime.algorithms as tsa
import pywt
import math

from .. import utils
from .. import preproc

'''
Correlation / dimensionality analysis
'''
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
        tuple: Tuple containing:
            | **log_likelihood_score (ndim, nfold):** Array of MLE FA score for each dimension for each fold
            | **iterations_required (ndim, nfold):** How many iterations of FA were required to converge for each fold
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

def get_pca_dimensions(data, max_dims=None, VAF=0.9, project_data=False):
    """
    Use PCA to estimate the dimensionality required to account for the variance in the given data. If requested it also projects the data onto those dimensions.
    
    Args:
        data (nt, nch): time series data where each channel is considered a 'feature' (nt=n_samples, nch=n_features)
        max_dims (int): (default None) the maximum number of dimensions to reduce data onto.
        VAF (float): (default 0.9) variance accounted for (VAF)
        project_data (bool): (default False). If the function should project the high dimensional input data onto the calculated number of dimensions

    Returns:
        tuple: Tuple containing: 
            | **explained_variance (list ndims long):** variance accounted for by each principal component
            | **num_dims (int):** number of principal components required to account for variance
            | **projected_data (nt, ndims):** Data projected onto the dimensions required to explain the input variance fraction. If the input 'project_data=False', the function will return 'projected_data=None'
    """
    pca = PCA()
    pca.fit(data)
    explained_variance = pca.explained_variance_ratio_
    total_explained_variance = np.cumsum(explained_variance)
    if max_dims is None:
        num_dims = np.min(np.where(total_explained_variance>VAF)[0])+1
    else:
        temp_dims = np.min(np.where(total_explained_variance>VAF)[0])+1
        num_dims = np.min([max_dims, temp_dims])

    if project_data:
        all_projected_data = pca.transform(data)
        projected_data = all_projected_data[:,:num_dims]
    else:
        projected_data = None

    return list(explained_variance), num_dims, projected_data


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
            | **extremum_time (float):** Interpolated (or extrapolated) peak time        
            | **extremum_value (float):** Approximated peak value.        
            | **f (np.poly):** Polynomial used to calculate peak time
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

def calc_task_rel_dims(neural_data, kin_data, conc_proj_data=False):
    '''
    Calculates the task relevant dimensions by regressing neural activity against kinematic data using least squares.
    If the input neural data is 3D, all trials will be concatenated to calculate the subspace. 
    Calculation is based on the approach used in Sun et al. 2022 https://doi.org/10.1038/s41586-021-04329-x
    
    .. math::
    
        R \\in \\mathbb{R}^{nt \\times nch}
        M \\in \\mathbb{R}^{nt \\times ndim}
        \\beta \\in \\mathbb{R}^{nch \\times ndim}
        R = M\\beta^T
        [\\beta_0 \beta_x \beta_y]^T = (M^T M)^{-1} M^T R

    Args:
        neural_data ((nt, nch) or list of (nt, nch)): Input neural data (:math:`R`) to regress against kinematic activity.
        kin_data ((nt, ndim) or list of (nt, ndim)): Kinematic variables (:math:`M`), commonly position or instantaneous velocity. 'ndims' refers to the number of physical dimensions that define the kinematic data (i.e. X and Y)
        conc_proj_data (bool): If the projected neural data should be concatenated.

    Returns:
        tuple: Tuple containing:
            | **(nch, ndim):** Subspace (:math:`\beta`) that best predicts kinematic variables. Note the first column represents the intercept, then the next dimensions represent the behvaioral variables
            | **((nt, nch) or list of (nt, ndim)):** Neural data projected onto task relevant subspace

    '''

    # If a list of segments from trials, concatenate them into one larget timeseries
    if type(neural_data) == list:
        ntrials = len(neural_data)

        conc_neural_data = np.vstack(neural_data) #(nt, nch)
        ntime = conc_neural_data.shape[0]
        
        # Set input neural data as a float
        conc_neural_data = conc_neural_data.astype(float)

        conc_kin_data = np.zeros((ntime,kin_data[0].shape[1]+1))*np.nan
        conc_kin_data[:,0] = 1
        conc_kin_data[:,1:] = np.vstack(kin_data)

        # Center neural data:
        conc_neural_data -= np.nanmean(conc_neural_data, axis=0)

        # Calculate task relevant subspace 
        task_subspace = np.linalg.pinv(conc_kin_data.T @ conc_kin_data) @ conc_kin_data.T @ conc_neural_data
    
    else:
        # Save original neural data as a list
        neural_data = [neural_data]
        
        # Set input neural data as a float
        neural_data_centered = neural_data[0].astype(float)
        
        # Center neural data:
        neural_data_centered -= np.nanmean(neural_data_centered, axis=0)
        ntime = neural_data_centered.shape[0]
        conc_kin_data = np.zeros((ntime, kin_data.shape[1]+1))*np.nan
        conc_kin_data[:,0] = 1
        conc_kin_data[:,1:] = kin_data
        
        # Calculate task relevant subspace 
        task_subspace = np.linalg.pinv(conc_kin_data.T @ conc_kin_data) @ conc_kin_data.T @ neural_data_centered
        ntrials = 1
        
    # Project neural data onto task subspace
    projected_data = []
    
    for itrial in range(ntrials):
        projected_data.append(neural_data[itrial] @ np.linalg.pinv(task_subspace))

    if conc_proj_data:
        return task_subspace.T, np.vstack(projected_data)
    else:    
        return task_subspace.T, projected_data

'''
METRIC CALCULATIONS
'''
def calc_rms(signal, remove_offset=True):
    '''
    Root mean square of a signal
    
    Args:
        signal (nt, ...): voltage along time, other dimensions will be preserved
        remove_offset (bool): if true, subtract the mean before calculating RMS

    Returns:
        float array: rms of the signal along the first axis. output dimensions will be the same non-time dimensions as the input signal
    '''
    if remove_offset:
        m = np.mean(signal, axis=0)
    else:
        m = 0
    
    return np.sqrt(np.mean(np.square(signal - m), axis=0))

def find_outliers(data, std_threshold):   
    '''
    Use kmeans clustering to find the center point of a dataset and distances from each data point
    to the center point. Data points further than a specified number of standard deviations away
    from the center point are labeled as outliers. This is particularily useful for high dimensional data
    
    Note: This function only uses the kmeans function to calculate centerpoint distances but does
    not output any useful information about data clusters. 
    
    Example::

        >>> data = np.array([[0.5,0.5], [0.75,0.75], [1,1], [10,10]])
        >>> outliers_labels, distance = aopy.analysis.find_outliers(data, 2)
        >>> print(outliers_labels, distance)
        [True, True, True, False] [3.6239, 3.2703, 2.9168, 9.8111]

    Args:
        data (n, nfeatures): Input data to plot in an nfeature dimensional space and compute outliers
        std_threshold (float): Number of standard deviations away a data point is required to be to be classified as an outlier
        
    Returns:
        tuple: Tuple containing: 
            | **good_data_idx (n):** Labels each data point if it is an outlier (True = good, False = outlier)
            | **distances (n):** Distance of each data point from center
    '''
    
    # Check ncluster input
    kmeans_model = KMeans(n_clusters = 1).fit(data)
    distances = kmeans_model.transform(data)
    cluster_labels = kmeans_model.labels_
    dist_std = np.sqrt(np.sum(distances**2)/len(distances))
    good_data_idx = (distances < (dist_std*std_threshold))
                  
    return good_data_idx.flatten(), distances.flatten()

def fit_linear_regression(X:np.ndarray, Y:np.ndarray, coefficient_coeff_warning_level:float = 0.5) -> np.ndarray:
    """
    Function that fits a linear regression to each matching column of X and Y arrays. 
    
    Args:
        X [np.ndarray]: number of data points by number of columns. columns of independant vars. 
        Y [np.ndarray]: number of data points by number of columns. columns of dependant vars
        coeffcient_coeff_warning_level (float): if any column returns a corr coeff less than this level 

    Returns:
        tuple: tuple containing:
            | **slope (n_columns):** slope of each fit
            | **intercept (n_columns):** intercept of each fit
            | **corr_coefficient (n_columns):** corr_coefficient of each fit
    """
    
    # Make sure the same shape
    assert X.shape == Y.shape
    
    n_columns = X.shape[1]

    slope = np.empty((n_columns,))
    intercept = np.zeros((n_columns,))
    corr_coeff = np.zeros((n_columns,))
    
    # Iterate through the columns
    for i in range(n_columns):
        
        x = X[:,i]
        y = Y[:,i]
        
        slope[i], intercept[i], corr_coeff[i],  *_ = scipy.stats.linregress(x, y)

        if abs(corr_coeff[i]) <= coefficient_coeff_warning_level: 
            warnings.warn(f'when fitting column number {i}, the correlation coefficient is {corr_coeff[i]}, less than {coefficient_coeff_warning_level} ')
        
    return slope, intercept, corr_coeff

def calc_freq_domain_amplitude(data, samplerate, rms=False):
    '''
    Use FFT to decompose time series data into frequency domain to calculate the
    amplitude of the non-negative frequency components

    Args:
        data (nt, nch): timeseries data, can be a single channel vector
        samplerate (float): sampling rate of the data
        rms (bool, optional): compute root-mean square amplitude instead of peak amplitude

    Returns:
        tuple: Tuple containing:
            | **freqs (nt):** array of frequencies (essentially the x axis of a spectrogram) 
            | **amplitudes (nt, nch):** array of amplitudes at the above frequencies (the y axis)
    '''
    if np.ndim(data) < 2:
        data = np.expand_dims(data, 1)

    # Compute FFT along time dimension
    freq_data = np.fft.fft(data, axis=0)
    length = np.shape(freq_data)[0]
    freq = np.fft.fftfreq(length, d=1./samplerate)
    data_ampl = abs(freq_data[freq>=0,:])*2/length # compute the one-sided amplitude
    non_negative_freq = freq[freq>=0]

    # Apply factor of root 2 to turn amplitude into RMS amplitude
    if rms:
        data_ampl[1:,:] = data_ampl[1:,:]/np.sqrt(2)
    return non_negative_freq, data_ampl

def calc_ISI(data, fs, bin_width, hist_width, plot_flag = False):
    '''
    Computes inter-spike interval histogram. The input data is the sampled thresholded data (0 or 1 data).

    Example:
        >>> data = np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]])
        >>> data_T = data.T
        >>> fs = 100
        >>> bin_width = 0.01
        >>> hist_width = 0.1
        >>> ISI_hist, hist_bins = analysis.calc_ISI(data_T, fs, bin_width, hist_width)
        >>> print(ISI_hist) 
            [[0. 0.]
            [2. 3.]
            [2. 1.]
            [2. 1.]
            [1. 2.]
            [0. 0.]
            [0. 0.]
            [0. 0.]
            [0. 0.]]

    Args:
        data (nt, n_unit): time series spike data with multiple units.
        fs (float): sampling rate of data [Hz]
        bin_width (float): bin_width to compute histogram [s]
        hist_width (float): determines bin edge (0 < t < histo_width) [s]
        plot_flag (bool, optional): display histogram. In plotting, number of intervals is summed across units.

    Returns:
        ISI_hist (n_bins, n_unit) : number of intervals
        hist_bins (n_bins): bin edge to compute histogram
    '''

    n_unit = data.shape[1]
    dT = 1/fs
    hist_bins = np.arange(0, hist_width, bin_width)

    ISI_hist = np.zeros((len(hist_bins)-1, n_unit))
    for iU in range(n_unit):
        spike_idx = np.where( data[:,iU] )
        ISI = np.diff(spike_idx)*dT
        ISI_hist[:,iU], _ = np.histogram(ISI, hist_bins)
    
    hist_bins = hist_bins[:-1] + np.diff(hist_bins)/2 # change hist_bins to be the center of the bin, not the edges

    # for plot
    if plot_flag:
        plt.bar(hist_bins*1000, np.sum(ISI_hist,axis=1), width = bin_width*1000, edgecolor="black") #multiplied 1000 to rescale to [ms]
        plt.xlabel('Interspike interval (ms)')
        plt.ylabel('Number of intervals')

    return ISI_hist, hist_bins

def calc_sem(data, axis=None):
    '''
    This function calculates the standard error of the mean (SEM). The SEM is calculated with the following equation
    where :math:`\sigma` is the standard deviation and :math:`n` is the number of samples. When the data matrix includes NaN values,
    this function ignores them when calculating the :math:`n`. If no value for axis is input, the SEM will be 
    calculated across the entire input array.

    .. math::
    
        SEM = \\frac{\\sigma}{\\sqrt{n}}
        

    Args:
        data (nd array): Input data matrix of any dimension
        axis (int or tuple): Axis to perform SEM calculation on
    
    Returns:
        nd array: SEM value(s).
    '''
    n = np.sum(~np.isnan(data), axis=axis)
    SEM = np.nanstd(data, axis=axis)/np.sqrt(n)

    return SEM

def calc_rolling_average(data, window_size=11):
    """
    Computes the rolling average of a 1D array using a convolutional kernel. The
    first and last valid datapoint (fully overlapping with the kernel) are copied
    backwards and forwards, respectively, such that the size of the output is the
    same as the size of the input.

    Args:
        data (nt): The 1D array of data to compute the rolling average for.
        window_size (int): The size of the kernel in number of samples. Must be odd.
    
    Returns:
        (nt,) array: The rolling average of the input data.
    """
    assert window_size % 2 == 1, "Kernel size must be odd."
    
    kernel = np.ones(window_size) / window_size
    data_convolved = np.convolve(data, kernel, mode='same')
    mid_kernel_idx = math.floor(window_size/2)
    data_convolved[:mid_kernel_idx] = data_convolved[mid_kernel_idx]
    data_convolved[-mid_kernel_idx:] = data_convolved[-(mid_kernel_idx+1)]
    return data_convolved

def calc_corr_over_elec_distance(acq_data, acq_ch, elec_pos, bins=20, method='spearman', exclude_zero_dist=True):
    '''
    Calculates mean absolute correlation between acq_data across channels with the same distance between them.
    
    Args:
        acq_data (nt, nch): acquisition data indexed by acq_ch
        acq_ch (nelec): 1-indexed list of acquisition channels that are connected to electrodes
        elec_pos (nelec, 2): x, y position of each electrode
        bins (int or array): input into scipy.stats.binned_statistic, can be a number or a set of bins
        method (str, optional): correlation method to use ('pearson' or 'spearman')
        exclude_zero_dist (bool, optional): whether to exclude distances that are equal to zero. default True
        
    Returns:
        tuple: tuple containing:
            |**dist (nbins):** electrode distance at each bin
            |**corr (nbins):** correlation at each bin

    '''
    dist = utils.calc_euclid_dist_mat(elec_pos)
    if method == 'spearman':
        c, _ = stats.spearmanr(acq_data, axis=0)
    elif method == 'pearson':
        c = np.corrcoef(acq_data.T)
    else:
        raise ValueError(f"Unknown correlation method {method}")
    
    c_ = c[np.ix_(acq_ch-1, acq_ch-1)] # note use of open mesh to get the right logical index
    
    if exclude_zero_dist:
        zero_dist = dist == 0
        dist = dist[~zero_dist]
        c_ = c_[~zero_dist]
        
    corr, edges, _ = stats.binned_statistic(dist.flatten(), np.abs(c_.flatten()), statistic='mean', bins=bins)
    dist = (edges[:-1] + edges[1:]) / 2

    return dist, corr

def calc_erp(data, event_times, time_before, time_after, samplerate, subtract_baseline=True, baseline_window=None):
    '''
    Calculates the event-related potential (ERP) for the given timeseries data.

    Args:
        data (nt, nch): timeseries data across channels
        event_times (ntrial): list of event times
        time_before (float): number of seconds to include before each event
        time_after (float): number of seconds to include after each event
        samplerate (float): sampling rate of the data
        subtract_baseline (bool, optional): if True, subtract the mean of the aligned data during
            the time_before period preceding each event. Must supply a positive time_before. Default True
        baseline_window ((2,) float, optional): range of time to compute baseline (in seconds before event)
            Default is the entire time_before period.

    Returns:
        (ntr, nt, nch): array of event-aligned responses for each channel during the given time periods

    '''
    if subtract_baseline and time_before <= 0:
        raise ValueError("Input time_before must be positive in order to calculate baseline")
        
    # Align the data to the given event times (shape is [trials x time x channels])
    n_events = len(event_times)
    aligned_data = preproc.trial_align_data(data, event_times, time_before, time_after, samplerate)

    if subtract_baseline:
        
        # Take a mean across the data before the events as a baseline
        if not baseline_window:
            baseline_window = (0, time_before)
        elif len(baseline_window) < 2 or baseline_window[1] < baseline_window[0]:
            raise ValueError("baseline_window must be in the form (t0, t1) where \
                t1 is greater than t0")
        before_samples = int(time_before*samplerate)
        s0 = before_samples - int(baseline_window[1]*samplerate)
        s1 = before_samples - int(baseline_window[0]*samplerate)
        event_mean = np.mean(aligned_data[:,s0:s1,:], axis=1)

        # Subtract the baseline to calculate ERP
        n_samples = aligned_data.shape[1]
        event_mean = np.tile(event_mean, (n_samples, 1, 1)).swapaxes(0,1)
        erp = aligned_data - event_mean
    else:

        # Just use the aligned data as-is
        erp = aligned_data

    return erp

def calc_max_erp(data, event_times, time_before, time_after, samplerate, subtract_baseline=True, baseline_window=None, max_search_window=None, trial_average=True):
    '''
    Calculates the maximum (across time) mean (across trials) event-related potential (ERP) 
    for the given timeseries data.
    
    Args:
        data (nt, nch): timeseries data across channels
        event_times (ntrial): list of event times
        time_before (float): number of seconds to include before each event
        time_after (float): number of seconds to include after each event
        samplerate (float): sampling rate of the data
        subtract_baseline (bool, optional): if True, subtract the mean of the aligned data during
            the time_before period preceding each event. Must supply a positive time_before. Default True
        baseline_window ((2,) float, optional): range of time to compute baseline (in seconds before event)
            Default is the entire time_before period.
        max_search_window ((2,) float, optional): range of time to search for maximum value (in seconds 
            after event). Default is the entire time_after period.
        trial_average (bool, optional): by default, average across trials before calculating max
        
    Returns:
        nch: array of maximum mean-ERP for each channel during the given time periods
    '''
    erp = calc_erp(data, event_times, time_before, time_after, samplerate, subtract_baseline, baseline_window)
    
    if trial_average:
        erp = np.expand_dims(np.nanmean(erp, axis=0), 0)

    # Limit the search to the given window
    start_idx = int(time_before*samplerate)
    end_idx = start_idx + int(time_after*samplerate)
    if max_search_window:
        if len(max_search_window) < 2 or max_search_window[1] < max_search_window[0]:
            raise ValueError("max_search_window must be in the form (t0, t1) where \
                t1 is greater than t0")
        end_idx = start_idx + int(max_search_window[1]*samplerate)
        start_idx += int(max_search_window[0]*samplerate)
    
    # Caculate max separately for each trial
    erp_window = erp[:,start_idx:end_idx,:]
    max_erp = np.zeros((erp.shape[0], erp.shape[2]))*np.nan
    for t in range(erp.shape[0]):
            
        # Find the index that maximizes the absolute value, then use that index to get the actual signed value
        idx_max_erp = start_idx + np.argmax(np.abs(erp_window[t]), axis=0)
        max_erp[t,:] = np.squeeze([erp[t,idx_max_erp[i],i] for i in range(erp.shape[2])])

    if trial_average:
        max_erp = max_erp[0]
        
    return max_erp
    
'''
MODEL FITTING
'''
def linear_fit_analysis2D(xdata, ydata, weights=None, fit_intercept=True):
    '''
    This functions fits a line to input data using linear regression, calculates the fitting score
    (coefficient of determination), and calculates Pearson's correlation coefficient. Optional weights
    can be input to adjust the linear fit. This function then applies the linear fit to the input xdata.

    Linear regression fit is calculated using:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    
    Pearson correlation coefficient is calculated using:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html


    Args:
        xdata (npts):
        ydata (npts):
        weights (npts):

    Returns:
        tuple: Tuple containing:
            | **linear_fit (npts):** Y value of the linear fit corresponding to each point in the input xdata.
            | **linear_fit_score (float):** Coefficient of determination for linear fit
            | **pcc (float):** Pearson's correlation coefficient
            | **pcc_pvalue (float):** Two tailed p-value corresponding to PCC calculation. Measures the significance of the relationship between xdata and ydata.
            | **reg_fit (sklearn.linear_model._base.LinearRegression)
    '''
    xdata = xdata.reshape(-1, 1)
    ydata = ydata.reshape(-1,1)

    reg_fit = LinearRegression(fit_intercept=fit_intercept).fit(xdata,ydata, sample_weight=weights)
    linear_fit_score = reg_fit.score(xdata, ydata)
    pcc_all = stats.pearsonr(xdata.flatten(), ydata.flatten())

    linear_fit = reg_fit.coef_[0][0]*xdata.flatten() + reg_fit.intercept_

    return linear_fit, linear_fit_score, pcc_all[0], pcc_all[1], reg_fit

def classify_by_lda(X_train_lda, y_class_train, 
                                 n_splits=5,
                                 n_repeats=3, 
                                 random_state=1):
    """
    Trains a linear discriminant model on the training data (X_train_lda) and their labels (y_class_train) with data spliting and
    k-fold validation. Returns accuracy and variance based on how well the model is able to predict the left-out data.

    Args:
        X_train_lda (n_classes, n_features): 2d training data. first dimension is the number of examples, second dimension is the size of each example
        y_class_train (n_classes): class to which each example belongs
        n_splits (int, optional): number of paritions to split data Defaults to 5.
        n_repeats (int, optional): number of repeated fitting Defaults to 3.
        random_state (int, optional): random state for data spliting Defaults to 1.

    Returns:
        accuracy (float): mean accuracy of the repeated lda runs.
        std (float): standard deviation of the repeated lda runs.
    """

    assert X_train_lda.shape[0] == len(y_class_train)

    # get the model
    model = LinearDiscriminantAnalysis()
    
    # define model evaluation method
    cv = model_selection.RepeatedStratifiedKFold(n_splits=n_splits,
                                 n_repeats=n_repeats, 
                                 random_state=random_state)
    # evaluate model
    scores = model_selection.cross_val_score(model, X_train_lda, y_class_train, 
                                            scoring='accuracy', cv=cv, n_jobs=-1)

    mean_accuracy,  std = np.mean(scores), np.std(scores)

    return mean_accuracy, std

'''
Spectral Estimation and Analysis
'''
def calc_cwt_tfr(data, freqs, samplerate, fb=1.5, f0_norm=1.0, verbose=False):
    '''
    Use morlet wavelet decomposition to calculate a time-frequency representation of your data.
    
    Args:
        data (nt, nch): time series data
        freqs (nfreq): frequencies to decompose
        samplerate (float): sampling rate of the data
        fb (float, optional): time-decay parameter, inverse relationship with bandwidth 
            of the wavelets; setting a higher number results in narrower frequency resolution
        f0_norm (float, optional): center frequency of the wavelets, normalized to the sampling 
            rate. Default to 1.0, or the same frequency as the sampling rate.
        verbose (bool, optional): print out information about the wavelets

    Returns:
        (nfreq, nt, nch): tfr representation for each channel

    Examples:
        
        .. code-block:: python

            fig, ax = plt.subplots(3,1,figsize=(4,6))

            samplerate = 1000
            data_200_hz = aopy.utils.generate_multichannel_test_signal(2, samplerate, 8, 200, 2)
            nt = data_200_hz.shape[0]
            data_200_hz[:int(nt/3),:] /= 3
            data_200_hz[int(2*nt/3):,:] *= 2

            data_50_hz = aopy.utils.generate_multichannel_test_signal(2, samplerate, 8, 50, 2)
            data_50_hz[:int(nt/2),:] /= 2

            data = data_50_hz + data_200_hz
            print(data.shape)
            aopy.visualization.plot_timeseries(data, samplerate, ax=ax[0])
            aopy.visualization.plot_freq_domain_amplitude(data, samplerate, ax=ax[1])

            freqs = np.linspace(1,250,100)
            coef = aopy.analysis.calc_cwt_tfr(data, freqs, samplerate, fb=10, f0_norm=1, verbose=True)
            t = np.arange(nt)/samplerate
            
            print(data.shape)
            print(coef.shape)
            print(t.shape)
            print(freqs.shape)
            pcm = aopy.visualization.plot_tfr(abs(coef[:,:,0]), t, freqs, 'plasma', ax=ax[2])

            fig.colorbar(pcm, label='Power', orientation = 'horizontal', ax=ax[2])
            
        .. image:: _images/tfr_cwt_50_200.png

    '''
    freqs = np.flip(freqs)/samplerate
    wav = pywt.ContinuousWavelet(f'cmor{fb}-{f0_norm}') # 'cmorB-C' for a complex Morlet wavelet with the
                                                        # given time-decay (B) and center frequency (C) params.
    scale = pywt.frequency2scale(wav, freqs)
    coef, _ = pywt.cwt(data, scale, wav, axis=0)
    if verbose:
        print(wav.bandwidth_frequency)
        print(f"Wavelet ({wav.lower_bound}, {wav.upper_bound})")
        print(f"Scale ({scale[0]}, {scale[-1]})")
        print(f"Freqs ({freqs[0]}, {freqs[-1]})")
    
    return np.flip(coef, axis=0)

def get_sgram_multitaper(data, fs, win_t, step_t, nw=None, bw=None, adaptive=False):
    """get_sgram_multitaper

    Compute multitaper estimate from multichannel signal input.

    Args:
        data (nt, nch): nd array of input neural data (multichannel)
        fs (int): sampling rate
        win_t (float): spectrogram window length (in seconds)
        step_t (float): step size between spectrogram windows (in seconds)
        nw (float, optional): time-half-bandwidth product. Defaults to None.
        bw (float, optional): spectrogram frequency bin bandwidth. Defaults to None.
        adaptive (bool, optional): adaptive taper weighting. Defaults to False.

    Returns:
        tuple: Tuple containing:
            | **fxx (np.array):** spectrogram frequency array (equal in length to win_t * fs // 2 + 1)
            | **txx (np.array):** spectrogram time array (equal in length to (len(data)/fs - win_t)/step_t)
            | **Sxx (len(fxx):** x len(txx) x nch): multitaper spectrogram estimate. Last dimension squeezed for 1-d inputs.
    """
    jackknife = False
    sides = 'onesided'
    if len(data.shape) < 2:
        data = data[:,None]
    assert len(data.shape) < 3, f"only 1- or 2-dim data arrays accepted - {data.shape}-dim input given"
    (n_sample, n_ch) = data.shape
    total_t = n_sample/fs
    n_window = int((total_t-win_t)/step_t)
    assert n_window > 0
    window_len = int(win_t*fs)
    step_len = int(step_t*fs)
    n_fbin = window_len // 2 + 1
    txx = np.arange(n_window)*step_t # window start time
    Sxx = np.zeros((n_fbin,n_window,n_ch))

    data = interp_multichannel(data)

    for idx_window in range(n_window):
        window_sample_range = np.arange(window_len) + step_len*idx_window
        win_data = data[window_sample_range,:]
        _f, _win_psd, _ = tsa.multi_taper_psd(win_data.T, fs, nw, bw, adaptive, jackknife, sides)
        try:
            Sxx[:,idx_window,...] = _win_psd.T
        except:
            breakpoint()
    if n_ch == 1:
        Sxx = Sxx.squeeze(axis=-1)

    fxx = _f

    return fxx, txx, Sxx

def calc_mt_tfr(ts_data, n, p, k, fs, step=None, fk=None, pad=2, ref=True):
    '''
    Compute multitaper time-frequency estimate from multichannel signal input. This code is adapted from the Pesaran lab `tfspec`.    
    
    Args:
        ts_data (nt, nch): time series array
        n (int): window length in seconds
        p (int): standardized half bandwidth in hz
        k (int): number of DPSS tapers to use
        fs (float): sampling rate
        step (float): window step. Defaults to step = n/10.
        fk (float): frequency range to return in Hz ([0, fk]). Defaults to fs/2.
        pad (int):  padding factor for the FFT. This should be 1 or a multiple of 2.
                    For N=500, if pad=1, we pad the FFT to 512 points.
                    If pad=2, we pad the FFT to 1024 points. 
                    If pad=4, we pad the FFT to 2024 points.
        ref (bool): referencing flag. If True, mean of neural signals across electrodes 
                    for each time window is subtracted to remove common noise
                    so that you can get spacially-localized signals.
                    If you only analyze single channel data, this has to be False.
                    This paper discuss referencing scheme
                    https://iopscience.iop.org/article/10.1088/1741-2552/abce3c
                       
    Returns:
        f (n_freq): frequency axis for spectrogram
        t (n_time): time axis for spectrogram
        spec (n_freq,n_time,nch): multitaper spectrogram estimate
        
    Examples:
        
        .. code-block:: python

            fig, ax = plt.subplots(3,1,figsize=(4,6))

            NW = 0.075
            BW = 20
            n, p, k = aopy.precondition.convert_taper_parameters(NW, BW)
            step = None
            fk = 1000
            samplerate = 1000
            
            t = np.arange(2*samplerate)/samplerate
            f0 = 1
            t1 = 2
            f1 = 1000
            data = 1e-6*np.expand_dims(signal.chirp(t, f0, t1, f1, method='quadratic', phi=0),1)

            fig, ax = plt.subplots(3,1,figsize=(4,6),tight_layout=True)
            aopy.visualization.plot_timeseries(data, samplerate, ax=ax[0])
            aopy.visualization.plot_freq_domain_amplitude(data, samplerate, ax=ax[1])
            f_spec,t_spec,spec = aopy.analysis.calc_mt_tfr(data, n, p, k, samplerate, step=step, fk=fk, pad=2, ref=False)
            pcm = aopy.visualization.plot_tfr(spec[:,:,0], t_spec, f_spec, 'plasma', ax=ax[2])
            fig.colorbar(pcm, label='Power', orientation = 'horizontal', ax=ax[2])
            
        .. image:: _images/tfr_mt_chirp.png
            
        .. code-block:: python
            T = 5
            t = np.linspace(0,T,num_points)
            test_data = np.zeros((t.shape[0],2))
            test_data[:,0] = 1.5*np.sin(2*np.pi*10*t)
            test_data[:,1] = 2*np.cos(2*np.pi*30*t)
            test_data[t>=T/2,0] = 2*np.sin(2*np.pi*5*t[t<=T/2])
            test_data[t>=T/2,1] = 1*np.cos(2*np.pi*15*t[t<=T/2])
    
            NW = 2
            BW = 1
            fs = num_points/T
            dn = 0.1
            fk = 50
            n, p, k = aopy.precondition.convert_taper_parameters(NW, BW)
            f_spec,t_spec,spec = aopy.analysis.calc_mt_tfr(test_data, n, p, k, fs, dn, fk, pad=2, ref=False)

            fig,ax=plt.subplots(1,4,figsize=(8,2),tight_layout=True)
            ax[0].plot(t,test_data[:,0],linewidth=0.2)
            ax[0].plot(t,test_data[:,1],linewidth=0.2)
            ax[0].set(xlabel='Time (s)',ylabel='Amplitude',title='Signals')
            ax[1].imshow((spec[:,:,0]),aspect='auto',origin='lower',extent=[0,T,0,f_spec[-1]])
            ax[1].set(ylabel='Frequency (Hz)',xlabel='Time [s]',title='Spectrogram (ch1)')
            ax[2].imshow((spec[:,:,1]),aspect='auto',origin='lower',extent=[0,T,0,f_spec[-1]])
            ax[2].set(ylabel='Frequency (Hz)',xlabel='Time [s]',title='Spectrogram (ch2)')
            ax[3].plot(f_spec,spec[:,10,0],'-',label='ch 1')
            ax[3].plot(f_spec,spec[:,10,1],'-',label='ch 2')
            ax[3].set(ylabel='Power',xlabel='Frequency (Hz)',xlim=(0,50),title='Power spectral')
            ax[3].legend(title=f't = {t_spec[10]}s',frameon=False, fontsize=7)
            
        .. image:: _images/tfspec.png

        
    See Also:
        :func:`~aopy.analysis.calc_cwt_tfr`
    '''   
    
    def nextpow2(x):
        #   Next higher power of 2.
        #   NEXTPOW2(N) returns the first P such that 2.^P >= abs(N).
        #   It is often useful for finding the nearest power of two sequence length for FFT operations.
        return 1 if x == 0 else math.ceil(math.log2(x))
    
    if ts_data.ndim == 1:
        ts_data = np.reshape(ts_data,(-1,1))
    if ts_data.shape[1] == 1:
        ref = False
    if step == None:
        step = n/10
    if fk == None:
        fk = fs/2
        
    ts_data = ts_data.T
    nch,nt = ts_data.shape
    fk = np.array([0,fk])
    tapers, _ = dpsschk(n*fs, p, k)
    
    win_size = tapers.shape[0] # window size (data points of tapers)
    step_size = int(np.floor(step*fs)) # step size
    nf = np.max([256,pad*2**nextpow2(win_size+1)]) # 0 padding for efficient computation in FFT
    nfk = np.floor(fk/fs*nf) # number of data points in frequency axis
    nwin = int(np.floor((nt-win_size)/step_size)) # number of windows
    f = np.linspace(fk[0],fk[1],int(np.diff(nfk))) # frequency axis for spectrogram

    spec = np.zeros((int(np.diff(nfk)),nwin,nch))
    for iwin in range(nwin):
        if ref:
            m_data = np.sum(ts_data[:,step_size*iwin:step_size*iwin+win_size],axis=0)/nch # Mean across channels for that window
            win_data = (ts_data[:,step_size*iwin:step_size*iwin+win_size]-m_data).T # Subtract mean from data           
        else:
            win_data = (ts_data[:,step_size*iwin:step_size*iwin+win_size]).T
        
        # Compute power for each taper
        power = 0
        for ik in range(k):
            tapers_ik = tapers[:,ik].reshape(-1,1) # keep the dimension for broadcast
            fk_data = fft(tapers_ik*win_data, nf, axis=0) # tapers_ik.shape:(win_size,1), win_data.shape:(win_size,nch)
            fk_data = fk_data[int(nfk[0]):int(nfk[1]),:] # extract a part of frequency components
            power += fk_data*fk_data.conj()
        power = power/k # average power across number of tapers.   
        spec[:,iwin,:] = power.real  # Power is already real at this time, but still have 0j part.
    
    t = np.arange(nwin)*step + n/2 # Center of each window is time axis
    
    return f, t, spec

def get_psd_multitaper(data, fs, NW=None, BW=None, adaptive=False, jackknife=True, sides='default'):
    '''
    Computes power spectral density using Multitaper functions

    Args:
        data (nt, nch): time series data where time axis is assumed to be on the last axis
        fs (float): sampling rate of the signal
        NW (float): Normalized half bandwidth of the data tapers in Hz
        BW (float): sampling bandwidth of the data tapers in Hz
        adaptive (bool): Use an adaptive weighting routine to combine the PSD estimates of different tapers.
        jackknife (bool): Use the jackknife method to make an estimate of the PSD variance at each point.
        sides (str): This determines which sides of the spectrum to return.

    Returns:
        tuple: Tuple containing:
            | **f (nfft):** Frequency points vector
            | **psd_est (nfft, nch):** estimated power spectral density (PSD)
            | **nu (nfft, nch):** if jackknife = True; estimated variance of the log-psd. If Jackknife = False; degrees of freedom in a chi square model of how the estimated psd is distributed wrt true log - PSD
    '''
    data = data.T # move time to the last axis
    
    f, psd_mt, nu = tsa.multi_taper_psd(data, fs, NW, BW,  adaptive, jackknife, sides)
    return f, psd_mt.T, nu.T

def multitaper_lfp_bandpower(f, psd_est, bands, no_log):
    '''
    Estimate band power in specified frequency bands using multitaper power spectral density estimate

    Args:
        f (nfft) : Frequency points vector
        psd_est (nfft, nch): power spectral density - output of bandpass_multitaper_filter_data
        bands (list): lfp bands should be a list of tuples representing ranges e.g., bands = [(0, 10), (10, 20), (130, 140)] for 0-10, 10-20, and 130-140 Hz
        no_log (bool): boolean to select whether lfp band power should be in log scale or not

    Returns:
        lfp_power (n_features, nch): lfp band power for each channel for each band specified
    '''
    if psd_est.ndim == 1:
        psd_est = np.expand_dims(psd_est, 1)

    lfp_power = np.zeros((len(bands), psd_est.shape[1]))
    small_epsilon = 0 # TODO: what is this for? It does nothing now, should it be possible to make nonzero? -Leo
    fft_inds = dict()

    for band_idx, band in enumerate(bands):
            fft_inds[band_idx] = [freq_idx for freq_idx, freq in enumerate(f) if band[0] <= freq < band[1]]

    for idx, band in enumerate(bands):
        if no_log:
            lfp_power[idx, :] = np.mean(psd_est[fft_inds[idx],:], axis=0)
        else:
            lfp_power[idx, :] = np.mean(np.log10(psd_est[fft_inds[idx],:] + small_epsilon), axis=0)

    return lfp_power

def get_psd_welch(data, fs,n_freq = None):
    '''
    Computes power spectral density using Welch's method. Welch’s method computes an estimate of the power spectral density by dividing the data into overlapping segments, computes a modified periodogram for each segment and then averages the periodogram. Periodogram is averaged using median.

    Args:
        data (nt, ...): time series data.
        fs (float): sampling rate
        n_freq (int): no. of frequency points expected

    Returns:
        tuple: Tuple containing:
            | **f (nfft):** frequency points vector
            | **psd_est (nfft, ...):** estimated power spectral density (PSD)
    '''
    if n_freq:
        f, psd = signal.welch(data, fs, average='median', nperseg=2*n_freq, axis=0)
    else:
        f, psd = signal.welch(data, fs, average='median', axis=0)
    return f, psd

def interp_multichannel(x):
    """interp_multichannel

    Args:
        x (n_sample, n_ch): input data array containing nan-valued missing entries

    Returns:
        x_interp (n_sample, n_ch): interpolated data, uses `numpy.interp` method.
    """
    nan_idx = np.isnan(x)
    ok_idx = ~nan_idx
    xp = ok_idx.ravel().nonzero()[0]
    fp = x[ok_idx]
    idx = nan_idx.ravel().nonzero()[0]
    x[nan_idx] = np.interp(idx,xp,fp)

    return x


def calc_corr2_map(data1, data2, knlsz=15, align_maps=False):
    '''
    This function creates a map showning the local correlation between two input datamaps. If specified, it also aligns the input
    maps by finding the location of the peak of the 2D correlation function. Note, if these shifts are unexpectedly high, there
    is likely not high enough correlation between the datamaps and alignment should not be used. This function uses 0-padding for all
    edge conditions and replaces input NaN values with 0 to calculate the correlation map. After the correlation map is calculated,
    all values were NaN in the input data are again set to NaN. If a window of data has all 0's, the NCC is set to nan. 
    Note, the worst correlation in the example image is not at the edge of the image because of zero padding.

    .. image:: _images/calc_corr2_map.png

    Args:
        data1 (nrow, ncol): First input data array. Used as baseline if map alignment is required.
        data2 (nrow, ncol): Second input data array. Shifted to match the baseline if map alignment is required
        knlsz (int): Length of the kernel window in units of data points. The kernel is a square so each side will have the lenght specified here. This value should always be odd.
        align_maps (bool): Whether or not to align maps.

    Returns:
        tuple: Tuple containing:
            | **NCC (nrow, ncol):** Spatial correlation map (NCC: normalized correlatoin coefficient)
            | **shifts (tuple):** Contains (row_shifts, col_shifts)
    '''
    
    # Make sure knlsz is odd
    if knlsz % 2 == 0:
        print('Warning: Kernel size (knlsz) is even in calc_corr2_map')

    NCC = np.zeros((data1.shape))
    data_sz = data1.shape[0]
    
    # Get nan value locations 
    nan_idx1 = np.isnan(data1)
    nan_idx2 = np.isnan(data2)
    
    # Replace NaNs with 0s so correlation doesn't output NaN
    data1[nan_idx1] = 0
    data2[nan_idx2] = 0
    
    # Get maxidx of 2D spatial correlation matrix to ensure data maps are aligned.
    if align_maps:
        corr = scipy.signal.correlate2d(data1/np.linalg.norm(data1), data2/np.linalg.norm(data2), 
                                        boundary='fill', mode='same')
        irow, icol = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
        row_shift = int(irow - (data1.shape[0]-1)/2)
        col_shift = int(icol - (data1.shape[1]-1)/2)
        data2_align = np.roll(data2, row_shift, axis=0)
        data2_align = np.roll(data2_align, col_shift, axis=1)
        shifts = (row_shift, col_shift)
    else:
        data2_align = data2
        shifts = (0,0)
    
    # Pad data
    data1_pad = np.pad(data1, int((knlsz-1)/2), mode='constant')
    data2_pad = np.pad(data2_align, int((knlsz-1)/2), mode='constant')
    
    start_idx = int((knlsz-1)/2)
    end_idx = int(data_sz + (knlsz-1)/2)
    middle_ncc_idx = int(2*(knlsz-1)/2)
    for xx in range(start_idx, end_idx):
        for yy in range(start_idx,end_idx):
            # Normalize input arrays based on the norm
            data_subset1 = data1_pad[(xx-start_idx):(xx+start_idx+1),(yy-start_idx):(yy+start_idx+1)]
            data_subset2 = data2_pad[(xx-start_idx):(xx+start_idx+1),(yy-start_idx):(yy+start_idx+1)]

            # If either data subset is all 0's set the NCC to 0
            if np.linalg.norm(data_subset1)==0 or np.linalg.norm(data_subset2)==0:
                NCC[xx - start_idx, yy - start_idx] = np.nan
            else:
                data_subset1 /= np.linalg.norm(data_subset1)
                data_subset2 /= np.linalg.norm(data_subset2)
                NCC[xx - start_idx, yy - start_idx] = scipy.signal.correlate2d(data_subset1,data_subset2)[middle_ncc_idx, middle_ncc_idx]
    
    # Replace NaNs in correlation map
    NCC[nan_idx1] = np.nan
    
    return NCC, shifts