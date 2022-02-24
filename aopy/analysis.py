# analysis.py
# Code for neural data analysis; functions here should return interpretable results such as
# firing rates, success rates, direction tuning, etc.

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

import scipy
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy import stats, signal
import warnings
from numpy.linalg import inv as inv  # used in Kalman Filter

import warnings
from . import preproc

TARGET_ON_CODES = range(17, 26)
CURSOR_ENTER_TARGET_CODES = range(81, 89)
TRIAL_END = 239
SUCCESS_CODE = 48
CENTER_ON = 16

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
        num_dims = np.min(np.where(total_explained_variance > VAF)[0]) + 1
    else:
        temp_dims = np.min(np.where(total_explained_variance > VAF)[0]) + 1
        num_dims = np.min([max_dims, temp_dims])

    if project_data:
        all_projected_data = pca.transform(data)
        projected_data = all_projected_data[:, :num_dims]
    else:
        projected_data = None

    return list(explained_variance), num_dims, projected_data


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
    temp_pd = np.arctan2(b2sign * b2 ** 2, b1sign * b1 ** 2)
    if temp_pd < 0:
        pd = (2 * np.pi) + temp_pd
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

    fit_params = np.empty((nunits, 3)) * np.nan
    md = np.empty((nunits)) * np.nan
    pd = np.empty((nunits)) * np.nan

    for iunit in range(nunits):
        # If there is a nan in the values of interest skip curve fitting, otherwise fit
        if ~np.isnan(mean_fr[iunit, :]).any():
            params, _ = curve_fit(curve_fitting_func, targets, mean_fr[iunit, :])
            fit_params[iunit, :] = params

            md[iunit] = get_modulation_depth(params[0], params[1])
            pd[iunit] = get_preferred_direction(params[0], params[1])

        # If this doesn't work, check if fit_with_nans is true. It it is remove nans and fit
        elif fit_with_nans:
            nonnanidx = ~np.isnan(mean_fr[iunit, :])
            if np.sum(nonnanidx) >= min_data_pts:  # If there are enough data points run curve fitting, else return nan
                params, _ = curve_fit(curve_fitting_func, targets[nonnanidx], mean_fr[iunit, nonnanidx])
                fit_params[iunit, :] = params

                md[iunit] = get_modulation_depth(params[0], params[1])
                pd[iunit] = get_preferred_direction(params[0], params[1])
            else:
                md[iunit] = np.nan
                pd[iunit] = np.nan

    return fit_params, md, pd


'''
Performance metrics
'''


def calc_success_percent(events, start_events=[b"TARGET_ON"], end_events=[b"REWARD", b"TRIAL_END"],
                         success_events=b"REWARD", window_size=None):
    '''
    A wrapper around get_trial_segments which counts the number of trials with a reward event 
    and divides by the total number of trials. This function can either calculated the success percent
    across all trials in the input events, or compute a rolling success percent based on the 'window_size' 
    input argument.  

    Args:
        events (nevents): events vector, can be codes, event names, anything to match
        start_events (int, str, or list, optional): set of start events to match
        end_events (int, str, or list, optional): set of end events to match
        success_events (int, str, or list, optional): which events make a trial a successful trial
        window_size (int, optional): [Untis: number of trials] For computing rolling success perecent. How many trials to include in each window. If None, this functions calculates the success percent across all trials.

    Returns:
        float or array (nwindow): success percent = number of successful trials out of all trials attempted.
    '''
    segments, _ = preproc.get_trial_segments(events, np.arange(len(events)), start_events, end_events)
    n_trials = len(segments)
    success_trials = [np.any(np.isin(success_events, trial)) for trial in segments]

    # If requested, calculate success percent across entire input events
    if window_size is None:
        n_success = np.count_nonzero(success_trials)
        success_percent = n_success / n_trials

    # Otherwise, compute rolling success percent
    else:
        filter_array = np.ones(window_size)
        success_per_window = signal.convolve(success_trials, filter_array, mode='valid', method='direct')
        success_percent = success_per_window / window_size

    return success_percent


def calc_success_rate(events, event_times, start_events, end_events, success_events, window_size=None):
    '''
    Args:
        events (nevents): events vector, can be codes, event names, anything to match
        event_times (nevents): time of events in 'events'
        start_events (int, str, or list, optional): set of start events to match
        end_events (int, str, or list, optional): set of end events to match
        success_events (int, str, or list, optional): which events make a trial a successful trial
        window_size (int, optional): [ntrials] For computing rolling success perecent. How many trials to include in each window. If None, this functions calculates the success percent across all trials.

    Returns:
        float or array (nwindow): success rate [success/s] = number of successful trials completed per second of time between the start event(s) and end event(s).
    '''
    # Get event time information
    _, times = preproc.get_trial_segments(events, event_times, start_events, end_events)
    trial_acq_time = times[:, 1] - times[:, 0]
    ntrials = times.shape[0]

    # Get % of successful trials per window 
    success_perc = calc_success_percent(events, start_events, end_events, success_events, window_size=window_size)

    # Determine rolling target acquisition time info 
    if window_size is None:
        nsuccess = success_perc * ntrials
        acq_time = np.sum(trial_acq_time)

    else:
        nsuccess = success_perc * window_size
        filter_array = np.ones(window_size)
        acq_time = signal.convolve(trial_acq_time, filter_array, mode='valid', method='direct')

    success_rate = nsuccess / acq_time

    return success_rate


'''
Cell type classification analysis
'''


def classify_cells_spike_width(waveform_data, samplerate, std_threshold=3, pca_varthresh=0.75, min_wfs=10):
    '''
    Calculates waveform width and classifies units into putative exciatory and inhibitory cell types based on pulse width.
    Units with lower spike width are considered inhibitory cells (label: 0) and higher spike width are considered excitatory cells (label: 1)
    The pulse width is defined as the time between the waveform trough to the waveform peak. (trough-to-peak time)
    Assumes all waveforms are recorded for the same number of time points. 

    This function conducts the following processing steps:

        | **1. ** For each unit, project each waveform into the top PCs. Number of PCs determined by 'pca_varthresh'
        | **2. ** For each unit, remove outlier spikes. Outlier threhsold determined by 'std_threshold'. If the number of waveforms is less than 'min_wf', no waveforms are removed.
        | **3. ** For each unit, average remaining waveforms.
        | **4. ** For each unit, calculate spike width using a local polynomial interpolation.
        | **5. ** Use a gaussian mixture model to classify all units

    Args:
        waveform_data (nunit long list of (nt x nwaveforms) arrays): Waveforms of each unit. Each element of the list is a 2D array for each unit. Each 2D array contains the timeseries of all recorded waveforms for a given unit.
        samplerate (float): sampling rate of the points in each waveform. 
        std_threshold (float): For outlier removal. The maximum number of standard deviations (in PC space) away from the mean a given waveform is allowed to be. Defaults to 3
        pca_varthresh (float): Variance threshold for determining the number of dimensions to project spiking data onto. Defaults to 0.75.
        min_wfs (int): Minimum number of waveform samples required to perform outlier detection.

    Returns:
        tuple: A tuple containing
            | **TTP (nunit):** Spike width of each unit. [us]
            | **unit_labels (nunit):** Label of each unit. 0: low spike width (inhibitory), 1: high spike width (excitatory)
            | **avg_wfs (nunit, nt):** Average waveform of accepted waveforms for each unit
            | **sss_unitid (1D):*** Unit index of spikes with a lower number of spikes than allowed by 'min_wfs'
    '''
    TTP = []
    sss_unitid = []

    # Get data size parameters.
    nt, _ = waveform_data[0].shape
    nunits = len(waveform_data)

    # Initialize array for average waveforms
    avg_wfs = np.zeros((nt, nunits))

    # Iterate through all units
    for iunit in range(nunits):
        iwfdata = waveform_data[iunit]  # shape (nt, nunit) - waveforms for each unit

        # Use PCA and kmeans to remove outliers if there are enough data points
        if iwfdata.shape[1] >= min_wfs:
            # Use each time point as a feature and each spike as a sample.
            _, _, iwfdata_proj = get_pca_dimensions(iwfdata.T, max_dims=None, VAF=pca_varthresh, project_data=True)
            good_wf_idx, _ = find_outliers(iwfdata_proj, std_threshold)
        else:
            good_wf_idx = np.arange(iwfdata.shape[1])
            sss_unitid.append(iunit)

        iwfdata_good = iwfdata[:, good_wf_idx]

        # Average good waveforms
        iwfdata_good_avg = np.mean(iwfdata_good, axis=1)
        avg_wfs[:, iunit] = iwfdata_good_avg

        # Calculate 1st order TTP approximation
        troughidx_1st, peakidx_1st = find_trough_peak_idx(iwfdata_good_avg)

        # Interpolate peaks with a parabolic fit
        troughidx_2nd, _, _ = interpolate_extremum_poly2(troughidx_1st, iwfdata_good_avg, extrap_peaks=False)
        peakidx_2nd, _, _ = interpolate_extremum_poly2(peakidx_1st, iwfdata_good_avg, extrap_peaks=False)

        # Calculate 2nd order TTP approximation
        TTP.append(1e6 * (peakidx_2nd - troughidx_2nd) / samplerate)

    gmm_proc = GaussianMixture(n_components=2, random_state=0).fit(np.array(TTP).reshape(-1, 1))
    unit_labels = gmm_proc.predict(np.array(TTP).reshape(-1, 1))

    # Ensure lowest TTP unit is inhibitory (0)
    minttpidx = np.argmin(TTP)
    if unit_labels[minttpidx] == 1:
        unit_labels = 1 - unit_labels

    return TTP, unit_labels, avg_wfs, sss_unitid


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
            | **troughidx (nch):** Array of indices corresponding to the trough time for each channel
            | **peakidx (nch):** Array of indices corresponding ot the peak time for each channel. 
    '''
    # Handle condition where the input data is a 1D array
    if len(unit_data.shape) == 1:
        troughidx = np.argmin(unit_data)

        wfdecreaseidx = np.where(np.diff(unit_data[troughidx:]) < 0)

        if np.size(wfdecreaseidx) == 0:
            peakidx = len(unit_data) - 1
        else:
            peakidx = np.min(wfdecreaseidx) + troughidx

    # Handle 2D input data array  
    else:
        troughidx = np.argmin(unit_data, axis=0)
        peakidx = np.empty(troughidx.shape)

        for trialidx in range(len(peakidx)):

            wfdecreaseidx = np.where(np.diff(unit_data[troughidx[trialidx]:, trialidx]) < 0)

            # Handle the condition where there is no negative derivative.
            if np.size(wfdecreaseidx) == 0:
                peakidx[trialidx] = len(unit_data[:, trialidx]) - 1
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
            | **extremum_time (float):** Interpolated (or extrapolated) peak time        
            | **extremum_value (float):** Approximated peak value.        
            | **f (np.poly):** Polynomial used to calculate peak time
    '''

    # Handle condition where the peak is at the beginning of a dataset
    if extremum_idx == 0:
        edge_idx = True
        xpts = np.arange((extremum_idx), extremum_idx + 3, 1)
        ypts = data[extremum_idx:extremum_idx + 3]

    # Handle condition where the peak is at the end of a dataset
    elif extremum_idx == len(data) - 1:
        edge_idx = True
        xpts = np.arange((extremum_idx - 2), extremum_idx + 1, 1)
        ypts = data[extremum_idx - 2:extremum_idx + 1]

    # Condition where the peak is in the middle of the dataset
    else:
        edge_idx = False
        xpts = np.arange((extremum_idx - 1), extremum_idx + 2, 1)
        ypts = data[extremum_idx - 1:extremum_idx + 2]

    f = interpolate.lagrange(xpts, ypts)
    extremum_time = -f[1] / (2 * f[2])
    extremum_value = (f[2] * (extremum_time ** 2)) + (f[1] * extremum_time) + f[0]

    # If end points should not be extrapolated from...
    if extrap_peaks == False and edge_idx:
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
            | **unit_mean:** The mean spike counts for each unit across the input time
            | **unit_variance:** The spike count variance for each unit across the input time
    '''

    counts = np.sum(spiking_data, axis=1)  # Counts has the shape (nunits, ntr)
    unit_mean = np.mean(counts, axis=1)  # Averge the counts for each unit across all trials
    unit_variance = np.var(counts, axis=1)  # Calculate the count variance for each unit across all trials

    return unit_mean, unit_variance


'''
KALMAN FILTER 
'''


class KFDecoder(object):
    """
    Class for the Kalman Filter Decoder
    Parameters
    -----------
    C - float, optional, default 1
    This parameter scales the noise matrix associated with the transition in kinematic states.
    It effectively allows changing the weight of the new neural evidence in the current update.
    Our implementation of the Kalman filter for neural decoding is based on that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
    with the exception of the addition of the parameter C.
    The original implementation has previously been coded in Matlab by Dan Morris (http://dmorris.net/projects/neural_decoding.html#code)
    """

    def __init__(self, C=1):
        self.C = C

    def fit(self, X_kf_train, y_train):
        """
        Train Kalman Filter Decoder
        Parameters
        ----------
        X_kf_train (2D numpy array): [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.
            See example file for an example of how to format the neural data correctly
        y_train (2D numpy array): [n_samples(i.e. timebins), n_outputs]
            This is the outputs that are being predicted

        Calculations for A,W,H,Q are as follows:
        .. math:: A = X2*X1' (X1*X1')^{-1}
        .. math:: W = \frac{(X_2 - A*X_1)(X_2 - A*X_1)'}{(timepoints - 1)}
        .. math:: H = Y*X'(X*X')^{-1}
        .. math:: Q = \frac{(Y-HX)(Y-HX)' }{time points}
        """

        # Renaming and reformatting variables to be in a more standard kalman filter nomenclature (from Wu et al, 2003):
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_train.T)
        Z = np.matrix(X_kf_train.T)

        # number of time bins
        nt = X.shape[1]

        # Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        # In our case, this is the transition from one kinematic state to the next
        X2 = X[:, 1:]
        X1 = X[:, 0:nt - 1]

        A = X2 * X1.T * inv(X1 * X1.T)  # Transition matrix
        W = (X2 - A * X1) * (X2 - A * X1).T / (
                nt - 1) / self.C  # Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        # Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        # In our case, this is the transformation from kinematics to spikes
        H = Z * X.T * (inv(X * X.T))  # Measurement matrix
        Q = ((Z - H * X) * ((Z - H * X).T)) / nt  # Covariance of measurement matrix

        params = [A, W, H, Q]
        print('Shape of State Transition model (A) :' + str(A.shape))
        print('Shape of Covariance of State Transition model :' + str(W.shape))
        print('Shape of Observation model (H) :' + str(H.shape))
        print('Shape of Covariance of Observation model :' + str(Q.shape))
        self.model = params

    def fit_awf(self, X_kf_train, y_train, A, W):
        """
        Train Kalman Filter Decoder with A and W fixed. A is the state transition model and W is the associated covariance

        Parameters
        ----------
        X_kf_train (2D numpy array): [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.
            See example file for an example of how to format the neural data correctly
        y_train (2D numpy array): [n_samples(i.e. timebins), n_outputs]
            This is the outputs that are being predicted

        Calculations as follows:
        .. math:: H = Y*X'(X*X')^{-1}
        .. math:: Q = \frac{(Y-HX)(Y-HX)' }{time points}
        """

        # Renaming and reformatting variables to be in a more standard kalman filter nomenclature (from Wu et al, 2003):
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_train.T)
        Z = np.matrix(X_kf_train.T)

        # number of time bins
        nt = X.shape[1]

        # Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        # In our case, this is the transition from one kinematic state to the next
        X2 = X[:, 1:]
        X1 = X[:, 0:nt - 1]

        # A=X2*X1.T*inv(X1*X1.T) #Transition matrix
        # W=(X2-A*X1)*(X2-A*X1).T/(nt-1)/self.C #Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        # Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        # In our case, this is the transformation from kinematics to spikes
        H = Z * X.T * (inv(X * X.T))  # Measurement matrix
        Q = ((Z - H * X) * ((Z - H * X).T)) / nt  # Covariance of measurement matrix

        print('Shape of State Transition model (A) :' + str(A.shape))
        print('Shape of Covariance of State Transition model :' + str(W.shape))
        print('Shape of Observation model (H) :' + str(H.shape))
        print('Shape of Covariance of Observation model :' + str(Q.shape))
        params = [A, W, H, Q]
        self.model = params

    def predict(self, X_kf_test, y_test):
        """
        Predict outcomes using trained Kalman Filter Decoder
        Parameters
        ----------
        X_kf_test (2D numpy array):  [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.
        y_test (2D numpy array): [n_samples(i.e. timebins),n_outputs]
            The actual outputs
            This parameter is necesary for the Kalman filter (unlike other decoders)
            because the first value is nececessary for initialization
        Returns
        -------
        y_test_predicted (2D numpy array):  [n_samples(i.e. timebins),n_outputs]
            The predicted outputs
        """

        # Extract parameters
        A, W, H, Q = self.model

        # First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (I am following Wu et al):
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_test.T)
        Z = np.matrix(X_kf_test.T)

        # Initializations
        num_states = X.shape[0]  # Dimensionality of the state
        states = np.empty(
            X.shape)  # Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m = np.matrix(np.zeros([num_states, num_states]))  # This is a priori estimate of X covariance
        P = np.matrix(np.zeros([num_states, num_states]))  # This is a posteriori estimate of X covariance
        state = X[:, 0]  # Initial state
        states[:, 0] = np.copy(np.squeeze(state))

        # Get predicted state for every time bin
        for t in range(X.shape[1] - 1):
            # Do first part of state update - based on transition matrix
            P_m = A * P * A.T + W  # a priori estimate of x covariance ( P(k) = A*P(k-1)*A' + W )
            state_m = A * state  # a priori estimate of x ( X(k|k-1) = A*X(k-1) )

            # Do second part of state update - based on measurement matrix
            K = P_m * H.T * inv(H * P_m * H.T + Q)  # Calculate Kalman gain ( K = P_ap*H'* inv(H*P_ap*H' + Q) )
            P = (np.matrix(np.eye(num_states)) - K * H) * P_m  # (a posteriori estimate, P (I - K*H)*P_ap )
            state = state_m + K * (Z[:,
                                   t + 1] - H * state_m)  # compute a posteriori estimate of x (X(k) = X(k|k-1) + K*(Z - H*X(k|k-1))
            states[:, t + 1] = np.squeeze(state)  # Record state at the timestep
        y_test_predicted = states.T
        return y_test_predicted


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
        float array: rms of the signal along the first axis. output dimensions will 
            be the same non-time dimensions as the input signal
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
    kmeans_model = KMeans(n_clusters=1).fit(data)
    distances = kmeans_model.transform(data)
    cluster_labels = kmeans_model.labels_
    dist_std = np.sqrt(np.sum(distances ** 2) / len(distances))
    good_data_idx = (distances < (dist_std * std_threshold))

    return good_data_idx.flatten(), distances.flatten()


def fit_linear_regression(X: np.ndarray, Y: np.ndarray, coefficient_coeff_warning_level: float = 0.5) -> np.ndarray:
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

        x = X[:, i]
        y = Y[:, i]

        slope[i], intercept[i], corr_coeff[i], *_ = scipy.stats.linregress(x, y)

        if corr_coeff[i] <= coefficient_coeff_warning_level:
            warnings.warn(
                f'when fitting column number {i}, the correlation coefficient is {corr_coeff[i]}, less than {coefficient_coeff_warning_level} ')

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
    freq = np.fft.fftfreq(length, d=1. / samplerate)
    data_ampl = abs(freq_data[freq >= 0, :]) * 2 / length  # compute the one-sided amplitude
    non_negative_freq = freq[freq >= 0]

    # Apply factor of root 2 to turn amplitude into RMS amplitude
    if rms:
        data_ampl[1:, :] = data_ampl[1:, :] / np.sqrt(2)
    return non_negative_freq, data_ampl


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
    SEM = np.nanstd(data, axis=axis) / np.sqrt(n)

    return SEM


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
    ydata = ydata.reshape(-1, 1)

    reg_fit = LinearRegression(fit_intercept=fit_intercept).fit(xdata, ydata, sample_weight=weights)
    linear_fit_score = reg_fit.score(xdata, ydata)
    pcc_all = stats.pearsonr(xdata.flatten(), ydata.flatten())

    linear_fit = reg_fit.coef_[0][0] * xdata.flatten() + reg_fit.intercept_

    return linear_fit, linear_fit_score, pcc_all[0], pcc_all[1], reg_fit


def get_eye_trajectories_by_trial(
        eye_data, exp_data,
        start_events=[TARGET_ON_CODES], end_events=[TRIAL_END],
        eye_sample_rate=25000
):
    '''
    Finds eye trajectories in monitor space by trial.
    Args:
        eye_data (dict) eye data after eye calibration:
        exp_data (dict) preprocessed experimental data:
        start_events (list(int)) event codes to mark start of each trial:
        end_events (list(int)) event codes to mark end of each trial:
        eye_sample_rate (int)

    Returns:
        tuple: tuple containing:
            | **eye_data_by_trial (list of list of position):**  trajectories of eye movement in monitor space by trial
            | **trial_segments (list of list of events):** a segment of each trial
            | **times (ntrials, 2):** list of 2 timestamps for each trial corresponding to the start and end events
    '''
    events = exp_data['events']
    clock = exp_data['clock']
    event_times = clock['timestamp_sync'][events['time']]

    eye_calibed = eye_data["calibrated_data"]
    # get all segments from peripheral target on -> trial end
    trial_segments, trial_times_bmi3d = preproc.get_trial_segments(events['code'], event_times, start_events,
                                                                   end_events)
    # grab eye trajectories
    eye_data_by_trial = []
    for trial_start, trial_end in trial_times_bmi3d:
        # grab list of eye positions
        eye_index_start = (trial_start * eye_sample_rate).astype(int)
        eye_index_end = (trial_end * eye_sample_rate).astype(int)
        trial_eye_calibed = eye_calibed[eye_index_start:eye_index_end, :]
        eye_data_by_trial.append(trial_eye_calibed)
    return eye_data_by_trial, trial_segments, trial_times_bmi3d


def get_cursor_trajectories_by_trial(exp_data, start_events=[TARGET_ON_CODES], end_events=[TRIAL_END]):
    '''
    Finds cursor trajectories by trial.
    Args:
        exp_data (dict) preprocessed experimental data:
        start_events (list(int)) event codes to mark start of each trial:
        end_events (list(int)) event codes to mark end of each trial:
        eye_sample_rate (int)

    Returns:
        tuple: tuple containing:
            | **cursor_data_by_trial (list of list of position):**  trajectories of cursor for each trial
            | **trial_segments (list of list of events):** a segment of each trial
            | **times (ntrials, 2):** list of 2 timestamps for each trial corresponding to the start and end events
    '''
    # grab cursor trajectories
    # Find cursor data
    events = exp_data['events']

    cursor_data = exp_data['task']['cursor'][:, [0, 2]]
    event_cycles = events['time']
    trial_segments, trial_cycles = preproc.get_trial_segments(events['code'], event_cycles, start_events, end_events)
    cursor_data_by_trial = []
    for trial_start, trial_end in trial_cycles:
        # grab list of eye positions
        trial_cursor_pos = cursor_data[trial_start:trial_end, :]
        cursor_data_by_trial.append(trial_cursor_pos)
    return cursor_data_by_trial, trial_segments, trial_cycles


def get_target_positions(exp_data):
    # Preprocessing to get target positions
    target_pos_by_idx = np.empty([9, 3], dtype=object)
    for trial in exp_data['trials']:
        target_pos_by_idx[trial["index"], :] = trial["target"]
    target_pos_by_idx = target_pos_by_idx[:, [0, 2]]
    return target_pos_by_idx


def get_dist_to_targets(eye_data, exp_data, start_events=[TARGET_ON_CODES], end_events=[TRIAL_END],
                        eye_sample_rate=25000):
    '''
    Given eye and experimental data, grab trials where the cursor reaches the peripheral target
    for these trials, calculate the eye and cursor trajectories' distance to peripheral targets
    Args:
        eye_data (dict) eye data after eye calibration:
        exp_data (dict) preprocessed experimental data:
        start_events (list(int)) event codes to mark start of each trial:
        end_events (list(int)) event codes to mark end of each trial:
        eye_sample_rate (int)

    Returns:
        tuple: tuple containing:
            | **dist_eye_target (list of list of distances):**  distances of eye position to peripheral target for each trial
            | **dist_cursor_target (list of list of distances):** distances of cursor position to peripheral target for each trial
    '''

    eye_data_by_trial, _, _ = get_eye_trajectories_by_trial(eye_data, exp_data, start_events, end_events,
                                                            eye_sample_rate)
    cursor_data_by_trial, trial_segments, _ = get_cursor_trajectories_by_trial(exp_data, start_events, end_events)

    target_pos_by_idx = get_target_positions(exp_data)

    # Grab all successful trials in session
    success_indices = [i for i, t in enumerate(trial_segments) if SUCCESS_CODE in t]

    # segment indexes always start with peripheral TARGET_ON
    # use CENTER_ON code to calculate
    # the peripheral target index for each trial.
    target_indices = [t[0] - CENTER_ON for t in trial_segments]

    # Grab out data corresponding to successful trials from each list
    success_eye_pos = [eye_data_by_trial[i] for i in success_indices]
    success_cursor_pos = [cursor_data_by_trial[i] for i in success_indices]
    target_pos = [target_pos_by_idx[target_indices[i]] for i in success_indices]

    dist_eye_target = []
    for i, eye_pos in enumerate(success_eye_pos):
        dist = np.sqrt((eye_pos[:, 0] - target_pos[i][0]) ** 2 + (eye_pos[:, 1] - target_pos[i][1]) ** 2)
        dist_eye_target.append(dist)

    dist_cursor_target = []
    for i, cursor_pos in enumerate(success_cursor_pos):
        dist = np.sqrt((cursor_pos[:, 0] - target_pos[i][0]) ** 2 + (cursor_pos[:, 1] - target_pos[i][1]) ** 2)
        dist_cursor_target.append(dist)
    return dist_eye_target, dist_cursor_target


def get_movement_error_var_for_session(exp_data, start_codes=[TARGET_ON_CODES], end_codes=[CURSOR_ENTER_TARGET_CODES]):
    events = exp_data['events']

    # grab cursor trajectories
    # Find cursor data
    cursor_data = exp_data['task']['cursor'][:, [0, 2]]
    event_cycles = events['time']

    trial_segments, trial_cycles = preproc.get_trial_segments(events['code'], event_cycles, start_codes, end_codes)

    # Preprocessing to get target positions
    target_pos_by_idx = np.empty([9, 3], dtype=object)
    for trial in exp_data['bmi3d_trials']:
        target_pos_by_idx[trial["index"], :] = trial["target"]
    target_pos_by_idx = target_pos_by_idx[:, [0, 2]]

    origin_pos = target_pos_by_idx[0]

    movement_error_by_trial = []
    movement_var_by_trial = []
    for events, times in zip(trial_segments, trial_cycles):
        start_time, end_time = times
        target = events[0] - 16
        target_pos = target_pos_by_idx[target]
        trial_cursor_pos = cursor_data[start_time:end_time, :]
        perp_dists = np.array([
            np.abs(np.cross(target_pos - origin_pos, origin_pos - pos)) / np.linalg.norm(target_pos - origin_pos)
            for pos in trial_cursor_pos
        ])
        error = np.sqrt(np.sum(np.absolute(perp_dists))) / perp_dists.shape[0]
        avg = np.average(perp_dists)
        var = np.sqrt(np.sum((perp_dists - avg) ** 2) / perp_dists.shape[0])
        movement_error_by_trial.append(error)
        movement_var_by_trial.append(var)
    return movement_error_by_trial, movement_var_by_trial


def get_time_to_target_for_session(exp_data, start_codes=[TARGET_ON_CODES], end_codes=[CURSOR_ENTER_TARGET_CODES]):
    events = exp_data['events']
    clock = exp_data['clock']

    # grab cursor trajectories
    # Find cursor data
    event_cycles = events['time']

    trial_segments, trial_cycles = preproc.get_trial_segments(events['code'], event_cycles, start_codes, end_codes)

    time_to_targets = []
    for start_time, end_time in trial_cycles:
        timer_start = clock['timestamp_sync'][start_time]
        timer_end = clock['timestamp_sync'][end_time]
        time_to_targets.append(timer_end - timer_start)
    return time_to_targets
