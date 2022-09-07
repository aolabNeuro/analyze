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
from numpy.linalg import inv as inv # used in Kalman Filter
import nitime.algorithms as tsa
from . import preproc

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
        iwfdata = waveform_data[iunit] # shape (nt, nunit) - waveforms for each unit
        
        # Use PCA and kmeans to remove outliers if there are enough data points
        if iwfdata.shape[1] >= min_wfs:
            # Use each time point as a feature and each spike as a sample.
            _, _, iwfdata_proj = get_pca_dimensions(iwfdata.T, max_dims=None, VAF=pca_varthresh, project_data=True)
            good_wf_idx, _ = find_outliers(iwfdata_proj, std_threshold)
        else:
            good_wf_idx = np.arange(iwfdata.shape[1])
            sss_unitid.append(iunit)
            
        iwfdata_good = iwfdata[:,good_wf_idx]

        # Average good waveforms
        iwfdata_good_avg = np.mean(iwfdata_good, axis = 1)    
        avg_wfs[:,iunit] = iwfdata_good_avg

        # Calculate 1st order TTP approximation
        troughidx_1st, peakidx_1st = find_trough_peak_idx(iwfdata_good_avg)

        # Interpolate peaks with a parabolic fit
        troughidx_2nd, _, _  = interpolate_extremum_poly2(troughidx_1st, iwfdata_good_avg, extrap_peaks=False)
        peakidx_2nd, _, _ = interpolate_extremum_poly2(peakidx_1st, iwfdata_good_avg, extrap_peaks=False)

        # Calculate 2nd order TTP approximation
        TTP.append(1e6*(peakidx_2nd - troughidx_2nd)/samplerate)    
    
    gmm_proc = GaussianMixture(n_components = 2, random_state = 0).fit(np.array(TTP).reshape(-1, 1))
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

    counts = np.sum(spiking_data, axis=1) # Counts has the shape (nunits, ntr)
    unit_mean = np.mean(counts, axis=1) # Averge the counts for each unit across all trials
    unit_variance = np.var(counts, axis=1) # Calculate the count variance for each unit across all trials

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
            state = state_m + K * (Z[:,t + 1] - H * state_m)  # compute a posteriori estimate of x (X(k) = X(k|k-1) + K*(Z - H*X(k|k-1))
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
        plt.show()

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

def calc_max_erp(data, event_times, time_before, time_after, samplerate, subtract_baseline=True, baseline_window=None, max_search_window=None):
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

    Returns:
        nch: array of maximum mean-ERP for each channel during the given time periods

    '''
    mean_erp = np.mean(calc_erp(data, event_times, time_before, time_after, samplerate, subtract_baseline, baseline_window), axis=0)

    # Limit the search to the given window
    start_idx = int(time_before*samplerate)
    end_idx = start_idx + int(time_after*samplerate)
    if max_search_window:
        if len(max_search_window) < 2 or max_search_window[1] < max_search_window[0]:
            raise ValueError("max_search_window must be in the form (t0, t1) where \
                t1 is greater than t0")
        end_idx = start_idx + int(max_search_window[1]*samplerate)
        start_idx += int(max_search_window[0]*samplerate)
    mean_erp_window = mean_erp[start_idx:end_idx,:]

    # Find the index that maximizes the absolute value, then use that index to get the actual signed value
    idx_max_erp = start_idx + np.argmax(np.abs(mean_erp_window), axis=0)
    max_erp = np.array([mean_erp[idx_max_erp[i],i] for i in range(mean_erp.shape[1])])

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


def calc_activity_onset_accLLR(data, altcond, nullcond, modality, bin_width, thresh_proportion=0.15, max_accLLR=None, trial_average=True):
    '''
    This function calculates the accumulated log-likelihood ratio (AccLLR) that the input data matches the timeseries input as condition 1 or condition 2. 
    This approach was designed to work on a trial-by-trial basis and determine if the input data matches a condition and arrival time of the difference. Therefore, AccLLR can also be used to determine the arrival time of neural activity.
    Positive values of AccLLR correspond to the data better matching condition 1. Negative values of AccLLR correspond to the data better matching condition 2.
    AccLLR for spikes assumes Poisson firing statistics, and AccLLR for LFP assumes Gaussian activity. Based on Banerjee et al. 2010, more paper informtion listed below.
    Increasing the proportion of AccLLR maximum used for classification results in more conservative and accurate classification results
    but leads to longer delays in seletion time and more trials that are unclassfied. 
    If multiple trials are input, use the trial averaging approach outlined in the paper where the LLR is calculated and summed at each time point accross trials.
    
    Since spikes assume Poisson firing statistics, input spiking data must be binary. However, the null and alternative condition must be float arrays. 
    
    Banerjee A, Dean HL, Pesaran B. A likelihood method for computing selection times in spiking and local field potential activity. J Neurophysiol. 2010 Dec;104(6):3705-20. doi: 10.1152/jn.00036.2010. Epub 2010 Sep 8.
    https://pubmed.ncbi.nlm.nih.gov/20884767/

    If modality is 'lfp' use the following log-likelihood ratio equation. (Equation (3) in the paper):
    
    .. math::
    
        LL(t)=log \\frac{P[x(t)-\mu_1(t)|\sigma^2]}{P[x(t)-\mu_2(t)|\sigma^2]}
        
    If modality is 'spikes' use the following log-likelihood ratio equation. (Equation (5) in the paper):
    
    .. math::
    
        LL(t)=log \\frac{P[dN(t))|\lambda_1(t)]}{P[dN(t))|\lambda_2(t)]}
    
    Args:
        data (npts, ntrial): Input data to compare to each condition
        altcond (npts): neural activity series of interest (condition 1)
        nullcond (npts): neural activity during a baseline period for comparison (condition 2)
        modality (str): Either 'lfp' or 'spikes'. 'lfp' uses a Gaussian distribution assumption and 'spikes' uses a Poisson distribution assumption
        bin_width (float): Bin width of input activity
        thresh_proportion (float): Proportion of maximum AccLLR where the threshold is set to classify trials as condition 1 or condition 2. 
        max_accLLR (float): 
        trial_average (bool, optional): Flag to do a trial average calculation (default) or just a single trial at a time

    Returns: 
        If trial_average, a tuple containing:
            | **accLLR (npts, ntrials):** AccLLR time series
            | **selection_time (ntrials):** Time where AccLLR crosses the threshold

        Otherwise, a tuple containing:
            | **accLLR (npts, ntr):** AccLLR time series on each trial
            | **selection_time (ntr):** Time where AccLLR crosses the threshold on each trial
    '''
    # Ensure data is 2D
    if len(data.shape) == 1:
        data = data[:,None]
    npts = data.shape[0]
    ntrials = data.shape[1]
    if trial_average:
        LLR = np.zeros(npts)*np.nan # LLR at exact time points
    else:
        LLR = np.zeros((npts, ntrials))*np.nan
    
    if modality == 'spikes':
        # If modality is spikes, ensure data is binary (all 0's and 1's)
        binary_spike_mask = np.logical_and(~np.isclose(data, 0), ~np.isclose(data,1))
        if np.sum(binary_spike_mask) > 0:
            warnings.warn('Input spiking activity is not binary (all 0s and 1s)')

        # Calculate AccLLR across all trials at each point
        for ipt in range(npts):
            temp_LLR = np.zeros(ntrials)*np.nan
            for itrial in range(ntrials):
                temp_LLR[itrial] = (nullcond[ipt] - altcond[ipt])*bin_width + data[ipt, itrial]*np.log(altcond[ipt]/nullcond[ipt])

            if trial_average:
                LLR[ipt] = np.nansum(temp_LLR)
            else:
                LLR[ipt,:] = temp_LLR
        
    
    elif modality == 'lfp':
    
        # Calculate AccLLR parameters
        sigma_sq = np.var(data, axis=0)

        # Calculate AccLLR across all trials at each point
        for ipt in range(npts):
            temp_LLR = np.zeros(ntrials)*np.nan
            for itrial in range(ntrials):                
                temp_LLR[itrial] = (((data[ipt, itrial]-nullcond[ipt])**2) - ((data[ipt, itrial]-altcond[ipt])**2))/(2*sigma_sq[itrial])
            
            if trial_average:
                LLR[ipt] = np.nansum(temp_LLR)
            else:
                LLR[ipt,:] = temp_LLR

    else:
        warnings.warn('Please input a valid modality')
    
    
    accLLR = np.nancumsum(LLR, axis=0)

    if max_accLLR is None:
        max_accLLR = np.max(np.abs(accLLR), axis=0)

    thresh_val = thresh_proportion*max_accLLR
    above_thresh = np.abs(accLLR) > thresh_val

    if trial_average:
        selection_time = np.nan
        if above_thresh.any():
            selection_time = np.where(above_thresh)[0][0]*bin_width

    else:
        selection_time = np.zeros(ntrials)*np.nan
        for tr_idx in range(ntrials):
            if above_thresh[:,tr_idx].any():
                selection_time[tr_idx] = np.where(above_thresh[:,tr_idx])[0][0]*bin_width

    return accLLR, selection_time

def calc_accLLR_threshold(altcond_train, nullcond_train, altcond_test, nullcond_test, modality, bin_width, thresh_step_size=0.01, false_alarm_prob=0.05):
    '''
    Sweeps the AccLLR method over the thresh_proportion parameter, estimates false alarm rates, 
    and then choose a value for thresh_proportion that gives us the desired false alarm rate.
    
    Estimate the false alarm probability in the accLLR method by the proportion of 
    trials from condition 2 whose AccLLR hit the upper detection threshold within the maximum 
    accumulation time. 

    Banerjee A, Dean HL, Pesaran B. A likelihood method for computing selection times in spiking and local field potential activity. J Neurophysiol. 2010 Dec;104(6):3705-20. doi: 10.1152/jn.00036.2010. Epub 2010 Sep 8.
    https://pubmed.ncbi.nlm.nih.gov/20884767/

    Args:
        altcond_train (npts): training timeseries from condition 1
        nullcond_train (npts): training timeseries from condition 2
        nullcond_test (npts, ntr): test trials from condition 2
        modality (str): Either 'lfp' or 'spikes'. 'lfp' uses a Gaussian distribution assumption and 'spikes' uses a Poisson distribution assumption
        bin_width (float): Bin width of input activity
        thresh_step_size (float, optional): Size of the steps in the sweep of the threshold proportion parameter. Defaults to 0.01.
        false_alarm_prob (float, optional): Desired false alarm probability. Defaults to 0.05.

    Returns:
        (tuple): Tuple containing:
            | **best_tp (float):** threshold proportion that yields the desired false alarm probability, or np.nan
            | **thresh_props (nsteps):** threshold proportions used for each false alarm rate calculation
            | **fa_rates (nsteps):** false alarm rates at each threshold proportion
    '''
    npts = altcond_train.shape[0]
    ntrials = nullcond_test.shape[1]

    # Get max accLLR value for threshold calculation by calculating the max accLLR with the training data
    accLLR, _ = calc_activity_onset_accLLR(altcond_test, altcond_train, nullcond_train, modality=modality, bin_width=bin_width)
    max_accLLR = np.max(accLLR)
    
    thresh_props = np.arange(thresh_step_size, 1, thresh_step_size)
    fa_rates = []    
    for tp in thresh_props:
        accLLR, selection_time_idx = calc_activity_onset_accLLR(nullcond_test, altcond_train, nullcond_train, modality, bin_width, thresh_proportion=tp, max_accLLR=max_accLLR, trial_average=False) 
        accLLR_altcond_within_time = np.sum(np.logical_and(accLLR > 0, selection_time_idx < npts), axis=0)
        n_accllr_altcond_within_time = np.count_nonzero(accLLR_altcond_within_time)
        false_alarms = n_accllr_altcond_within_time / ntrials
        fa_rates.append(false_alarms)
    fa_rates = np.array(fa_rates)
    fa_rates_above_desired = thresh_props[fa_rates > false_alarm_prob]
    best_tp = np.nan
    if any(fa_rates_above_desired):
        best_tp = fa_rates_above_desired[0]
    return best_tp, thresh_props, fa_rates


def accLLR_wrapper(data_altcond, data_nullcond, modality, bin_width, train_prop_input=0.7, thresh_step_size=0.01, false_alarm_prob=0.05, trial_average=True, match_selectivity=True):
    '''
    See the function analysis.calc_activity_onset_accLLR for specifics about the computation performed in this wrapper. This function is intended to compare a time-series of alternative and null conditions
    to determine when the alternative condition deviates from the null condition. This function assumes it is known that the alternative condition deviates from the null condition. It is recommended that 
    other methods are implemented to determine if a candidate alternative condition dataset is significantly different from the null condition dataset (i.e. t-test).

    Computations performed in this wrapper:
    1. Separates data into 'model building' (training + validate) and testing groups by trials 
        Assumes model building dataset is split into a 70/30 split
    2. Calculates accLLR threshold on training and test data (analysis.calc_accLLR_threshold)
    3. (optional) Match signal selectivity. Has only been implemented for lfp data. (analysis.match_selectivity_accLLR)
    4. Implements accLLR on test data. (analysis.calc_activity_onset_accLLR)
    
    Example data allocation breakdown by trial (train_prop_input = 0.7)
    40% is used to train threshold proportion (train)
    30% is used to validate threshold proportion (test)
    30% is used to calculate selection time (validate)

    Selection time is max time if alternative condition is not reached.


    Banerjee A, Dean HL, Pesaran B. A likelihood method for computing selection times in spiking and local field potential activity. J Neurophysiol. 2010 Dec;104(6):3705-20. doi: 10.1152/jn.00036.2010. Epub 2010 Sep 8.
    https://pubmed.ncbi.nlm.nih.gov/20884767/

    Args:
        data_altcond (npts, nch, ntrials): Data from alternative condition.
        data_nullcond (npts, nch, ntrials): Data from null condition to compare the alternative condition to.
        modality (str): either 'spikes' or 'lfp'
        train_prop_input (float): proportion of trials to build the model with
        
    Returns:
        (tuple): Tuple containing:
            | **accllr_altcond (nt, nch, ntrials):** Time series of accllr for each channel. If trial_average=True the output will have the shape (nt, nch)
            | **selection_time_altcond (nch, ntrials):** Selection time of the alternative condition.  If trial_average=True the output will have the shape (nt, nch)
    '''
    nt = data_altcond.shape[0]
    nch = data_altcond.shape[1]
    ntrials = data_altcond.shape[2]

    # Split into model building (train, validate) and testing data sets (outputs (ntrial, npt) datasets)
    # test_data_altcond, build_data_altcond, test_data_nullcond, build_data_nullcond = model_selection.train_test_split(data_altcond.T, data_nullcond.T, test_size=train_prop_input)
    ntrain_trials = int(np.ceil(ntrials*train_prop_input*0.7))
    nvalid_trials = int(np.ceil(ntrials*train_prop_input*0.3))
    ntest_trials = ntrials - ntrain_trials - nvalid_trials

    trialidx = np.arange(ntrials)
    train_trialidx = np.random.choice(trialidx, size=ntrain_trials, replace=False)
    valid_trialidx = np.random.choice(trialidx[~np.in1d(trialidx, train_trialidx)], size=nvalid_trials, replace=False)
    test_trialidx = trialidx[~np.in1d(trialidx, np.concatenate((train_trialidx, valid_trialidx)))]

    # (nt, nch, ntrials)
    train_data_altcond = data_altcond[:,:,train_trialidx]
    train_data_nullcond = data_nullcond[:,:,train_trialidx]
    valid_data_altcond = data_altcond[:,:,valid_trialidx]
    valid_data_nullcond = data_nullcond[:,:,valid_trialidx]
    test_data_altcond = data_altcond[:,:,test_trialidx]
    test_data_nullcond = data_nullcond[:,:,test_trialidx]

    # Smooth spike events by convolving with a 5ms Gaussian
    if modality == 'spikes':
        gaus_sigma = 5
        time_axis = np.arange(-3*gaus_sigma, 3*gaus_sigma+1) # Constrain filter to +/-3std
        gaus_filter = (1/(gaus_sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*(time_axis**2)/(gaus_sigma**2))

        train_data_altcond = scipy.ndimage.convolve1d(train_data_altcond, gaus_filter, output=float, mode='reflect', axis=0)
        train_data_nullcond = scipy.ndimage.convolve1d(train_data_nullcond, gaus_filter, output=float, mode='reflect', axis=0)

    # Input train and validate data to calculate acclllr threshold (transpose data back to input into accllr functions)
    accllr_thresh = np.zeros(nch)*np.nan
    for ich in range(nch):
        accllr_thresh[ich], thresh_props, fa_rates = calc_accLLR_threshold(np.mean(train_data_altcond[:,ich,:], axis=1), np.mean(train_data_nullcond[:,ich,:], axis=1), valid_data_altcond[:,ich,:], valid_data_nullcond[:,ich,:], modality, bin_width, thresh_step_size=thresh_step_size, false_alarm_prob=false_alarm_prob)

    if match_selectivity:
        if modality == 'lfp':
            test_data_altcond = match_selectivity_accLLR(test_data_altcond, train_data_altcond, train_data_nullcond, modality, bin_width, accllr_thresh)
        else:
            print('Sorry! Matching signal selectivity for spiking data has not been implemented. Results will be without matching.')

    # Use the calculated threshold to run accllr on the test datasets
    if trial_average:
        selection_time_altcond = np.zeros((nch))*np.nan
        accllr_altcond = np.zeros((nt,nch))*np.nan
    else:
        selection_time_altcond = np.zeros((nch, ntest_trials))*np.nan
        accllr_altcond = np.zeros((nt,nch, ntest_trials))*np.nan

    for ich in range(nch):
        accllr_altcond_temp, selection_time_altcond_temp = calc_activity_onset_accLLR(test_data_altcond[:,ich,:], np.mean(train_data_altcond[:,ich,:], axis=1), np.mean(train_data_nullcond[:,ich,:], axis=1), modality, bin_width, thresh_proportion=accllr_thresh[ich], trial_average=trial_average)
        if trial_average:
            accllr_altcond[:,ich] = accllr_altcond_temp
            selection_time_altcond[ich] = selection_time_altcond_temp
        else:
            accllr_altcond[:,ich,:] = accllr_altcond_temp
            selection_time_altcond[ich,:] = selection_time_altcond_temp
        
    return selection_time_altcond, accllr_altcond


def match_selectivity_accLLR(test_data_altcond, train_data_altcond, train_data_nullcond, modality, bin_width, thresh_proportion):
    '''
    Calculates the ROC curve for each channel and adds noise to keep signal selectivity constant across all channel

    Args:
        test_data_altcond (npts, nch, ntrials):
        train_data_altcond (npts, nch, ntrials):
        modality (str): either 'spikes' or 'lfp'
        train_prop_input (float): proportion of trials to build the model with

    Returns:
        (npts, nch, ntrials): test_data_altcond with added noise to match selectivity across channels

    '''
    nt = test_data_altcond.shape[0]
    nch = test_data_altcond.shape[1]
    ntrials = test_data_altcond.shape[2]

    # Calculate choice probability at final time point to determine selectivity 
    choice_probability = np.zeros((nch))*np.nan
    for ich in range(nch):
        temp_accllr, _ = calc_activity_onset_accLLR(test_data_altcond[:,ich,:], np.mean(train_data_altcond[:,ich,:], axis=1), np.mean(train_data_nullcond[:,ich,:],axis=1), modality, bin_width, trial_average=False) #[nt, ntrial]
        choice_probability[ich] = np.sum(temp_accllr[-1,:] > 0)/ntrials # Find proportion of trials more selective for the alternative condition.


    # Find channel with the lowest choice probability to initialize
    match_ch_idx = np.zeros(nch, dtype=bool)
    match_ch_idx[choice_probability == np.min(choice_probability)] = True
    match_ch_prob = np.min(choice_probability)

    # If all signals are similarly selective, return input array
    if np.sum(match_ch_idx) == nch:
        return test_data_altcond

    # Match selectivity of all channels to the channel with the lowest selectivity
    if modality == 'spikes':

        timebin_size = 5 # Use a 5ms bin to manipulate spiking
        ntimebin = nt//timebin_size 
        noisy_test_data = test_data_altcond

        while np.sum(match_ch_idx) < nch:
            spike_move_prob_step = 0.01
            spike_move_prob = spike_move_prob_step

            for ich in range(nch):
                for ibin in range(ntimebin):
                    tstartidx = ibin*timebin_size
                    tstopidx = tstartidx + timebin_size
                    new_spikes = test_data_altcond[tstartidx:tstopidx,ich,:] #(ntimebin, ntrials)
                    spike_trials = np.sum(new_spikes, axis=0) > 0                

                    # use spike_move_prob to get the trials to delete a spike from
                    trials_to_delete_spikes = np.random.uniform(0,1,size=ntrials) < spike_move_prob # may not be necessary
                    trialidx_to_delete_spikes = np.where(trials_to_delete_spikes)[0]

                    # For each trial with a spike to delete, randomly select which spike to delete and add to a remaining trial
                    #TODO - not sure if it is correct to randomly select a spike to delete
                    #TODO - not sure if it is correct to remove all spikes and then add to the same array after
                    #TODO - not sure what to do if there are more spikes removed than available places to put them
                    if np.sum(trials_to_delete_spikes)>0:
                        for trialidx in trialidx_to_delete_spikes:
                            spike_bin_idx = np.where(new_spikes[:, trialidx] > 0)[0]
                            spike_bin_idx_remove = np.random.choice(spike_bin_idx)
                            new_spikes[spike_bin_idx_remove,trialidx] = 0 

                        # Find a remaining spike with followed by a quiet period of 3ms and add a spike 2ms after it
                        nspikes_placed = 0 # Counter for the number of spikes placed
                        for itrial in enumerate(ntrials):
                            if nspikes_placed < len(trialidx_to_delete_spikes):
                                # determine if there is a spike followed by a quiet period of 3ms in this trial
                                spike_idx = np.where(new_spikes[:,itrial])[0]
                                valid_first_spike = len(spike_idx) > 0 and np.max(spike_idx) < 2
                                if valid_first_spike:                            
                                    quiet_after_spike = new_spikes[np.max(spike_idx)+1,itrial] == 0 and new_spikes[np.max(spike_idx)+2,itrial] == 0 and new_spikes[np.max(spike_idx)+3,itrial] == 0
                                else:
                                    quiet_after_spike = False
                                
                                # If spike has 3ms quiet after, place spike
                                if valid_first_spike and quiet_after_spike:
                                    new_spikes[np.max(spike_idx)+2, itrial] = 1

                    noisy_test_data[tstartidx,tstopidx,ich,:] = new_spikes

            # Calculate choice probability 
            unmatch_choice_prob = np.zeros(len(unmatch_chs_idx))*np.nan
            for ich in range(len(unmatch_chs_idx)):
                temp_accllr, _ = calc_activity_onset_accLLR(noisy_test_data, train_data_altcond, train_data_nullcond, modality, bin_width, trial_average=False) #[nt, ntrial]
                unmatch_choice_prob[ich] = np.sum(temp_accllr[-1,:] > 0)/ntrials # Find proportion of trials more selective for the alternative condition.

            # Replace choice probabilities and get matched channels
            choice_probability[unmatch_chs_idx] = unmatch_choice_prob

            match_ch_idx = choice_probability <= match_ch_prob

            spike_move_prob += spike_move_prob_step

        return noisy_test_data

    elif modality == 'lfp':
        noise_sd_step = 1 # Standard deviation of noise - may want to pull this out
        noise_sd = noise_sd_step
        noisy_test_data = test_data_altcond
        while np.sum(match_ch_idx) < nch:
            # Add noise to non-matched channels
            unmatch_chs_idx = np.arange(nch) #unmatched channel idx
            unmatch_chs_idx = unmatch_chs_idx[~match_ch_idx]
            # print('unmatch_ch_idx', unmatch_chs_idx, noise_sd)
            noisy_test_data[:,unmatch_chs_idx,:] = test_data_altcond[:,unmatch_chs_idx,:] + np.random.normal(0,noise_sd, size=(nt,len(unmatch_chs_idx), ntrials))

            # Calculate choice probability 
            unmatch_choice_prob = np.zeros(len(unmatch_chs_idx))*np.nan
            for ich in range(len(unmatch_chs_idx)):
                temp_accllr, _ = calc_activity_onset_accLLR(noisy_test_data[:,ich,:], np.mean(train_data_altcond[:,ich,:],axis=1), np.mean(train_data_nullcond[:,ich,:], axis=1), modality, bin_width, trial_average=False) #[nt, ntrial]
                unmatch_choice_prob[ich] = np.sum(temp_accllr[-1,:] > 0)/ntrials # Find proportion of trials more selective for the alternative condition.

            # Replace choice probabilities and get matched channels
            choice_probability[unmatch_chs_idx] = unmatch_choice_prob

            match_ch_idx = choice_probability <= match_ch_prob

            noise_sd += noise_sd_step

        return noisy_test_data

######### Spectral Estimation and Analysis ############

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
        fxx (np.array): spectrogram frequency array (equal in length to win_t * fs // 2 + 1)
        txx (np.array): spectrogram time array (equal in length to (len(data)/fs - win_t)/step_t)
        Sxx (len(fxx) x len(txx) x nch): multitaper spectrogram estimate. Last dimension squeezed for 1-d inputs.
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
        x (n_sample x n_ch): input data array containing nan-valued missing entries

    Returns:
        x_interp (n_sample x n_ch): interpolated data, uses `numpy.interp` method.
    """
    nan_idx = np.isnan(x)
    ok_idx = ~nan_idx
    xp = ok_idx.ravel().nonzero()[0]
    fp = x[ok_idx]
    idx = nan_idx.ravel().nonzero()[0]
    x[nan_idx] = np.interp(idx,xp,fp)

    return x


'''
Behavioral metrics 
'''
def calc_success_percent(events, start_events=[b"TARGET_ON"], end_events=[b"REWARD", b"TRIAL_END"], success_events=b"REWARD", window_size=None):
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
        success_percent = success_per_window/window_size

    return success_percent

def calc_success_rate(events, event_times, start_events, end_events, success_events, window_size=None):
    '''
    Calculate the number of successful trials per second with a given trial start and end definition.

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
    trial_acq_time = times[:,1]-times[:,0]
    ntrials = times.shape[0]
    
    # Get % of successful trials per window 
    success_perc = calc_success_percent(events, start_events, end_events, success_events, window_size=window_size)
    
    # Determine rolling target acquisition time info 
    if window_size is None:
        nsuccess = success_perc*ntrials
        acq_time = np.sum(trial_acq_time)

    else:
        nsuccess = success_perc*window_size
        filter_array = np.ones(window_size)
        acq_time = signal.convolve(trial_acq_time, filter_array, mode='valid', method='direct')
    
    success_rate = nsuccess / acq_time

    return success_rate

def compute_path_length_per_trajectory(trajectory):
    '''
    This function calculates the path length by computing the distance from all points for a single trajectory. The input trajectry could be cursor or eye trajectory from a single trial. It returns a single value for path length.

    Args:
        trajectory (nt x 2): single trial trajectory, could be a cursor trajectory or eye trajectory

    Returns:
        path_length (float): length of the trajectory
    '''
    lengths = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)) # compute the distance from all points in trajectory
    path_length = np.sum(lengths)
    return path_length


def time_to_target(event_codes, event_times, target_codes=list(range(81, 89)) , go_cue_code=32 , reward_code=48):
    '''
    This function calculates reach time to target only on rewarded trials given trial aligned event codes and event times See: :func:`aopy.preproc.base.get_trial_segments_and_times` .

    Note:
        Trials are filtered to only include rewarded trials so that all trials have the same length.

    Args:
        event_codes (list) : trial aligned event codes
        event_times (list) : trial aligned event times corresponding to the event codes. These event codes and event times could be the output of preproc.base.get_trial_segments_and_times()
        target_codes (list) : list of event codes for cursor entering peripheral target 
        go_cue_code (int) : event code for go cue 
        reward_code (int) : event code for reward

    Returns:
      tuple: tuple containing:
        | **reachtime_pertarget (list)**: duration of each segment after filtering
        | **trial_id (list):** target index on each segment
    '''
    tr_T = np.array([event_times[iTr] for iTr in range(len(event_times)) if reward_code in event_codes[iTr]])
    tr_E = np.array([event_codes[iTr] for iTr in range(len(event_times)) if reward_code in event_codes[iTr]])
    leave_center_idx = np.argwhere(tr_E == go_cue_code)[0, 1]
    reach_target_idx = np.argwhere(np.isin(tr_E[0], target_codes))[0][0] # using just the first trial to get reach_target_idx
    reachtime = tr_T[:, reach_target_idx] - tr_T[:, leave_center_idx]
    target_dir = tr_E[:,reach_target_idx]

    return reachtime, target_dir

def calc_segment_duration(events, event_times, start_events, end_events, target_codes=list(range(81, 89)), trial_filter=lambda x:x):
    '''
    Calculates the duration of trial segments. Event codes and event times for this function are raw and not trial aligned.

    Args:
        events (nevents): events vector, can be codes, event names, anything to match
        event_times (nevents): time of events in 'events'
        start_events (int, str, or list, optional): set of start events to match
        end_events (int, str, or list, optional): set of end events to match
        target_codes (list, optional): list of target codes to use for finding targets within trials
        trial_filter (function, optional): function to apply to each trial's events to determine whether or not to keep it

    Returns:
        tuple: tuple containing:
        | **segment_duration (list)**: duration of each segment after filtering
        | **target_codes (list):** target index on each segment
    '''
    trial_events, trial_times = preproc.get_trial_segments(events, event_times, start_events, end_events)
    trial_events, trial_times = zip(*[(e, t) for e, t in zip(trial_events, trial_times) if trial_filter(e)])

    segment_duration = np.array([t[1] - t[0] for t in trial_times])
    target_idx = [np.argwhere(np.isin(te, target_codes))[0][0] for te in trial_events]
    target_codes = np.array([trial_events[trial_idx][idx] for trial_idx, idx in enumerate(target_idx)]) - np.min(target_codes)

    return segment_duration, target_codes
