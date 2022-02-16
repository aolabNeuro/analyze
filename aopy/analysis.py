# analysis.py
# Code for neural data analysis; functions here should return interpretable results such as
# firing rates, success rates, direction tuning, etc.

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn import model_selection
from scipy import interpolate
import warnings
from numpy.linalg import inv as inv # used in Kalman Filter

from . import preproc

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

def run_curvefitting(means, make_plot=True, fig_title='Tuning Curve', n_subplot_rows=None, n_subplot_cols= None):
    '''
    Args:
        means (2D array) : Mean firing rate [n_targets x n_neurons]
        make_plot (bool) : Generates plot with curve fitting and mean firing rate for n_neuons
        Fig title (str) : Figure title
        n_rows (int) : No of rows in subplot
        n_cols (int) : No. of cols in subplot

    Returns:
        tuple: Tuple containing:
            | **params_day (Numpy array):** Curve fitting parameters
            | **modulation depth (Numpy array):** Modulation depth of neuron
            | **preferred direction (Numpy array):** preferred direction of neurons
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

def calc_success_rate(events, start_events=[b"TARGET_ON"], end_events=[b"REWARD", b"TRIAL_END"], success_events=b"REWARD"):
    '''
    A wrapper around get_trial_segments which counts the number of trials with a reward event 
    and divides by the total number of trials to calculate success rate

    Args:
        events (nt): events vector, can be codes, event names, anything to match
        start_events (list, optional): set of start events to match
        end_events (list, optional): set of end events to match
        success_events (list, optional): which events make a trial a successful trial

    Returns:
        float: success rate = number of successful trials out of all trials
    '''
    segments, _ = preproc.get_trial_segments(events, np.arange(len(events)), start_events, end_events)
    n_trials = len(segments)
    success_trials = [np.any(np.isin(success_events, trial)) for trial in segments]
    n_success = np.count_nonzero(success_trials)
    success_rate = n_success / n_trials
    return success_rate

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

def get_pca_dimensions(data, max_dims=None, VAF=0.9, project_data=False):
    """
    Use PCA to estimate the dimensionality required to account for the variance in the given data. If requested it also projects the data onto those dimensions.
    
    Args:
        data (nt, nch): time series data where each channel is considered a 'feature' (nt=n_samples, nch=n_features)
        max_dims (int): (default None) the maximum number of dimensions
                        if left unset, will equal the dimensions (number of columns) in the dataset
        VAF (float): (default 0.9) variance accounted for (VAF)
        project_data (bool): (default False). If the function should project the high dimensional input data onto the calculated number of dimensions

    Returns:
        tuple: Tuple containing: 
            | **explained_variance (list):** variance accounted for by each principal component
            | **num_dims (int):** number of principal components required to account for variance
            | **projected_data (nt, ndims):** Data projected onto the dimensions required to explain the input variance fraction. If the input 'project_data=False', the function will return 'projected_data=None'
    """

    if max_dims is None:
        max_dims = np.shape(data)[1]

    pca = PCA()
    pca.fit(data)
    explained_variance = pca.explained_variance_ratio_
    total_explained_variance = np.cumsum(explained_variance)
    num_dims = np.min(np.where(total_explained_variance>VAF)[0])+1

    if project_data:
        all_projected_data = pca.transform(data)
        projected_data = all_projected_data[:,:num_dims]
    else:
        projected_data = None

    return list(explained_variance), num_dims, projected_data

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
        data [n, nfeatures]: Input data to plot in an nfeature dimensional space and compute outliers
        std_threshold [float]: Number of standard deviations away a data point is required to be to be classified as an outlier
        
    Returns:
        tuple: Tuple containing: 
            | **good_data_idx [n]:** Labels each data point if it is an outlier (True = good, False = outlier)
            | **distances [n]:** Distance of each data point from center
    '''
    
    # Check ncluster input
    kmeans_model = KMeans(n_clusters = 1).fit(data)
    distances = kmeans_model.transform(data)
    cluster_labels = kmeans_model.labels_
    dist_std = np.std(distances)
    good_data_idx = (distances < (dist_std*std_threshold))
                  
    return good_data_idx.flatten(), distances.flatten()

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