# behavior.py
#
# Behavioral metrics code, e.g. trajectory path lengths, eye movement analysis, success rate, etc.

import numpy as np
from scipy import signal
from sklearn.feature_selection import r_regression
from tqdm.auto import tqdm 

from .base import calc_rolling_average
from .. import preproc
from .. import postproc
from ..data import load_bmi3d_task_codes
'''
Behavioral metrics 
'''
def calc_success_percent(events, start_events=[b"TARGET_ON"], end_events=[b"REWARD", b"TRIAL_END"], success_events=b"REWARD", window_size=None):
    '''
    A wrapper around get_trial_segments which counts the number of trials with a reward event 
    and divides by the total number of trials. This function can either calculated the success percent
    across all trials in the input events, or compute a rolling success percent based on the 'window_size' 
    input argument. 
    
    See also:
        :func:`~aopy.analysis.calc_success_percent_trials`

    Args:
        events (nevents): events vector, can be codes, event names, anything to match
        start_events (int, str, or list, optional): set of start events to match
        end_events (int, str, or list, optional): set of end events to match
        success_events (int, str, or list, optional): which events make a trial a successful trial
        window_size (int, optional): [in number of trials] For computing rolling success perecent. How many trials 
            to include in each window. If None, this functions calculates the success percent across all trials.

    Returns:
        float or array (nwindow): success percent = number of successful trials out of all trials attempted.
    '''
    segments, _ = preproc.get_trial_segments(events, np.arange(len(events)), start_events, end_events)
    trial_success = [np.any(np.isin(success_events, trial)) for trial in segments]

    return calc_success_percent_trials(trial_success, window_size)

def calc_success_percent_trials(trial_success, window_size=None):
    '''
    A wrapper around get_trial_segments which counts the number of trials with a reward event 
    and divides by the total number of trials. This function can either calculated the success percent
    across all trials in the input events, or compute a rolling success percent based on the 'window_size' 
    input argument. 
    
    See also:
        :func:`~aopy.analysis.calc_success_percent`

    Args:
        trial_success ((ntrial,) bool array): boolean array of trials where success is non-zero and failure is zero
        window_size (int, optional): [in number of trials] For computing rolling success perecent. How many trials 
            to include in each window. If None, this functions calculates the success percent across all trials.

    Returns:
        float or array (nwindow): success percent = number of successful trials out of all trials attempted.
    '''
    n_trials = len(trial_success)

    # If requested, calculate success percent across entire input events
    if window_size is None:
        n_success = np.count_nonzero(trial_success)  
        success_percent = n_success / n_trials

    # Otherwise, compute rolling success percent
    else:
        success_percent = calc_rolling_average(trial_success, window_size)

    return success_percent

def calc_success_rate(events, event_times, start_events, end_events, success_events, window_size=None):
    '''
    Calculate the number of successful trials per second with a given trial start and end definition. 
    Inputs are raw event codes and times.

    See also:
        :func:`~aopy.analysis.calc_success_rate_trials`

    Args:
        events (nevents): events vector, can be codes, event names, anything to match
        event_times (nevents): time of events in 'events'
        start_events (int, str, or list, optional): set of start events to match
        end_events (int, str, or list, optional): set of end events to match
        success_events (int, str, or list, optional): which events make a trial a successful trial
        window_size (int, optional): [ntrials] For computing rolling success perecent. How many trials 
            to include in each window. If None, this functions calculates the success percent across all trials.

    Returns:
        float or array (nwindow): success rate [success/s] = number of successful trials completed per second of time between the start event(s) and end event(s).
    '''
    # Get event time information
    segments, times = preproc.get_trial_segments(events, event_times, start_events, end_events)
    trial_acq_time = times[:,1]-times[:,0]
    trial_success = [np.any(np.isin(success_events, trial)) for trial in segments]

    return calc_success_rate_trials(trial_success, trial_acq_time, window_size=window_size)

def calc_success_rate_trials(trial_success, trial_time, window_size=None):
    '''
    Calculate the number of successful trials per second with a given trial start and end definition. 

    See also:
        :func:`~aopy.analysis.calc_success_rate`

    Args:
        trial_success ((ntrial,) bool array): boolean array of trials where success is non-zero and failure is zero
        trial_time ((ntrial,) array): float array of the time taken in each trial (e.g. acquisition time)
        window_size (int, optional): [ntrials] For computing rolling success perecent. How many trials 
            to include in each window. If None, this functions calculates the success percent across all trials.

    Returns:
        float or array (nwindow): success rate [success/s] = number of successful trials completed per second of 
            time between the start event(s) and end event(s).
    '''
    assert len(trial_time) == len(trial_success), "Mismatched trial lengths"

    # Get % of successful trials per window 
    success_perc = calc_success_percent_trials(trial_success, window_size=window_size)
    ntrials = len(trial_time)

    # Determine rolling target acquisition time info 
    if window_size is None:
        nsuccess = success_perc*ntrials
        acq_time = np.sum(trial_time)

    else:
        nsuccess = success_perc
        acq_time = calc_rolling_average(trial_time, window_size)
    
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

def compute_movement_error(trajectory, target_position, rotation_vector=np.array([1, 0]), error_axis=1):
    """
    Computes movement error of a trajectory relative to the straight line between the origin and a target position. 

    Args:
        trajectory (nt, 2): The trajectory coordinates for each point in time. 
        target_position (2,): Target position coordinates.
        rotation_vector ((2,) array, optional): The vector onto which movement will be
            projected to calculate error. Defaults to np.array([1, 0]).
        error_axis (int, optional): axis (after rotation) along which to compute the error 
            statistics. Default 1.

    Returns:
        (nt,): The error of the trajectory relative to the target position
    """
    assert np.count_nonzero(target_position) > 0, "Please check target position. Must be non-zero"

    rotated_traj = postproc.rotate_spatial_data(trajectory, rotation_vector, target_position)
    return np.array(rotated_traj)[:, error_axis]

def compute_movement_stats(trajectory, target_position, rotation_vector=np.array([1, 0]), error_axis=1, 
                           return_all_stats=False):
    """
    Computes movement statistics of a trajectory relative to a target position. 

    Args:
        trajectory (nt, 2): The trajectory coordinates for each point in time. 
        target_position (2,): Target position coordinates.
        rotation_vector ((2,) array, optional): The vector onto which movement will be
            projected to calculate error. Defaults to np.array([1, 0]).
        error_axis (int, optional): axis (after rotation) along which to compute the error 
            statistics. Default 1.

    Returns:
        tuple: A tuple containing:
            | **mean (float):** The mean error of the trajectory relative to the target position.
            | **std (float):** The variance of the error of the trajectory relative to the target position.
            | **auc (float):** The area under the curve ofthe trajectory relative to the target position.
        additionally, with return_all_stats=True: 
            | **abs_mean (float):** The mean of the absolute value of the trajectory error.
            | **abs_min (float):** The minimum absolute trajectory error.
            | **abs_max (float):** The maximum absolute trajectory error.
            | **abs_auc (float):** The area under the curve of the absolute value of the trajectory error.
            | **sign (float):** 1 if the maximum positive value is bigger than the maximum negative value. -1 otherwise.
            | **signed_min (float):** The minimum value if the sign is 1, otherwise the maximum value of the trajectory error.
            | **signed_max (float):** The maximum value if the sign is 1, otherwise the minimum value of the trajectory error.
            | **signed_abs_mean (float):** The sign multiplied by the absolute value of the mean trajectory error.
        
    """
    dist_ts = compute_movement_error(trajectory, target_position, rotation_vector, error_axis)
    
    # Statistics of the error axis
    mean = np.mean(dist_ts)
    std = np.std(dist_ts)
    auc = np.sum(dist_ts)

    if not return_all_stats:
        return (mean, std, auc)
    
    # Unsigned statistics
    abs_mean = np.mean(np.abs(dist_ts))
    abs_min = np.min(np.abs(dist_ts))
    abs_max = np.max(np.abs(dist_ts))
    abs_auc = np.sum(np.abs(dist_ts))

    # Signed statistics
    sign = -1 if abs(np.min(dist_ts)) > abs(np.max(dist_ts)) else 1 # bigger negative or positive?
    signed_min = np.min(dist_ts) if sign == 1 else np.max(dist_ts)
    signed_max = np.max(dist_ts) if sign == 1 else np.min(dist_ts)
    signed_abs_mean = sign * abs_mean
    
    return (mean, std, auc, abs_mean, abs_min, abs_max, abs_auc, sign, signed_min, signed_max, signed_abs_mean)

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

def get_movement_onset(cursor_traj, fs, trial_start, target_onset, gocue, numsd=3.0, butter_order=4, low_cut=20, thr=None):
    '''
    Compute movement onset when cursor speed crosses threshold based on mean and standard deviation in baseline period.
    Speed is estimated from cursor trajectories and low-pass filtered to remove noise.
    Baseline is defined as the period between target onset and gocue because speed still exists soon after the cursor enters the center target.
    
    Args:
        cursor_traj ((ntr,) np object array) : cursor trajectory that begins with the time when the cursor enters the center target
        fs (float) : sampling rate in Hz
        trial_start (ntr) : trial start time (the time when the cursor enters the center target) relative to experiment start time in sec
        target_onset (ntr) : target onset relative to experiment start time in sec
        gocue (ntr) : gocue (the time when the center target disappears) relative to experiment start time in sec
        numsd (float) : for determining threshold at each trial
        butter_order (int) : the order for the butterworth filter
        low_cut (float) : cut off frequency for low pass filter in Hz
        thr (float) : thr when you want to use constant threshold across trials. If thr=None, thr is computed by mean + numsd*std in the period from target onset to gocue.
        
    Returns:
        movement_onset (ntr) : movement onset relative to trial start time (the time when the cursor enters the center target) in sec
    '''
    
    target_from_start = target_onset - trial_start # target onset relative to trial start time
    gocue_from_start = gocue - trial_start # gocue relative to trial start time
    dt = 1/fs
    
    b, a = signal.butter(butter_order, low_cut, btype='lowpass', fs=fs)

    movement_onset = []
    for itr in range(cursor_traj.shape[0]):
        # compute speed
        dist = np.linalg.norm(cursor_traj[itr],axis=1)
        speed_tmp = np.diff(dist)/(1/fs)
        speed_tmp = np.insert(speed_tmp,0,speed_tmp[0]) # complement the first data point
        speed = signal.filtfilt(b, a, speed_tmp, axis=0)
        
        # compute threshold based on mean and std in baseline
        t_cursor = np.arange(dist.shape[0])*dt
        if thr is None:
            baseline_idx = (t_cursor<gocue_from_start[itr]) & (t_cursor>target_from_start[itr])
            baseline_speed = np.mean(speed[baseline_idx])
            baseline_std = np.std(speed[baseline_idx],ddof=1)
            thr = baseline_speed + numsd*baseline_std
        
        # get movement onset
        movement_onset.append(t_cursor[np.where((speed>thr)&(t_cursor>target_from_start[itr]))[0][0]])
        
    return np.array(movement_onset)

def get_cursor_leave_time(cursor_traj, samplerate, target_radius, cursor_radius=0):
    '''
    Compute the times when the cursor leaves the center target radius
    
    Args:
        cursor_traj ((ntr,) np object array) : cursor trajectory that begins with the time when the cursor enters the center target
        fs (float) : sampling rate in Hz
        target_radius (float) : the radius of the center target in cm
        cursor_radius (float) : the radius of the cursor in cm. Default is 0
        
    Returns:
        cursor_leave_time (ntr): cursor leave times relative to the time when the cursor enters the center target
    '''
    
    ntr = len(cursor_traj)
    cursor_leave_time = []
    
    for itr in range(ntr):
        t_axis = np.arange(cursor_traj[itr].shape[0])/samplerate
        
        dist = np.linalg.norm(cursor_traj[itr],axis=1)
        leave_idx = np.where(dist > target_radius-cursor_radius)[0][0]
        cursor_leave_time.append(t_axis[leave_idx])
    
    return np.array(cursor_leave_time)

'''
Continuous tracking behavioral metrics
'''
def calc_tracking_error(user_traj, target_traj):
    '''
    Computes the mean-squared error between the user position and target position over time.

    Args:
        user_traj (nt,ndim): user trajectory over a trial segment
        target_traj (nt,ndim): target trajectory over a trial segment

    Returns:
        float array (ndim,): tracking error in each dimension
    '''
    assert len(user_traj) == len(target_traj), "User and target trajectories must be the same length!"
    return np.mean((user_traj - target_traj)**2, axis=0) # compute mean over time axis

def calc_tracking_in_time(event_codes, event_times, proportion=False):
    '''
    Computes the total amount of time that the cursor is inside the target over a trial segment.

    Args:
        event_codes (nevents,): list of event codes
        event_times (nevents,): list of event times
        proportion (bool, optional): whether to return the time as a proportion of the total trial segment time. 
            Default is False.

    Returns:
        float: amount of time (in seconds) that the cursor was in the target for
    '''   
    # get all the individual times when cursor was inside target
    task_codes = load_bmi3d_task_codes()
    start_events = [task_codes['CURSOR_ENTER_TARGET']]
    end_events = [task_codes['CURSOR_LEAVE_TARGET'], task_codes['REWARD']] # once cursor is in target, cursor can leave or trial can finish  
    cursor_in_target_segment, cursor_in_target_times = preproc.get_trial_segments_and_times(event_codes, event_times, start_events, end_events)

    # add up the individual times
    tracking_in_time = sum([t[1] - t[0] for t in cursor_in_target_times]) # end time of segment - start time of segment

    # optionally return the time as a proportion of the total segment length
    if proportion:
        tracking_in_time = tracking_in_time/(event_times[-1] - event_times[0])
        
    return tracking_in_time

'''
Hand behavior metrics
'''
def unit_vector(vector):
    '''
    Finds the unit vector of a given vector.

    Args:
        vector (list or array): D-dimensional vector

    Returns:
        unit_vector (list or array): D-dimensional vector with a magnitude of 1
    '''
    return vector/np.linalg.norm(vector)

def angle_between(v1, v2, in_degrees=False):
    '''
    Computes the angle between two vectors. By default, the angle will be in radians and fall within the range [0,pi].

    Args:
        v1 (list or array): D-dimensional vector
        v2 (list or array): D-dimensional vector
        in_degrees (bool, optional): whether to return the angle in units of degrees. Default is False.

    Returns:
        float: angle (in radians or degrees)
    '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    if in_degrees:
        angle = angle*180/np.pi

    return angle

def vector_angle(vector, in_degrees=False):
    '''
    Computes the angle of a vector on the unit circle.

    Args:
        vector (list or array): D-dimensional vector
        in_degrees (bool, optional): whether to return the angle in units of degrees. Default is False.

    Returns:
        float: angle (in radians or degrees)
    '''
    D = len(vector)
    assert D==2, "This function currently works best for 2-dimensional vectors"

    ref_vector = np.zeros((D,))
    ref_vector[0] = 1
    angle = angle_between(ref_vector, vector)
    
    # take the explementary (conjugate) angle for vectors that lie in Q3 or Q4
    if vector[1]<0: # negative y-coordinate
        angle = 2*np.pi - angle

    if in_degrees:
        angle = angle*180/np.pi

    return angle


def correlate_trajectories(trajectories, center=True, verbose=False):
    '''
    Correlates multiple trajectory datasets across trials by computing the 
    Pearson correlation coefficient (R) between all pairs of trials. This function computs R for each trajectory dimension, then returns a weighted average based on the variance. 

    Args:
        trajectories (nt, ndim, ntrials): Trajectories to correlate, 
                                 where `nt` is the number of timepoints, 
                                 `ntrials` is the number of trials, and 
                                 `ndims` is the number of dimensions for each trajectory (i.e. x, y, z).
        center (bool): If each trial should be centered before computing correlation. (Safaie et al. 2023 sets this to true)
        verbose (bool): If `True`, prints a progress bar during computation via the tqdm module. 

    Returns:
        ndarray: A 2D numpy array of Pearson correlation (R) scores between each pair of trials. 
                 The shape of the output will be (ntrials, ntrials).
    '''

    nt, ndims, ntrials = trajectories.shape
    
    traj_correlation = np.zeros((ntrials, ntrials))*np.nan

    trial_variance = np.var(trajectories, axis=0) # (ndim, ntrials)

    iterator = tqdm(range(ntrials)) if verbose else range(ntrials)
    for itrial in iterator:
        temp_corrs = np.zeros((ntrials,ndims))*np.nan
        for idim in range(ndims):
            weight = trial_variance[idim, itrial]
            temp_corrs[:,idim] = r_regression(trajectories[:,idim,:], trajectories[:,idim,itrial], center=center) * weight

        traj_correlation[itrial,:] = np.sum(temp_corrs, axis=1) / np.sum(trial_variance[:, itrial])
    
    return traj_correlation
