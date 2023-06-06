# behavior.py
#
# Behavioral metrics code, e.g. trajectory path lengths, eye movement analysis, success rate, etc.

import numpy as np
from scipy import signal

from .base import calc_rolling_average
from .. import preproc
from .. import postproc

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
            |**mean (float):** The mean error of the trajectory relative to the target position.
            |**std (float):** The variance of the error of the trajectory relative to the target position.
            |**auc (float):** The area under the curve ofthe trajectory relative to the target position.
        additionally, with return_all_stats=True: 
            |**abs_mean (float):** The mean of the absolute value of the trajectory error.
            |**abs_min (float):** The minimum absolute trajectory error.
            |**abs_max (float):** The maximum absolute trajectory error.
            |**abs_auc (float):** The area under the curve of the absolute value of the trajectory error.
            |**sign (float):** 1 if the maximum positive value is bigger than the maximum negative value. -1 otherwise.
            |**signed_min (float):** The minimum value if the sign is 1, otherwise the maximum value of the trajectory error.
            |**signed_max (float):** The maximum value if the sign is 1, otherwise the minimum value of the trajectory error.
            |**signed_abs_mean (float):** The sign multiplied by the absolute value of the mean trajectory error.
        
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
