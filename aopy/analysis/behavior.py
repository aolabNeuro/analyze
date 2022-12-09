# behavior.py
#
# Behavioral metrics code, e.g. trajectory path lengths, eye movement analysis, success rate, etc.

import numpy as np
from .. import preproc
from scipy import signal


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
