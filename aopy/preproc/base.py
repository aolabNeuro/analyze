# preproc.py
# Code for preprocessing neural data (reorganize data into the needed form) including parsing
# experimental files, trial sorting, and subsampling

import numpy as np
from scipy import interpolate

from .. import analysis

'''
Timestamps and events
'''
def get_closest_value(timestamp, sequence, radius):
    '''
    Returns the value, within a specified radius, in given sequence 
    closest to given timestamp. if none exist, returns none. If two are
    equidistant, this function returns the lower value.

    Args: 
        timestamp (float): given timestamp
        sequence (nt): sequence to search for closest value
        radius (float): distance from timestamp to search for closest value

    Returns:
        tuple: tuple containing:
            | **closest_value (float):** value within sequence that is closest to timestamp
            | **closest_idx (int):** index of the closest_value in the sequence
    '''

    # initialize returned value
    closest_value = None
    minimum = None
    x_diff = np.zeros(len(sequence))

    # calculate differences
    x_diff = timestamp - np.array(sequence)
    # check x_diff within radius
    within_radius = np.abs(x_diff)<=radius
    # find closest sequence value based on x_diff
    if np.any(within_radius):
        minimum = np.argmin(np.abs(x_diff))
        closest_value = sequence[minimum]

    return closest_value, minimum

def find_measured_event_times(approx_times, measured_times, search_radius, return_idx=False):
    '''
    Uses closest_value() to repeatedly find a measured time for each approximate time.		

    Args:
        approx_times (nt): array of approximate event timestamps
        measured_times (nt'): array of measured timestamps that might correspond to the approximate timestamps
        search_radius (float): distance before and after each approximate time to search for a measured time 
        return_idx (bool, optional): if true, also return the index into measured time for each measured time
        
    Returns:
        tuple: tuple containing:
            | **parsed_ts (nt):** array of the same length as approximate timestamps, 
                but containing matching timestamps or np.nan
            | **prased_idx (nt):** array of indices into measured_times corresponding
                to the parsed timestamps
    '''

    parsed_idx = np.empty((len(approx_times),))
    parsed_ts = np.empty((len(approx_times),))
    parsed_idx[:] = np.nan
    parsed_ts[:] = np.nan

    # Find the closest neighbor for each approximate timestamp
    search_size = 1000
    idx_prev_closest = 0
    idx_next_closest = min(search_size, len(measured_times))
    for idx_ts, ts in enumerate(approx_times):

        # Try searching a small subset of measured times
        closest, idx_closest = get_closest_value(ts, measured_times[idx_prev_closest:idx_next_closest], search_radius)
        if closest:
            parsed_ts[idx_ts] = closest
            parsed_idx[idx_ts] = idx_prev_closest + idx_closest
            idx_prev_closest += idx_closest
            idx_next_closest = min(idx_next_closest + 1, len(measured_times))
            continue

        # If that doesn't work, look in the whole array. This approach speeds things up 
        # considerably if there are only a small number of missing measurements
        closest, idx_closest = get_closest_value(ts, measured_times[idx_next_closest:], search_radius)
        if closest:
            parsed_ts[idx_ts] = closest
            parsed_idx[idx_ts] = idx_next_closest + idx_closest
            idx_prev_closest = idx_next_closest + idx_closest
            idx_next_closest = min(idx_prev_closest + search_size, len(measured_times))

    if return_idx:
        return parsed_ts, parsed_idx
    else:
        return parsed_ts

def get_measured_clock_timestamps(estimated_timestamps, measured_timestamps, latency_estimate=0.01, search_radius=1./100):
    '''
    Takes estimated frame times and measured frame times and returns a time for each frame. If no closeby measurement
    can be found for a given estimate, that frame will be filled with np.nan

    Args:
        estimated_timestamps (nframes): timestamps when frames were thought to be displayed
        measured_timestamps (nt): timestamps when frames actually appeared on screen
        latency_estimate (float, optional): how long the display takes normally to update
        search_radius (float, optional): how far away to look for a measurement before giving up

    Returns:
        nframes array: measured timestamps, some of which will be np.nan if they were not displayed
    '''

    approx_timestamps = estimated_timestamps + latency_estimate
    return find_measured_event_times(approx_timestamps, measured_timestamps, search_radius)

def fill_missing_timestamps(uncorrected_timestamps):
    '''
    Fill missing timestamps by copying the subsequent timestamp over any NaNs. For example, if you have
    timestamps `[0.01, 0.08, np.nan, np.nan, 0.25, np.nan, 0.38]`, then apply fill_missing_timestamps, 
    the result would be `[0.01, 0.08, 0.25, 0.25, 0.25, 0.38, 0.38]`. Used by proc_exp() to give the times
    at which things appeared on the screen, since sometimes the screen will miss a refresh period and not 
    display something until the next cycle.

    Args:
        uncorrected_timestamps (nframes): timestamps with missing data (np.nan) because they were recorded
        from a source which sometimes skips frames

    Returns:
        corrected_timestamps (nframes): measured timestamps with missing values filled in with the next non-nan value
    '''

    # For any missing timestamps, the time at which they occurred is the next non-nan value
    missing = np.isnan(uncorrected_timestamps)
    corrected_timestamps = uncorrected_timestamps.copy()
    if missing.any():

        # Fill in missing values by reversing the order, then filling in the previous value
        backwards_timestamps = np.flip(corrected_timestamps)
        missing = np.isnan(backwards_timestamps)
        idx = np.where(~missing, np.arange(len(missing)), 0)
        np.maximum.accumulate(idx, out=idx) # apply maximum element-wise across the backwards array of indices
        backwards_timestamps[missing] = backwards_timestamps[idx[missing]]
        corrected_timestamps = np.flip(backwards_timestamps)

        # Unfortunately if the last few frames are missing then they never occurred, however this causes
        # problems in later analysis so we fill it in with the best guess. Shouldn't have much of
        # an impact since nothing ever happens on the last frame. So we repeat the process in the forward order
        missing = np.isnan(corrected_timestamps)
        idx = np.where(~missing, np.arange(len(missing)), 0)
        np.maximum.accumulate(idx, out=idx) # apply maximum element-wise across the backwards array of indices
        corrected_timestamps[missing] = corrected_timestamps[idx[missing]]

    return corrected_timestamps

def interp_timestamps2timeseries(timestamps, timestamp_values, samplerate=None, sampling_points=None, interp_kind='linear', extrap_values='extrapolate'):
    '''
    This function uses linear interpolation (scipy.interpolate.interp1d) to convert timestamped data to timeseries data given new sampling points.
    Timestamps must be monotonic. If the timestamps or timestamp_values include a nan, this function ignores the corresponding timestamp value and performs interpolation between the neighboring values.
    To calculate the new points from 'samplerate' this function creates sample points with the same range as 'timestamps' (timestamps[0], timestamps[-1]).
    Either the 'samplerate' or 'sampling_points' optional argument must be used. If neither are filled, the function will display a warning and return nothing.
    If both 'samplerate' and 'sampling_points' are input, the sampling points will be used. 
    If the input timestamps are not monotonic, the function will display a warning and return nothing.
    The optional argument 'interp_kind' corresponds to 'kind' and 'extrap_values' corresponds to 'fill_values' in scipy.interpolate.interp1d.
    More information about 'extrap_values' can be found on the scipy.interpolate.interp1d documentation page. 

    Example::
    >>> timestamps = np.array([1,2,3,4])
    >>> timestamp_values = np.array([100,200,100,300])
    >>> timeseries, sampling_points = interp_timestamps2timeseries(timestamps, timestamp_values, samplerate=2)
    >>> print(timeseries)
    np.array([100,150,200,150,100,200,300])
    >>> print(sampling_points)
    np.array([1,1.5,2,2.5,3,3.5,4])

    Args:
        timestamps (nstamps): Timestamps of original data to be interpolated between.
        timestamp_values (nstamps): Values corresponding to the timestamps.
        samplerate (float): Optional argument if new sampling points should be calculated based on the timstamps. Sampling rate of newly sampled output array. [Hz]
        output_array (nt): Optional argument to pass predefined sampling points. 
        interp_kind (str): Optional argument to define the kind of interpolation used. Defaults to 'linear'
        extrap_values (str, array, or tuple): Optional argument to define how values out of the range of 'timestamps' are fliled. This defaults to extrapolate but a tuple or array can be input to further define these values. ('fill_value' in scipy.interpolate.interp1d)

    Returns:
        tuple: tuple containing:
        | **timeseries (nt):** New timeseries of data.
        | **sampling_points (nt):** Sampling points used to calculate the new time series.

    '''
    # Check for nans and remove them
    if not np.all(np.logical_not(np.isnan(timestamps))) or not np.all(np.logical_not(np.isnan(timestamp_values))):
        nanmask_stamps = np.logical_not(np.isnan(timestamps))
        nanmask_values = np.logical_not(np.isnan(timestamp_values))
        nanmask = np.logical_and(nanmask_stamps, nanmask_values)
        timestamps = timestamps[nanmask]
        timestamp_values = timestamp_values[nanmask]

    # Check that timestamps are monotonic
    if not np.all(np.diff(timestamps) > 0):
        print("Warning: Input timemeseries is not monotonic")
        return

    # Check for sampling points information
    if samplerate is None and sampling_points is None:
        print("Warning: Not information to determine new sampling points is included. Please input the samplerate to calculate the new points from or the new sample points.")
        return

    # Calculate output sampling points if none are input
    if sampling_points is None:
        sampling_points = np.arange(timestamps[0], timestamps[-1]+(1/samplerate), 1/samplerate)

    # Interpolate
    f_interp = interpolate.interp1d(timestamps, timestamp_values, kind=interp_kind, fill_value=extrap_values)
    timeseries = f_interp(sampling_points)

    return timeseries, sampling_points

'''
Trial alignment
'''
def trial_separate(events, times, evt_start, n_events=8, nevent_offset=0):
    '''
    Compute the 2D matrices contaning events per trial and timestamps per trial. 
    If there are not enough events to fill n_events, the remaining indices will be a value of '-1' the events are ints or missing values if events are strings.

    Args:
        events (nt): events vector
        times (nt): times vector
        evt_start (int or str): event marking the start of a trial
        n_events (int): number of events in a trial
        nevent_offset (int): number of events before the start event to offset event alignment by. For example,
            if you wanted to align to "targ" in ["trial", "targ", "reward", "trial", "targ", "error"] but include the preceding "trial"
            event, then you could use nevent_offset=-1

    Returns:
        tuple: tuple containing:
            | **trial_events (n_trial, n_events):** events per trial
            | **trial_times (n_trial, n_events):** timestamps per trial
    '''

    # Pad the arrays a bit in case there is an evt_start at the beginning or end
    if np.issubdtype(events.dtype, np.number):
        if nevent_offset < 0:
            events = events.astype('int32')
            events = np.pad(events, (-nevent_offset, n_events), constant_values=(-1,))
            times = np.pad(times, (-nevent_offset, n_events), constant_values=(-1,))
        else:
            events = events.astype('int32')
            events = np.pad(events, (0, n_events+nevent_offset), constant_values=(-1,))
            times = np.pad(times, (0, n_events+nevent_offset), constant_values=(-1,))
    else:
        if nevent_offset < 0:
            events = np.pad(events, (-nevent_offset, n_events), constant_values=('',))
            times = np.pad(times, (-nevent_offset, n_events), constant_values=(-1,))
        else:
            events = np.pad(events, (0, n_events+nevent_offset), constant_values=('',))
            times = np.pad(times, (0, n_events+nevent_offset), constant_values=(-1,))    
    

    # Find the indices in events that correspond to evt_start 
    evt_start_idx = np.where(events == evt_start)[0]+nevent_offset

    # Find total number of trials
    num_trials = len(evt_start_idx)
    
    # Calculate trial_events and trial_times matrices
    trial_events = np.empty((num_trials, n_events), dtype=events.dtype)
    trial_times = np.empty((num_trials, n_events), dtype=times.dtype)
    for iE in range(len(evt_start_idx)):
        trial_events[iE,:] = events[evt_start_idx[iE]: evt_start_idx[iE]+n_events]
        trial_times[iE,:] = times[evt_start_idx[iE]: evt_start_idx[iE]+n_events]

    return trial_events, trial_times

def trial_align_events(aligned_events, aligned_times, event_to_align):
    '''
    Compute a new trial_times matrix with offset timestamps for the given event_to_align.
    Any index corresponding to where aligned_events is empty will also be empty.
    
    Args:
        aligned_events (n_trial, n_event): events per trial
        aligned_times (n_trial, n_event): timestamps per trial
        event_to_align (int or str): event to align to

    Returns:
        (n_trial, n_event): number of trials by number of events
    '''

    # For each row, find the column that matches the given event, 
    # then subtract its timestamps from the entire row
    trial_aligned_times = np.zeros(aligned_times.shape)
    for idx_trial in range(aligned_events.shape[0]):
        idx_time = np.where(aligned_events[idx_trial,:] == event_to_align)[0][0] # take the first match
        time_offset = aligned_times[idx_trial, idx_time]
        offset_row = aligned_times[idx_trial, :] - time_offset
        trial_aligned_times[idx_trial] = offset_row

        # Handle case where the input row of aligned_events has missing values.
        zero_idx = np.where(np.logical_or(aligned_events[idx_trial,:] == -1, aligned_events[idx_trial,:] == ''))[0]
        if len(zero_idx) > 0:
            trial_aligned_times[idx_trial,zero_idx] = 0

    return trial_aligned_times

def trial_align_data(data, trigger_times, time_before, time_after, samplerate):
    '''
    Transform data into chunks of data triggered by trial start times. If trigger_times is too long
    relative to 'data/samplerate', only the triggers that correspond to data will be returned.

    Args:
        data (nt, nch): arbitrary data, can be multidimensional
        trigger_times (ntrial): start time of each trial [s]
        time_before (float): amount of time [s] to include before the start of each trial
        time_after (float): time [s] to include after the start of each trial
        samplerate (int): sampling rate of data [samples/s]
    
    Returns:
        (ntrial, nt, nch): trial aligned data
    '''
    dur = time_after + time_before
    n_samples = int(np.floor(dur * samplerate))

    if data.ndim == 1:
        data.shape = (data.shape[0], 1)
    trial_aligned = np.zeros((len(trigger_times), n_samples, *data.shape[1:]))*np.nan

    # Don't look at trigger times that are after the end of the data
    max_trigger_time = (data.shape[0]/samplerate) - time_after
    last_trigger_idx = np.where(trigger_times < max_trigger_time)[0][-1]
    for t in range(last_trigger_idx+1):
        t0 = trigger_times[t] - time_before
        if np.isnan(t0):
            continue
        # sub = subvec(data, t0, n_samples, samplerate)
        trial_data = np.empty((n_samples,data.shape[1]))
        idx_start = int(np.floor(t0*samplerate))
        idx_end = min(data.shape[0], idx_start+n_samples)
        trial_data[:idx_end-idx_start,:] = data[idx_start:idx_end,:]
        trial_aligned[t,:min(len(trial_data),n_samples),:] = trial_data[:min(len(trial_data),n_samples),:]
    return np.squeeze(trial_aligned)

def trial_align_times(timestamps, trigger_times, time_before, time_after, subtract=True):
    '''
    Takes timestamps and splits them into chunks triggered by trial start times

    Args:
        timestamps (nt): events in time to be trial aligned
        trigger_times (ntrial): start time of each trial
        time_before (float): amount of time to include before the start of each trial
        time_after (float): time to include after the start of each trial
        subtract (bool, optional): whether the start of each trial should be set to 0
    
    Returns:
        tuple: tuple containing:
            | **trial_aligned (ntrial, nt):** trial aligned timestamps
            | **trial_indices (ntrial, nt):** indices into timestamps in the same shape as trial_aligned
    '''
    trial_aligned = []
    trial_indices = []
    for t in range(len(trigger_times)):
        t0 = trigger_times[t] - time_before
        t1 = trigger_times[t] + time_after
        trial_idx = (timestamps > t0) & (timestamps <= t1)
        sub = timestamps[trial_idx]
        if subtract:
            sub -= trigger_times[t]
        trial_aligned.append(sub)
        trial_indices.append(np.where(trial_idx)[0])
    return trial_aligned, trial_indices


def get_trial_segments(events, times, start_events, end_events):
    '''
    Gets times for the start and end of each trial according to the given set of start_events and end_events

    Args:
        events (nt): events vector
        times (nt): times vector
        start_events (list): set of start events to match
        end_events (list): set of end events to match

    Returns:
        tuple: tuple containing:
            | **segments (list of list of events):** a segment of each trial
            | **times (ntrials, 2):** list of 2 timestamps for each trial corresponding to the start and end events

    Note:
        - if there are multiple matching start or end events in a trial, only consider the first one
    '''
    # Find the indices in events that correspond to start events
    evt_start_idx = np.where(np.in1d(events, start_events))[0]

    # Extract segments for each start event
    segments = []
    segment_times = []
    for idx_evt in range(len(evt_start_idx)):
        idx_start = evt_start_idx[idx_evt]
        idx_end = evt_start_idx[idx_evt] + 1

        # Look forward for a matching end event
        while idx_end < len(events):
            if np.in1d(events[idx_end], start_events): 
                break # start event must be followed by end event otherwise not valid
            if np.in1d(events[idx_end], end_events):
                segments.append(events[idx_start:idx_end+1])
                segment_times.append([times[idx_start], times[idx_end]])
                break 
            idx_end += 1
    segment_times = np.array(segment_times)
    return segments, segment_times

def get_data_segments(data, segment_times, samplerate):
    '''
    Gets arbitrary length segments of data from a timeseries

    Args:
        data (nt, ndim): arbitrary timeseries data that needs to segmented
        segment_times (nseg, 2) pairs of start and end times for each segment
        samplerate (int): sampling rate of the data

    Returns:
        list of 1d arrays (nt): nt is the length of each segment (can be different for each)
    '''
    segments = []
    for idx_seg in range(segment_times.shape[0]):
        idx_data_start = int(segment_times[idx_seg,0]*samplerate)
        idx_data_end = int(segment_times[idx_seg,1]*samplerate)
        seg = data[idx_data_start:idx_data_end]
        segments.append(seg)
    return segments

def get_unique_conditions(trial_idx, conditions, condition_name='target'):
    '''
    Gets the unique trial combinations of each condition set. Used to parse BMI3D
    data when there is no 'trials' array in the HDF file. Output looks something
    like this for a center-out experiment::

        'trial'     'index'     'target'
        0           0           (0, 0, 0)
        0           5           (8, 0, 0)
        1           0           (0, 0, 0)
        1           2           (0, 8, 0)
        ...

    Args:
        n_trials (int): number of trials
        trial_idx (int array): which trials happen on each cycle
        conditions (ndarray): which conditions happen on each cycle
        condition_name (str, optional): what the conditios are called

    Returns:
        record array: array of type [('trial', 'u8'), ('index', 'u8'), 
            (condition_name, 'f8', (3,)))] describing the unique conditions on each trial
    '''
    conditions = conditions.round(decimals=6)
    if conditions.ndim == 1:
        conditions = np.reshape(conditions, (conditions.shape[0], 1))
    unique_conditions = np.unique(conditions, axis=0)

    trial_dtype = np.dtype([('trial', 'u8'), ('index', 'u8'), (condition_name, 'f8', (conditions.shape[1],))])
    corrected_trials = np.empty((0,), dtype=trial_dtype)
    trial = np.empty((1,), dtype=trial_dtype)

    n_trials = len(np.unique(trial_idx))
    for idx_trial in range(n_trials):

        # For each unique condition, add a trial entry if it matches any condition that belong to this trial
        trial_conditions = conditions[np.reshape(trial_idx == idx_trial, -1),:]
        for idx_unique_cond in range(unique_conditions.shape[0]):
            if (trial_conditions == unique_conditions[idx_unique_cond]).all(axis=1).any():
                trial['trial'] = idx_trial
                trial['index'] = idx_unique_cond
                trial[condition_name] = unique_conditions[idx_unique_cond]
                corrected_trials = np.append(corrected_trials, trial)
    return corrected_trials

def locate_trials_with_event(trial_events, event_codes, event_columnidx=None):
    '''
    Given an array of trial separated events, this function goes through and finds the event sequences corresponding to the trials
    that include a given event. If an array of event codes are input, the function will find the trials corresponding to
    each event code. 
    
    Args:
        trial_events (ntr, nevents): Array of trial separated event codes
        event_codes (int, str, list, or 1D array): Event code(s) to find trials for. Can be a list of strings or ints
        event_column (int): Column index to look for events in. Indexing starts at 0. Keep as 'None' if all columns should be analyzed.
        
    Returns:
        tuple: Tuple containing:
            | **split_events (list of arrays):** List where each index includes an array of trials containing the event_code corresponding to that index. 
            | **split_events_combined (1D Array):** Concatenated indices for which trials correspond to which event code.
                        Can be used as indices to order 'trial_events' by the 'event_codes' input.

    Example::
        >>> aligned_events_str = np.array([['Go', 'Target 1', 'Target 1'],
                ['Go', 'Target 2', 'Target 2'],
                ['Go', 'Target 4', 'Target 1'],
                ['Go', 'Target 1', 'Target 2'],
                ['Go', 'Target 2', 'Target 1'],
                ['Go', 'Target 3', 'Target 1']])
        >>> split_events, split_events_combined = locate_trials_with_event(aligned_events_str, ['Target 1','Target 2'])
        >>> print(split_events)
        [array([0, 2, 3, 4, 5], dtype=int64), array([1, 3, 4], dtype=int64)]
        >>> print(split_events_combined)
        [0 2 3 4 5 1 3 4]      

    '''
    split_events = []
    if type(event_codes) == int or type(event_codes) == str:
        split_events.append(np.unique(np.where(trial_events[:,event_columnidx] == event_codes)[0]))
        split_events_combined = np.array(split_events).flatten()
    else:
        nevent_codes = len(event_codes)
        split_events_combined = np.array([]).astype(int)
        for ievent in range(nevent_codes):
            split_events.append(np.unique(np.where(trial_events[:,event_columnidx] == event_codes[ievent])[0]))
            split_events_combined = np.append(split_events_combined, split_events[ievent])
    
    return split_events, split_events_combined

def calc_eye_calibration(cursor_data, cursor_samplerate, eye_data, eye_samplerate, event_cycles, event_times, event_codes,
    align_events=range(81,89), trial_end_events=[239], offset=0., return_datapoints=False, debug=True):
    """
    Extracts cursor data and eyedata and calibrates, aligning them and calculating the least square fitting coefficients
    
    Args:
        
        align_events (list, optional): list of event codes to use for alignment. By default, align to
            when the cursor enters 8 peripheral targets
        trial_end_events (list, optional): list of end events to use for alignment. By default trial end is code 239
        offset (float, optional): time (in seconds) to offset from the given events to correct for a delay in eye movements
        return_datapoints (bool, optional): if true, also returns cusor_data_aligned, eye_data_aligned
        debug (bool, optional): prints additional debug information

    Returns:
        tuple: tuple containing:
            | **coefficients (neyech, 2):** coefficients [slope, intercept] for each eye channel
            | **correlation_coeff (neyech):** correlation coefficients for each eye channel
    """

    # Get cursor kinematics
    _, trial_cycles = get_trial_segments(event_codes, event_cycles, align_events, trial_end_events)
    if trial_cycles.size == 0:
        raise ValueError("Not enough trials to calculate eye calibration")
    align_cycles = trial_cycles[:,0] + int(offset * cursor_samplerate)
    cursor_data_aligned = cursor_data[align_cycles, :]
    if debug: print(f'Using {len(cursor_data_aligned)} cursor x,y positions to calibrate eye tracking data')

    # Get the corresponding eye data
    _, trial_times= get_trial_segments(event_codes, event_times, align_events, trial_end_events)
    align_times = trial_times[:,0] + offset
    sample_eye_enter_target  = (align_times * eye_samplerate).astype(int)
    eye_data_aligned = eye_data[sample_eye_enter_target,:]
    
    # Calibrate the eye data
    if eye_data_aligned.shape[1] == 4:
        cursor_data_aligned = np.tile(cursor_data_aligned, (1, 2)) # for two eyes
    slopes, intercepts, correlation_coeff = analysis.fit_linear_regression(eye_data_aligned, cursor_data_aligned)
    coeff = np.vstack((slopes, intercepts)).T

    if return_datapoints:
        return coeff, correlation_coeff, cursor_data_aligned, eye_data_aligned
    else:
        return coeff, correlation_coeff