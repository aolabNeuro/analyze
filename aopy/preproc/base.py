# preproc.py
# Code for preprocessing neural data (reorganize data into the needed form) including parsing
# experimental files, trial sorting, and subsampling

import warnings

import numpy as np
from scipy import interpolate

from .. import utils
from .. import precondition
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

def validate_measurements(expected_values, measured_values, diff_thr):
    '''
    Corrects sensor measurements to expected values. If the difference between any of the measured values 
    and the expected values fall outside the given threshold (diff_thr) then the expected value is used. If
    it is within the threshold, the measured value is used. The two input arrays must have the same lengths.

    Args:
        expected_values (nt): known or expected values
        measured_values (nt): measured data that may be spurious
        diff_thr (float): threshold above which differences are deemed to large and expected values are returned

    Returns:
        tuple: tuple containing:
            | **corrected_values (nt):** array of the same length as the inputs but with validated values
            | **diff_above_thr (nt):** boolean array of values passing the difference threshold
    '''
    expected_values = np.squeeze(expected_values)
    measured_values = np.squeeze(measured_values)
    assert expected_values.shape == measured_values.shape
    diff = np.abs(expected_values - measured_values)
    diff_above_thr = diff > diff_thr
    corrected_values = measured_values.copy()
    corrected_values[diff_above_thr] = expected_values[diff_above_thr]
    return corrected_values, diff_above_thr

def interp_timestamps2timeseries(timestamps, timestamp_values, samplerate=None, sampling_points=None, interp_kind='linear', extrapolate=False, remove_nan=True):
    '''
    This function uses linear interpolation (scipy.interpolate.interp1d) to convert timestamped data to timeseries data given new sampling points.
    Timestamps must be monotonic. If the timestamps or timestamp_values include a nan, this function ignores the corresponding timestamp value and performs interpolation between the neighboring values.
    To calculate the new points from 'samplerate' this function creates sample points with the same range as 'timestamps' (timestamps[0], timestamps[-1]).
    Either the 'samplerate' or 'sampling_points' optional argument must be used. If neither are filled, the function will display a warning and return nothing.
    If both 'samplerate' and 'sampling_points' are input, the sampling points will be used. 
    The optional argument 'interp_kind' corresponds to 'kind'. The optional argument 'extrapolate' determines whether timepoints falling outside the 
    range of the input timestamps should be extrapolated or not. If not, they are copied from the first and last valid values, depending on whether they
    appear at the beginnin or end of the timeseries, respectively.
    
    Note:
        Enforces monotonicity of the input timestamps by removing timestamps and their associated values that do not increase.

    Example:
        ::
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
        samplerate (float, optional): Optional argument if new sampling points should be calculated based on the timstamps. Sampling rate of newly sampled output array in Hz. Default None.
        sampling_points ((nt,) array, optional): Optional argument to pass predefined sampling points. Default None.
        interp_kind (str, optional): Optional argument to define the kind of interpolation used. Defaults to 'linear'
        extrapolate (bool, optional): If True, use scipy.interp1d's built-in extrapolate functionality. Otherwise, use the first and last valid data.
        remove_nan (bool, optional): If True, remove nan values before interpolation. Default True.

    Returns:
        tuple: tuple containing:
            | **timeseries (nt):** New timeseries of data.
            | **sampling_points (nt):** Sampling points used to calculate the new time series.

    '''
    # Check for nans and remove them
    if remove_nan and (not np.all(np.logical_not(np.isnan(timestamps))) or not np.all(np.logical_not(np.isnan(timestamp_values)))):
        nanmask_stamps = np.logical_not(np.isnan(timestamps))
        nanmask_values = np.logical_not(np.isnan(timestamp_values))
        if timestamp_values.ndim > 1:
            nanmask_values = np.any(nanmask_values, axis=tuple(range(1,nanmask_values.ndim))) # all axes except 0
        nanmask = np.logical_and(nanmask_stamps, nanmask_values)
        timestamps = timestamps[nanmask]
        timestamp_values = timestamp_values[nanmask]

    # Check that timestamps are monotonic
    if not np.all(np.diff(timestamps) > 0):
        tmask = np.insert(np.diff(timestamps) > 0, 0, True)
        timestamps = timestamps[tmask]
        timestamp_values = timestamp_values[tmask]

    # Check for sampling points information
    if samplerate is None and sampling_points is None:
        raise ValueError("No information to determine new sampling points is included. Please input the samplerate to calculate the new points from or the new sample points.")

    # Calculate output sampling points if none are input
    if sampling_points is None:
        sampling_points = np.arange(timestamps[0], timestamps[-1]+(1/samplerate), 1/samplerate)

    # Determine extrapolation values
    if extrapolate:
        fill_value = 'extrapolate'
    else:
        fill_value = (timestamp_values[0], timestamp_values[-1])

    # Interpolate
    f_interp = interpolate.interp1d(timestamps, timestamp_values, kind=interp_kind, fill_value=fill_value, bounds_error=False, axis=0)
    timeseries = f_interp(sampling_points)

    if remove_nan:
        assert np.count_nonzero(np.isnan(timeseries)) == 0, f"{np.count_nonzero(np.isnan(timeseries))} NaNs present in output -- something has gone wrong!"

    return timeseries, sampling_points

def sample_timestamped_data(data, timestamps, samplerate, upsamplerate=None, append_time=0, **kwargs):
    '''
    Convert irregularly spaced data into a timeseries at the given samplerate. 
    First interpolates the data (by default to 100 times the samplerate), then
    downsamples to the given samplerate. Optionally adds extra time at the end
    of the timeseries.

    Args:
        data (nt, ...): A numpy array of shape (nt, ...) containing the data to 
            be sampled. The first dimension must represent the time index.
        timestamps (nt,): The timestamp (in seconds) for each data point in data.
        samplerate (float): The desired output sampling rate in Hz.
        upsamplerate (float, optional): The upsampling rate to use for interpolation. 
            Defaults to 100 times the samplerate.
        append_time (float, optional): The amount of extra time to add at the end 
            of the timeseries, in seconds. Defaults to 0.
        kwargs (dict, optional): arguments to include in interpolation function

    Returns:
        (ns, ...): cursor_data_time containing the sampled data.
    '''
    assert len(data) == len(timestamps), (f"Data and timestamps should "
        f"have the same number of cycles ({len(data)} vs {len(timestamps)})")

    if upsamplerate is None:
        upsamplerate = samplerate * 10

    assert upsamplerate >= samplerate, "Upsamplerate must be greater than or equal to samplerate"

    time = np.arange(int((timestamps[-1] + append_time)*upsamplerate))/upsamplerate # add extra time
    data_time, _ = interp_timestamps2timeseries(timestamps, data, sampling_points=time, interp_kind='linear', extrapolate=False, **kwargs)
    if upsamplerate > samplerate:
        data_time = precondition.downsample(data_time, upsamplerate, samplerate)
    return data_time

def get_dch_data(digital_data, digital_samplerate, dch):
    '''
    Transform digital data stored as integers into timestamps and values corresponding to
    a single channel or a set of channels.

    Args:
        digital_data (nt,): timeseries of digital data
        digital_samplerate (float): sampling rate of the digital data
        dch (int or list): channel(s) to get data from

    Returns:
        (nedges,): structured np.ndarray of the form [('timestamp', 'f8'), ('value', 'f8')])
    '''
    dch_bit_mask = utils.convert_channels_to_mask(dch)
    dch_ts_data = utils.extract_bits(digital_data, dch_bit_mask)
    dch_timestamps, dch_values = utils.detect_edges(dch_ts_data, digital_samplerate, rising=True, falling=True)
    dch_data = np.empty((len(dch_timestamps),), dtype=[('timestamp', 'f8'), ('value', 'f8')])
    dch_data['timestamp'] = dch_timestamps
    dch_data['value'] = dch_values
    return dch_data

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
        evt_start (int or str or list): event or events marking the start of a trial
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
    if np.issubdtype(np.array(events).dtype, np.number):
        events = np.array(events).astype('int32')
        fill_value = -1
    else:
        fill_value = ''
    if nevent_offset < 0:
        events = np.pad(events, (-nevent_offset, n_events), constant_values=(fill_value,))
        times = np.pad(times, (-nevent_offset, n_events), constant_values=(-1,))
    else:
        events = np.pad(events, (0, n_events+nevent_offset), constant_values=(fill_value,))
        times = np.pad(times, (0, n_events+nevent_offset), constant_values=(-1,))    
    

    # Find the indices in events that correspond to evt_start 
    evt_start_idx = np.where(np.isin(events, evt_start))[0]+nevent_offset

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

def get_trial_data(data, trigger_time, time_before, time_after, samplerate, channels=None):
    '''
    Get one chunk of data triggered by a trial start time. Chunks are padded with
    nan if the data does not span the entire chunk

    Args:
        data (nt, nch): arbitrary data, can be multidimensional
        trigger_time (float): start time of the trial
        time_before (float): amount of time [s] to include before the start of the trial
        time_after (float): time [s] to include after the start of the trial
        samplerate (int): sampling rate of data [samples/s]
    
    Returns:
        (nt, nch): trial aligned data chunk
    '''
    dur = time_after + time_before
    n_samples = int(np.floor(dur * samplerate))
    if channels is not None:
        nch = len(channels)
    else:
        nch = data.shape[1]
        channels = ...

    if data.ndim == 1:
        data.shape = (data.shape[0], 1)

    t0 = trigger_time - time_before
    if np.isnan(t0):
        return
    
    trial_data = np.zeros((n_samples,nch))*np.nan
    idx_start = int(np.round(t0*samplerate, 0))
    idx_end = min(data.shape[0], idx_start+n_samples)
    if idx_start < 0:
        trial_data[-idx_start:idx_end-idx_start] = data[:idx_end,channels]
    else:
        trial_data[:(idx_end-idx_start),:] = data[idx_start:idx_end,channels]
    return trial_data

def trial_align_data(data, trigger_times, time_before, time_after, samplerate):
    '''
    Transform data into chunks of data triggered by trial start times. If trigger_times is too long
    relative to 'data/samplerate', only the triggers that correspond to data will be included, while
    others will be filled with np.nan.

    Args:
        data (nt, nch): arbitrary data, can be multidimensional
        trigger_times (ntrial): start time of each trial [s]
        time_before (float): amount of time [s] to include before the start of each trial
        time_after (float): time [s] to include after the start of each trial
        samplerate (int): sampling rate of data [samples/s]
    
    Returns:
        (nt, nch, trial): trial aligned data

    Modified Sept. 2023 - transpose output
    '''
    dur = time_after + time_before
    n_samples = int(np.floor(dur * samplerate))
    trigger_times = np.array(trigger_times)

    if data.ndim == 1:
        data.shape = (data.shape[0], 1)
    trial_aligned = np.zeros((len(trigger_times), n_samples, *data.shape[1:]))*np.nan

    # Don't look at trigger times that are after the end of the data
    max_trigger_time = (data.shape[0]/samplerate) - time_after
    if max_trigger_time < trigger_times[0]:
        warnings.warn("No valid triggers to align to")
        return trial_aligned.transpose(1,2,0)
    last_trigger_idx = np.where(trigger_times < max_trigger_time)[0][-1]
    for t in range(last_trigger_idx+1):
        if np.isnan(trigger_times[t]):
            continue
        trial_aligned[t] = get_trial_data(data, trigger_times[t], time_before, time_after, samplerate)
    return trial_aligned.transpose(1,2,0)

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

def get_trial_segments(events, times, start_events, end_events, repeating_start_events=False):
    '''
    Gets times for the start and end of each trial according to the given set of start_events and end_events

    Args:
        events (nt): events vector
        times (nt): times vector
        start_events (list): set of start events to match
        end_events (list): set of end events to match
        repeating_start_events (bool): whether the start events might occur multiple times within one segment. Otherwise always use the last start event within a segment. May lead to segments spanning multiple trials if used improperly. Default False.

    Returns:
        tuple: tuple containing:
            | **segments (list of list of events):** a segment of each trial
            | **times (ntrials, 2):** list of 2 timestamps for each trial corresponding to the start and end events

    Note:
        - if there are multiple matching start or end events in a trial, the default is to only consider the first one. See :func:`~aopy.preproc.get_trial_segments_and_times` for more detail.
    '''
    segments, segment_times = get_trial_segments_and_times(events, times, start_events, end_events, repeating_start_events=repeating_start_events)
    segment_times = np.array([[t[0], t[-1]] for t in segment_times])
    return segments, segment_times

def get_trial_segments_and_times(events, times, start_events, end_events, repeating_start_events=False, return_idx=False):
    '''
    This function is similar to get_trial_segments() except it returns the timestamps of all events in event code.
    Trial align the event codes with corresponding event times.

    Args:
        events (nt): events vector
        times (nt): times vector
        start_events (list): set of start events to match
        end_events (list): set of end events to match
        repeating_start_events (bool, optional): whether the start events might occur multiple times within one segment. Otherwise always use the last start event within a segment. May lead to segments spanning multiple trials if used improperly. Default False.
        return_idx (bool, optional): whether to return the indices into events and times corresponding to the segments
        
    Returns:
        tuple: tuple containing:
            | **segments (list of list of events):** a segment of each trial
            | **times (list of list of times):** list of timestamps corresponding to each event in the event code
            | **idx (list of list of indices, optional):** list of indices into events and times corresponding to each event in the event code
    
    Note:
        - if there are multiple matching start or end events in a trial, the default is to only consider the first one. See examples for more detail.

    Examples:

        For segments that do not contain multiple start events (e.g. trials from center-out reaching task):

        .. code-block:: python

            task_codes = load_bmi3d_task_codes()
            start_events = [task_codes['CENTER_TARGET_ON']]
            end_events = [task_codes['TRIAL_END']]
            print(start_events, end_events)

            # example task data
            events = [16, 80, 18, 32, 82, 48, 239, 16, 80, 19, 66, 239, 16, 80, 19, 32, 83, 48, 239]
            times = np.arange(0,len(events))

            # segments should contain only one 'CENTER_TARGET_ON' per segment, since the task is always to reach from the center to a random peripheral target
            segments, times = get_trial_segments_and_times(events, times, start_events, end_events)
            print(segments)
            print(times)

        For segments that contain multiple start events (e.g. trials from corners reaching task):

        .. code-block:: python    

            from aopy.data.bmi3d import load_bmi3d_task_codes
            task_codes = load_bmi3d_task_codes()
            start_events = [task_codes['CORNER_TARGET_ON']]
            end_events = [task_codes['TRIAL_END']]
            print(start_events, end_events)

            # example task data
            events = [18, 82, 19, 34, 83, 48, 239, 19, 83, 17, 66, 239, 19, 83, 17, 35, 81, 48, 239]
            times = np.arange(0,len(events))
            segments, times = get_trial_segments_and_times(events, times, start_events, end_events, repeating_start_events=True)
            
            # segments should contain multiple 'CORNER_TARGET_ON' events within the same segment, since the task is to reach from one random corner to another
            print(segments)
            print(times)

        For segments that do not contain multiple start events themselves but multiple start events may exist in the larger task structure (e.g. delay period from corners reaching task):

        .. code-block:: python   

            from aopy.data.bmi3d import load_bmi3d_task_codes
            task_codes = load_bmi3d_task_codes()
            start_events = [task_codes['CORNER_TARGET_ON']] # same as above, but now we want when the 2nd corner target in the sequence turns on
            end_events = [task_codes['CORNER_TARGET_OFF']] # when 1st corner target turns off, indicating end of delay period
            print(start_events, end_events)

            # example task data (same as above)
            events = [18, 82, 19, 34, 83, 48, 239, 19, 83, 17, 66, 239, 19, 83, 17, 35, 81, 48, 239]
            times = np.arange(0,len(events))
            segments, times = get_trial_segments_and_times(events, times, start_events, end_events, repeating_start_events=False)
            
            # without repeating_start_events, segments skip over the 1st corner target in the sequence and start instead from when the 2nd corner target turns on (e.g when the delay period begins)
            print(segments)
            print(times)

    '''
    if np.shape(start_events) == ():
        start_events = [start_events]
    if np.shape(end_events) == ():
        end_events = [end_events]
        
    # Find the indices in events that correspond to start events
    evt_start_idx = [i for i, evt in enumerate(events) if evt in start_events]

    # Extract segments for each start event
    segment_idx = []
    segments = []
    segment_times = []
    prev_segment_last_evt_idx = 0
    for idx_evt in range(len(evt_start_idx)):
        idx_start = evt_start_idx[idx_evt]
        idx_end = evt_start_idx[idx_evt] + 1

        if repeating_start_events:
            if idx_start < prev_segment_last_evt_idx:
                continue

        # Look forward for a matching end event
        while idx_end < len(events):
            if not repeating_start_events:
                if events[idx_end] in start_events:
                    break # start event must be followed by end event otherwise not valid
            if events[idx_end] in end_events:
                segment_idx.append(list(range(idx_start, idx_end+1)))
                segments.append(events[idx_start:idx_end+1])
                segment_times.append(times[idx_start:idx_end+1])
                prev_segment_last_evt_idx = idx_end
                break
            idx_end += 1

    if return_idx:
        return segments, segment_times, segment_idx    
    else:
        return segments, segment_times

def get_data_segment(data, start_time, end_time, samplerate, channels=None):
    '''
    Gets a single arbitrary length sengment of data from a timeseries

    Args:
        data (nt, ndim): arbitrary timeseries data that needs to segmented
        start_time (float): start of the segment in seconds
        end_time (float): end of the segment in seconds
        samplerate (int): sampling rate of the data
        channels (list, optional): list of channels to include in the segment.
            If None, include all channels. Useful for subselecting HDF data without
            loading it all into memory.

    Returns:
        (nt', ndim): short segment of data from the given times
    '''
    idx_data_start = max(0, int(start_time*samplerate))
    idx_data_end = min(len(data), int(end_time*samplerate))
    if channels is not None:
        return data[idx_data_start:idx_data_end,channels]
    else:
        return data[idx_data_start:idx_data_end]

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
        segments.append(get_data_segment(data, segment_times[idx_seg,0], segment_times[idx_seg,1], samplerate))
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

def calc_eye_calibration(cursor_data, cursor_samplerate, eye_data, eye_samplerate, event_times, event_codes,
    align_events=range(81,89), penalty_events=[64], offset=0., return_datapoints=False):
    """
    Extracts cursor data and eyedata and calibrates, aligning them and calculating the least square fitting coefficients
    
    Args:
        cursor_data (nt, 3): cursor data in time
        cursor_samplerate (float): sampling rate of the cursor data
        eye_data (nt, 2 or 4): eye data in time (optionally for both left and right eyes)
        eye_samplerate (float): sampling rate of the eye data
        event_times (nevent): times at which events occur
        event_codes (nevent): codes for each event
        align_events (list, optional): list of event codes to use for alignment. By default, align to
            when the cursor enters 8 peripheral targets
        penalty_events (list, optional): list of penalty event codes to exclude. By default,
            events followed by hold penalties are excluded
        offset (float, optional): time (in seconds) to offset from the given events. Positive offset yields later eye data
        return_datapoints (bool, optional): if true, also returns cusor_data_aligned, eye_data_aligned

    Returns:
        tuple: tuple containing:
            | **coefficients (neyech, 2):** coefficients [slope, intercept] for each eye channel
            | **correlation_coeff (neyech):** correlation coefficients for each eye channel
    """

    # Get the corresponding cursor and eye data
    trial_events, trial_times= trial_separate(event_codes, event_times, align_events, n_events=2)
    if len(trial_times) == 0:
        raise ValueError("Not enough trials to calibrate")
    exclude_trials = np.array([(t[-1] in penalty_events) for t in trial_events], dtype='bool')
    align_times = trial_times[~exclude_trials,0] + offset
    sample_cursor_enter_target  = (align_times * cursor_samplerate).astype(int)
    cursor_data_aligned = cursor_data[sample_cursor_enter_target,:]
    sample_eye_enter_target  = (align_times * eye_samplerate).astype(int)
    eye_data_aligned = eye_data[sample_eye_enter_target,:]
    
    # Find indexes with valid sample_eye_enter_target (not NaN)
    valid_indexes = ~np.isnan(sample_eye_enter_target)
    sample_eye_enter_target = sample_eye_enter_target[valid_indexes].astype(int)
    eye_data_aligned = eye_data[sample_eye_enter_target,:]
    cursor_data_aligned = cursor_data_aligned[valid_indexes, :]

    # Calibrate the eye data
    if eye_data_aligned.shape[1] == 4:
        cursor_data_aligned = np.tile(cursor_data_aligned, (1, 2)) # for two eyes
    slopes, intercepts, correlation_coeff = analysis.fit_linear_regression(eye_data_aligned, cursor_data_aligned)
    coeff = np.vstack((slopes, intercepts)).T

    if return_datapoints:
        return coeff, correlation_coeff, cursor_data_aligned, eye_data_aligned
    else:
        return coeff, correlation_coeff

def calc_eye_target_calibration(eye_data, eye_samplerate, event_times, event_codes, target_pos,
    align_events=range(81,89), penalty_events=[64], cursor_enter_center_target=80, offset=0., duration=0.1, return_datapoints=False):
    """
    Extracts eye data and calibrates, aligning eye data and calculating the least square fitting coefficients 
    between eye positions and target positions
    
    Args:
        eye_data (nt, 2 or 4): eye data in time (optionally for both left and right eyes)
        eye_samplerate (float): sampling rate of the eye data
        event_times (nevent): times at which events occur
        event_codes (nevent): codes for each event
        target_pos (ntarget, 3): N target positions (x,y,z). This should be ordered by their target idx
        align_events (list, optional): list of event codes to use for alignment. By default, align to
            when the cursor enters 8 peripheral targets
        penalty_events (list, optional): list of penalty event codes to exclude. By default,
            events followed by hold penalties are excluded
        cursor_enter_center_target (int, optional): event code for cursor_enter_center_target
        offset (float, optional): time (in seconds) to offset from the given events. Positive offset yields later eye data
        duration (float, optional): duration (in seconds) to average eye trajectories in the interval from the given events + offset
        return_datapoints (bool, optional): if true, also returns cusor_data_aligned, eye_data_aligned

    Returns:
        tuple: tuple containing:
            | **coefficients (neyech, 2):** coefficients [slope, intercept] for each eye channel
            | **correlation_coeff (neyech):** correlation coefficients for each eye channel
    """

    # Get the corresponding cursor and eye data
    trial_events, trial_times= trial_separate(event_codes, event_times, align_events, n_events=2)
    if len(trial_times) == 0:
        raise ValueError("Not enough trials to calibrate")
    exclude_trials = np.array([(t[-1] in penalty_events) for t in trial_events], dtype='bool')
    align_times = trial_times[~exclude_trials,0] + offset
    
    start_time  = (align_times * eye_samplerate).astype(int)
    end_time =  ((align_times + duration) * eye_samplerate).astype(int)
    if duration == 0.:
        end_time += 1
    eye_data_aligned = np.array([np.nanmean(eye_data[start:end,:],axis=0) for start, end in zip(start_time,end_time)])
        
    # Get target position for every trial
    target_index = np.array([event[0] - cursor_enter_center_target for event in trial_events])
    target_index = target_index[~exclude_trials]
    
    target_pos_trial = np.nan*np.zeros((len(target_index), 2))
    for itrial in range(len(target_index)):
        target_pos_trial[itrial,:] = target_pos[target_index[itrial]-1,:2]
    
    # Calibrate the eye data
    if eye_data_aligned.shape[1] == 4:
        target_pos_trial = np.tile(target_pos_trial, (1, 2)) # for two eyes
    else:
        target_pos_trial = target_pos_trial

    slopes, intercepts, correlation_coeff = analysis.fit_linear_regression(eye_data_aligned, target_pos_trial)
    coeff = np.vstack((slopes, intercepts)).T
    
    if return_datapoints:
        return coeff, correlation_coeff, eye_data_aligned
    else:
        return coeff, correlation_coeff