# preproc.py
# code for preprocessing neural data
import numpy as np
import numpy.lib.recfunctions as rfn
from .data import *

'''
Digital calc
'''
def convert_analog_to_digital(analog_data, thresh=.3):
    '''
    This function takes analog data and converts it to digital data given a 
    threshold. It scales the analog to between 0 and 1 and uses thres as a 

    Args: 
        analog_data (nt, nch): Time series array of analog data
        thresh (float, optional): Minimum threshold value to use in conversion

    Returns:
        (nt, nch): Array of 1's or 0's indicating if the analog input was above threshold  
    '''
    # Scale data between 0 and 1 so that threshold is a percentange
    minval = np.min(analog_data)
    maxval = np.max(analog_data)

    analog_data_scaled = (analog_data - minval)/maxval

    # Initialize digital_data
    digital_data = np.empty(analog_data_scaled.shape) # Default to empty 
    digital_data[:] = np.nan

    # Set any value less than the threshold to be 0
    digital_data[analog_data_scaled < thresh] = 0

    # Set any value greater than threshold to be 0
    digital_data[analog_data_scaled >= thresh] = 1

    # Check that there are no nan values in output data

    return digital_data

def detect_edges(digital_data, samplerate, rising=True, falling=True, check_alternating=True):
    '''
    Finds the timestamp and corresponding value of all the bit flips in data. Assumes 
    the first element in data isn't a transition

    By default, also enforces that rising and falling edges must alternate, always taking the
    last edge as the most valid one. For example::

        >>> data = [0, 0, 3, 0, 3, 2, 2, 0, 1, 7, 3, 2, 2, 0]
        >>> ts, values = detect_edges(data, fs)
        >>> print(values)
        [3, 0, 3, 0, 7, 0]

    Args:
        digital_data (ntime x 1): masked binary data array
        samplerate (int): sampling rate of the data used to calculate timestamps
        rising (bool, optional): include low to high transitions
        falling (bool, optional): include high to low transitions
        check_alternating (bool, optional): if True, enforces that rising and falling
            edges must be alternating

    Returns:
        tuple: tuple containing:

            timestamps (nbitflips): when the bits flipped
            values (nbitflips): corresponding values for each change
    '''

    digital_data = np.squeeze(np.uint64(digital_data)) # important conversion for binary math
    rising_idx = (~digital_data[:-1] & digital_data[1:]) > 0 # find low->high transitions
    falling_idx = (~digital_data[1:] & digital_data[:-1]) > 0

    # Find any non-alternating edges
    invalid = np.zeros((len(digital_data)-1,), dtype='?')
    if check_alternating:
        all_edges = np.where(rising_idx | falling_idx)[0]
        next_edge_rising = True
        for idx in range(len(all_edges)):
            this_idx = all_edges[idx]
            if next_edge_rising and rising_idx[this_idx]:
                # Expected rising and found rising
                next_edge_rising = False 
            elif not next_edge_rising and falling_idx[this_idx]:
                # Expected falling and found falling
                next_edge_rising = True 
            elif idx > 0: # skip the first edge since there is no previous edge
                # Unexpected; there must be an extra edge somewhere.
                # We will count this one as valid and the previous one as invalid
                prev_idx = all_edges[idx-1]
                invalid[prev_idx] = True
    
    # Assemble final index    
    logical_idx = np.zeros((len(digital_data)-1,), dtype='?')
    if rising:
        logical_idx |= rising_idx
    if falling:
        logical_idx |= falling_idx
    logical_idx &= np.logical_not(invalid)
    logical_idx = np.insert(logical_idx, 0, False) # first element never a transition

    time = np.arange(np.size(digital_data))/samplerate
    return time[logical_idx], digital_data[logical_idx]

def mask_and_shift(data, bit_mask):
    '''
    Apply bit mask and shift data to the least significant set bit in the mask. 
    For example,
    mask_and_shift(0001000011110000, 1111111100000000) => 00010000
    mask_and_shift(0001000011110000, 0000000011111111) => 11110000

    Args:
        data (ntime): digital data
        bit_mask (int): which bits to filter

    Returns:
        (nt): masked and shifted data
    '''

    return np.bitwise_and(data, bit_mask) >> find_first_significant_bit(bit_mask)

def find_first_significant_bit(x):
    '''
    Find first significant big. Returns the index, counting from 0, of the
    least significant set bit in x. Helper function for mask_and_shift

    Args:
        x (int): a number

    Returns:
        int: index of first significant nonzero bit
    '''
    return (x & -x).bit_length() - 1 # no idea how it works! thanks stack overflow --LRS

def convert_channels_to_mask(channels):
    '''
    Helper function to take a range of channels into a bitmask

    Args:
        channels (int array): 0-indexed channels to be masked
    
    Returns:
        int: binary mask of the given channels
    '''
    try:
        # Range of channels
        _ = iter(channels)
        channels = np.array(channels)
        flags = np.zeros(64, dtype=int)
        flags[channels] = 1
        return int(np.dot(np.array([2**i for i in range(64)]), flags))
    except:
        
        # Single channel
        return int(1 << channels)

def convert_digital_to_channels(data_64_bit):
    '''
    Converts 64-bit digital data from eCube into channels.

    Args:
        data_64_bit (n): masked 64-bit data, little-endian

    Returns:
        (n, 64): where channel 0 is least significant bit
    '''

    # Take the input, split into bytes, then unpack each byte, all little endian
    packed = np.squeeze(np.uint64(data_64_bit)) # required conversion to unsigned int
    unpacked = np.unpackbits(packed.view(np.dtype('<u1')), bitorder='little')
    return unpacked.reshape((packed.size, 64))

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

            closest_value (float): value within sequence that is closest to timestamp
            closest_idx (int): index of the closest_value in the sequence
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

def find_measured_event_times(approx_times, measured_times, search_radius):
    '''
    Uses closest_value() to repeatedly find a measured time for each approximate time.		

    Args:
        approx_times (nt): array of approximate event timestamps
        measured_times (nt'): array of measured timestamps that might correspond to the approximate timestamps
        search_radius (float): distance before and after each approximate time to search for a measured time 
        
    Returns:
        tuple: tuple containing:

            parsed_ts (nt): array of the same length as approximate timestamps, 
            but containing matching timestamps or np.nan
    '''

    parsed_ts = np.empty((len(approx_times),))
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
            idx_prev_closest += idx_closest
            idx_next_closest = min(idx_next_closest + 1, len(measured_times))
            continue

        # If that doesn't work, look in the whole array. This approach speeds things up 
        # considerably if there are only a small number of missing measurements
        closest, idx_closest = get_closest_value(ts, measured_times[idx_next_closest:], search_radius)
        if closest:
            parsed_ts[idx_ts] = closest
            idx_prev_closest = idx_next_closest + idx_closest
            idx_next_closest = min(idx_prev_closest + search_size, len(measured_times))

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

def get_edges_from_onsets(onsets, pulse_width):
    '''
    This function calculates the values and timepoints corresponding to a given time series 
    of pulse onsets (timestamp corresponding to the rising edge of a pulse). 
    Args:
        onsets (nonsets): Time point corresponding to a pulse onset. 
        pulse_width (float): Pulse duration 
    Returns:
        tuple: tuple containing:
        timestampes (2*nonsets + 1): Timestamps of the rising and falling edges. Always starts at 0.
        values (2*nonsets + 1): Values corresponding to the output timestamps.
    '''
    timestamps = np.zeros((1+len(onsets)*2,))
    values = np.zeros((1+len(onsets)*2,))
    for t in range(len(onsets)):
        timestamps[1+2*t] = onsets[t]
        values[1+2*t] = 1
        timestamps[2+2*t] = onsets[t]+pulse_width
        values[2+2*t] = 0
    return timestamps, values

''' =========================================================================================================
Event filtering
''' 
def get_matching_events(event_log, event_to_match):
    '''
    Given a list of tuple of (events, timestamps), find the matched event and the timestamps
    
    Args:
        event_log (list of (event, timestamp) tuples): log of events and times
        event_to_match (int or str): event to be matched to
    
    Returns:
        list: returns a list of matched events and their time stamps
    '''
    #use python filter function to speed up the searching
    return list(filter(lambda k: k[0] == event_to_match, event_log) )

def get_event_occurrences(event_log, event_to_count):
    '''
    Given event_log, count the number of occurances of event_to_count

    Args:
        event_log (list of (event, timestamp) tuples): log of events and times
        event_to_count (int or str): event to be matched to

    Returns:
        int: num_occurances
    '''
    matched_events_in_list = get_matching_events(event_log, event_to_count)
    num_occurances = len(matched_events_in_list)
    return num_occurances

def calc_events_duration(event_log):
    '''
    given an event_log and succuss_event,
    calculate the succuss rate

    Args:
        event_log (list of (event, timestamp) tuples): log of events and times
    
    Returns:
        float: events_duration
    '''

    # Unpack the first and last events
    first_event_name, first_event_timestamp = event_log[0]
    last_event_name, last_event_timestamp = event_log[-1]

    # Take the difference between the timestamps
    events_duration = last_event_timestamp - first_event_timestamp
    return events_duration

def calc_event_rate(trial_events, event_codes, debug = False):
    '''
    Given an trial_log and event_name, calculate the fraction of trials with the event

    Args:
        trial_events (ntr, nevents): Array of trial separated event codes
        event_codes (int, str, list, or 1D array): event codes to calculate the fractions for. 

    Returns:
        float or an array: fraction of matching events divided by total events
    '''
    
    if type(event_codes) == int or type(event_codes) == str:
        event_codes = np.array([event_codes])

    event_occurances = np.zeros(len(event_codes))

    num_trials = len(trial_events)

    split_events, _ = locate_trials_with_event(trial_events, event_codes)

    event_occurances = np.array([len(e) for e in split_events])
    event_rates = event_occurances / num_trials

    if len(event_rates) == 1: return event_rates[0]
    
    return event_rates



def calc_reward_rate(event_log, event_name='REWARD'):
    '''
    A wrapper for calc_event_rate
    event_name defauls to be 'REWARD'

    Args:
        event_log (list of (event, timestamp) tuples): log of events and times
        event_name (str or int): event to be matched to

    Returns:
        float: reward_rate
    '''
    return calc_event_rate(event_log, event_name)

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

            trial_events (n_trial, n_events): events per trial

            trial_times (n_trial, n_events): timestamps per trial
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
    Transform data into chunks of data triggered by trial start times

    Args:
        data (nt, nch): arbitrary data, can be multidimensional
        trigger_times (ntrial): start time of each trial
        time_before (float): amount of time to include before the start of each trial
        time_after (float): time to include after the start of each trial
        samplerate (int): sampling rate of data
    
    Returns:
        (ntrial, nt, nch): trial aligned data
    '''
    dur = time_after + time_before
    n_samples = int(np.floor(dur * samplerate))

    if data.ndim == 1:
        data.shape = (data.shape[0], 1)
    trial_aligned = np.zeros((len(trigger_times), n_samples, *data.shape[1:]))
    for t in range(len(trigger_times)):
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

            trial_aligned (ntrial, nt): trial aligned timestamps
            trial_indices (ntrial, nt): indices into timestamps in the same shape as trial_aligned
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

            segments (list of list of events): a segment of each trial
            times (ntrials, 2): list of 2 timestamps for each trial corresponding to the start and end events

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
        (tuple):
            (list of arrays): List where each index includes an array of trials containing the event_code corresponding to that index. 
            (1D Array): Concatenated indices for which trials correspond to which event code.
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

def max_repeated_nans(a):
    '''
    Utility to calculate the maximum number of consecutive nans

    Args:
        a (ndarray): input sequence

    Returns:
        int: max consecutive nans
    '''
    mask = np.concatenate(([False],np.isnan(a),[False]))
    if ~mask.any():
        return 0
    else:
        idx = np.nonzero(mask[1:] != mask[:-1])[0]
        return (idx[1::2] - idx[::2]).max()

'''
Prepare experiment files
'''
def parse_bmi3d(data_dir, files):
    '''
    Wrapper around version-specific bmi3d parsers

    Args:
        data_dir (str): where to look for the data
        files (dict): dictionary of files for this experiment
    
    Returns:
        tuple: tuple containing:

            data (dict): bmi3d data
            metadata (dict): bmi3d metadata
    '''
    # Check that there is hdf data in files
    if not 'hdf' in files:
        raise ValueError('Cannot parse nonexistent data!')

    # Load bmi3d data to see which sync protocol is used
    try:
        events, event_metadata = load_bmi3d_hdf_table(data_dir, files['hdf'], 'sync_events')
        sync_version = event_metadata['sync_protocol_version']
    except:
        sync_version = -1

    # Pass files onto the appropriate parser
    if sync_version <= 0:
        data, metadata = _parse_bmi3d_v0(data_dir, files)
        metadata['bmi3d_parser'] = 0
        metadata['sync_protocol_version'] = sync_version

    elif sync_version < 6:
        data, metadata = _parse_bmi3d_v1(data_dir, files)
        metadata['bmi3d_parser'] = 1
    else:
        print("Warning: this bmi3d sync version is untested!")
        data, metadata = _parse_bmi3d_v1(data_dir, files)
        metadata['bmi3d_parser'] = 1

    # Standardize the parsed variable names and perform some error checking
    metadata['bmi3d_source'] = os.path.join(data_dir, files['hdf'])
    return _prepare_bmi3d_v0(data, metadata)

def _parse_bmi3d_v0(data_dir, files):
    '''
    Simple parser for BMI3D data which basically ignores timing from the eCube.

    Args:
        data_dir (str): where to look for the data
        files (dict): dictionary of files for this experiment
    
    Returns:
        tuple: tuple containing:

            data (dict): bmi3d data
            metadata (dict): bmi3d metadata
    '''
    bmi3d_hdf_filename = files['hdf']
    metadata = {}

    # Load bmi3d data
    bmi3d_task, bmi3d_task_metadata = load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task')
    bmi3d_state, _ = load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task_msgs')
    bmi3d_events, bmi3d_event_metadata = load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'sync_events')
    bmi3d_root_metadata = load_bmi3d_root_metadata(data_dir, bmi3d_hdf_filename)
    
    # Copy metadata
    metadata.update(bmi3d_task_metadata)
    metadata.update(bmi3d_event_metadata)
    metadata.update(bmi3d_root_metadata)
    metadata.update({
        'source_dir': data_dir,
        'source_files': files,
    }) 

    # Estimate timestamps
    bmi3d_cycles = np.arange(len(bmi3d_task))
    bmi3d_timestamps = bmi3d_cycles/bmi3d_task_metadata['fps']
    bmi3d_clock = np.empty((len(bmi3d_task),), dtype=[('time', 'u8'), ('timestamp', 'f8')])
    bmi3d_clock['time'] = bmi3d_cycles
    bmi3d_clock['timestamp'] = bmi3d_timestamps

    # Put data into dictionary
    bmi3d_data = dict(
        bmi3d_clock=bmi3d_clock,
        bmi3d_task=bmi3d_task,
        bmi3d_state=bmi3d_state,
        bmi3d_events=bmi3d_events,
    )
    return bmi3d_data, metadata

def _parse_bmi3d_v1(data_dir, files):
    '''
    Parser for BMI3D data which incorporates ecube data. Only compatible with sync versions > 0

    Args:
        data_dir (str): where to look for the data
        files (dict): dictionary of files for this experiment
    
    Returns:
        tuple: tuple containing:

            data_dict (dict): bmi3d data
            metadata_dict (dict): bmi3d metadata
    '''

    data_dict = {}
    metadata_dict = {}

    # Load bmi3d data
    bmi3d_hdf_filename = files['hdf']
    bmi3d_task, bmi3d_task_metadata = load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task')
    bmi3d_state, _ = load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task_msgs')
    bmi3d_events, bmi3d_event_metadata = load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'sync_events')
    sync_protocol_version = bmi3d_event_metadata['sync_protocol_version']
    bmi3d_sync_clock, _ = load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'sync_clock') # there isn't any clock metadata
    bmi3d_trials, _ = load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'trials') # there isn't any trial metadata
    bmi3d_root_metadata = load_bmi3d_root_metadata(data_dir, bmi3d_hdf_filename)

    # Copy metadata
    metadata_dict.update(bmi3d_task_metadata)
    metadata_dict.update(bmi3d_event_metadata)
    metadata_dict.update(bmi3d_root_metadata)
    metadata_dict.update({
        'source_dir': data_dir,
        'source_files': files,
    }) 

    # And data
    data_dict.update({
        'bmi3d_task': bmi3d_task,
        'bmi3d_state': bmi3d_state,
        'bmi3d_clock': bmi3d_sync_clock,
        'bmi3d_events': bmi3d_events,
        'bmi3d_trials': bmi3d_trials,
    })  

    if 'ecube' in files:
        ecube_filename = files['ecube']
    
        # Load ecube digital data to find the strobe and events from bmi3d
        digital_data, metadata = load_ecube_digital(data_dir, ecube_filename)
        digital_samplerate = metadata['samplerate']

        # Load ecube analog data for the strobe and reward system
        analog_channels = [bmi3d_event_metadata['screen_measure_ach'], bmi3d_event_metadata['reward_measure_ach']] # [5, 0]
        ecube_analog, metadata = load_ecube_analog(data_dir, ecube_filename, channels=analog_channels)
        clock_measure_analog = ecube_analog[:,0]
        reward_system_analog = ecube_analog[:,1]
        analog_samplerate = metadata['samplerate']

        # Mask and detect BMI3D computer events from ecube
        event_bit_mask = convert_channels_to_mask(bmi3d_event_metadata['event_sync_dch']) # 0xff0000
        ecube_sync_data = mask_and_shift(digital_data, event_bit_mask)
        ecube_sync_timestamps, ecube_sync_events = detect_edges(ecube_sync_data, digital_samplerate, rising=True, falling=False)
        sync_events = np.empty((len(ecube_sync_timestamps),), dtype=[('timestamp', 'f8'), ('code', 'u1')])
        sync_events['timestamp'] = ecube_sync_timestamps
        sync_events['code'] = ecube_sync_events
        if sync_protocol_version < 3:
            clock_sync_bit_mask = 0x1000000 # wrong in 1 and 2
        else:
            clock_sync_bit_mask = convert_channels_to_mask(bmi3d_event_metadata['screen_sync_dch']) 
        clock_sync_data = mask_and_shift(digital_data, clock_sync_bit_mask)
        clock_sync_timestamps, _ = detect_edges(clock_sync_data, digital_samplerate, rising=True, falling=False)
        sync_clock = np.empty((len(clock_sync_timestamps),), dtype=[('timestamp', 'f8')])
        sync_clock['timestamp'] = clock_sync_timestamps

        # Mask and detect screen sensor events (A5 and D5)
        clock_measure_bit_mask = convert_channels_to_mask(bmi3d_event_metadata['screen_measure_dch']) # 1 << 5
        clock_measure_data_online = mask_and_shift(digital_data, clock_measure_bit_mask)
        clock_measure_timestamps_online, clock_measure_values_online = detect_edges(clock_measure_data_online, digital_samplerate, rising=True, falling=True)
        measure_clock_online = np.empty((len(clock_measure_timestamps_online),), dtype=[('timestamp', 'f8'), ('value', 'f8')])
        measure_clock_online['timestamp'] = clock_measure_timestamps_online
        measure_clock_online['value'] = clock_measure_values_online
        clock_measure_digitized = convert_analog_to_digital(clock_measure_analog, thresh=0.5)
        clock_measure_timestamps_offline, clock_measure_values_offline = detect_edges(clock_measure_digitized, analog_samplerate, rising=True, falling=True)
        measure_clock_offline = np.empty((len(clock_measure_timestamps_offline),), dtype=[('timestamp', 'f8'), ('value', 'f8')])
        measure_clock_offline['timestamp'] = clock_measure_timestamps_offline
        measure_clock_offline['value'] = clock_measure_values_offline

        # And reward system (A0)
        reward_system_digitized = convert_analog_to_digital(reward_system_analog)
        reward_system_timestamps, reward_system_values = detect_edges(reward_system_digitized, analog_samplerate, rising=True, falling=True)
        reward_system = np.empty((len(reward_system_timestamps),), dtype=[('timestamp', 'f8'), ('state', '?')])
        reward_system['timestamp'] = reward_system_timestamps
        reward_system['state'] = reward_system_values

        # Wrap everything up
        data_dict.update({
            'sync_events': sync_events,
            'sync_clock': sync_clock,
            'measure_clock_online': measure_clock_online,
            'measure_clock_offline': measure_clock_offline,
            'reward_system': reward_system,
        })    
    return data_dict, metadata_dict

def _prepare_bmi3d_v0(data, metadata):
    '''
    Organizes the bmi3d data and metadata and computes some automatic conversions

    Args:
        data (dict): bmi3d data
        metadata (dict): bmi3d metadata

    Returns:
        tuple: tuple containing:

            data (dict): prepared bmi3d data
            metadata (dict): prepared bmi3d metadata
    '''
    parser_version = metadata['bmi3d_parser']
    internal_clock = data['bmi3d_clock']
    internal_events = data['bmi3d_events']
    task = data['bmi3d_task']
    state = data['bmi3d_state']

    # Calculate t0
    if 'sync_events' in data and 'sync_clock' in data and len(data['sync_clock']) > 0:

        event_exp_start = internal_events[internal_events['event'] == b'EXP_START']
        sync_events = data['sync_events']
        bmi3d_start_time = sync_events['timestamp'][sync_events['code'] == event_exp_start['code']]
        if len(bmi3d_start_time) == 0:
            bmi3d_start_time = data['sync_clock']['timestamp'][0]
        elif len(bmi3d_start_time) > 1:
            bmi3d_start_time = bmi3d_start_time[0] # TODO: why are there sometimes two????

        # Better estimate for t0 is actually the first clock cycle
        first_timestamp = data['sync_clock']['timestamp'][0]
        if abs(first_timestamp - bmi3d_start_time) < 0.1: # sanity check, EXP_START is more reliable
            bmi3d_start_time = first_timestamp
    else:
        bmi3d_start_time = 0

    # Estimate display latency
    if metadata['sync_protocol_version'] >= 3 and 'sync_clock' in data and 'measure_clock_offline' in data \
        and len(data['sync_clock']) > 0:

        # Estimate the latency based on the "sync" state at the beginning of the experiment
        sync_impulse = data['sync_clock']['timestamp'][1:3]
        measure_impulse = get_measured_clock_timestamps(sync_impulse, data['measure_clock_offline']['timestamp'],
            latency_estimate=0.01, search_radius=0.1)
        if np.count_nonzero(np.isnan(measure_impulse)) > 0:
            print("Warning: sync failed. Using latency estimate 0.01")
            measure_latency_estimate = 0.01
        else:
            measure_latency_estimate = np.mean(measure_impulse - sync_impulse)
            print("Sync latency estimate: {:.4f} s".format(measure_latency_estimate))
    else:

        # The latency in previous versions was around 10 ms
        measure_latency_estimate = 0.01
    metadata['measure_latency_estimate'] = measure_latency_estimate

    # By default use the internal clock and events. Just need to make sure not to include
    # any clock cycles from the sync period at the beginning of the experiment
    event_cycles = internal_events['time']
    if metadata['sync_protocol_version'] < 6:
        valid_cycles = np.in1d(event_cycles, internal_clock['time'])
        event_idx = np.in1d(internal_clock['time'], event_cycles[valid_cycles])
        event_timestamps = np.empty((len(event_cycles),), dtype='f')
        event_timestamps[:] = np.nan
        event_timestamps[valid_cycles] = internal_clock['timestamp'][event_idx]
    else:
        event_timestamps = internal_clock['timestamp'][event_cycles]
    corrected_events = rfn.append_fields(internal_events, 'timestamp_bmi3d', event_timestamps, dtypes='f8')

    # Correct the events based on sync if present
    if 'sync_events' in data and 'sync_clock' in data and len(data['sync_clock']) > 0:
        
        # Check that the events are all present
        sync_events = data['sync_events']
        event_dict = metadata['event_sync_dict'] # dictionary between event names and event codes
        if sync_events['code'][0] != internal_events['code'][0]:
            print("Warning: first event ({}) doesn't match bmi3d records ({})".format(sync_events['code'][0], internal_events['code'][0]))
            event = np.zeros((1,), dtype=corrected_events.dtype)
            event['code'] = sync_events['code'][0]
            event['time'] = 0
            event['timestamp_bmi3d'] = event_timestamps[0] # could be NaN
            # TODO decode
            corrected_events = np.insert(corrected_events, event, 0)
        if sync_events['code'][-1] != internal_events['code'][-1]:
            print("Warning: last event ({}) doesn't match bmi3d records ({})".format(sync_events['code'][-1], internal_events['code'][-1]))
            event = np.zeros((1,), dtype=corrected_events.dtype)
            event['code'] = sync_events['code'][-1]
            event['time'] = internal_events['time'][-1]
            event['timestamp_bmi3d'] = event_timestamps[-1] # could be NaN
            # TODO decode
            corrected_events = np.append(corrected_events, event)

        # Add sync timestamps
        corrected_events = rfn.append_fields(corrected_events, 'timestamp_sync', sync_events['timestamp'], dtypes='f8')

        # Remove events that aren't in internal_events
        # invalid_idx = np.where(internal_events['code'] != corrected_events['code'][:len(internal_events)])[0]
        # while len(invalid_idx) > 0:
        #     corrected_events = np.delete(corrected_events, invalid_idx[0])
        #     invalid_idx = np.where(internal_events['code'] != corrected_events['code'][:len(internal_events)])[0]

    # Check that the number of frames is consistent with the clock
    approx_clock = internal_clock.copy()
    if 'sync_clock' in data and len(data['sync_clock']) > 0:
        sync_clock = data['sync_clock']
        time_zero_events = internal_events['event'] == b'TIME_ZERO'
        try:
            time_zero_code = internal_events[time_zero_events]['code']
            bmi3d_time_zero = sync_events['timestamp'][sync_events['code'] == time_zero_code]
            approx_clock['timestamp'] += bmi3d_time_zero
        except:
            print("Warning: bmi3d time zero not recorded. Estimates might be off")
            bmi3d_time_zero = sync_events['timestamp'][0] # use the first event instead, probably off by tens of milliseconds at least
            approx_clock['timestamp'] += bmi3d_time_zero
        if len(sync_clock) == 0:
            print("Warning: no clock timestamps on the eCube. Maybe something was unplugged?")
            print("Using internal clock timestamps")
        elif len(sync_clock) < len(internal_clock):
            print("Warning: length of clock timestamps on eCube ({}) doesn't match bmi3d record ({})".format(len(sync_clock), len(internal_clock)))
            print("Adding internal clock timestamps to the end of the recording...")
            approx_clock['timestamp'][:len(sync_clock)] = sync_clock['timestamp']
        elif len(sync_clock) > len(internal_clock):
            raise RuntimeError("Extra timestamps detected, something has gone horribly wrong.")

    # Correct the clock
    corrected_clock = approx_clock.copy()
    corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_bmi3d', approx_clock['timestamp'], dtypes='f8')
    
    # 1. Digital clock from BMI3D via NI DIO card
    if 'sync_clock' in data and len(data['sync_clock']) > 0:
        sync_latency_estimate = 0
        sync_search_radius = 0.01
        timestamp_sync = get_measured_clock_timestamps(
            approx_clock['timestamp'], data['sync_clock']['timestamp'], 
                sync_latency_estimate, sync_search_radius)
        corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_sync', timestamp_sync, dtypes='f8')

    # 2. Screen photodiode measurements, digitized online by NXP microcontroller
    measure_search_radius = 0.01
    max_consecutive_missing_cycles = metadata['fps'] # maximum 1 second missing
    metadata['has_measured_timestamps'] = False
    if 'measure_clock_online' in data and len(data['measure_clock_online']) > 0:
        # Find the timestamps for each cycle of bmi3d's state machine from all the clock sources
        timestamp_measure_online = get_measured_clock_timestamps(
            approx_clock['timestamp'], data['measure_clock_online']['timestamp'], 
                measure_latency_estimate, measure_search_radius)
        corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_measure_online', timestamp_measure_online, dtypes='f8')

        # If there are few missing measurements, include this as the default `timestamp`
        corrected_timestamps = fill_missing_timestamps(timestamp_measure_online)
        metadata['latency_measured'] = np.nanmean(corrected_timestamps - timestamp_measure_online) - measure_latency_estimate
        metadata['n_missing_markers'] = np.count_nonzero(np.isnan(timestamp_measure_online))
        n_consecutive_missing_cycles = max_repeated_nans(timestamp_measure_online)
        if n_consecutive_missing_cycles < max_consecutive_missing_cycles:
            metadata['has_measured_timestamps'] = True
            corrected_clock['timestamp'] = corrected_timestamps
        else:
            print(f"Digital screen sensor missing too many markers ({n_consecutive_missing_cycles}/{max_consecutive_missing_cycles}). Ignoring")

    # 3. Screen photodiode measurements, raw voltage digitized offline
    if 'measure_clock_offline' in data and len(data['measure_clock_offline']) > 0:
        timestamp_measure_offline = get_measured_clock_timestamps(
            approx_clock['timestamp'], data['measure_clock_offline']['timestamp'], 
                measure_latency_estimate, measure_search_radius)
        corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_measure_offline', timestamp_measure_offline, dtypes='f8')
        
        # If there are few missing measurements, include this as the default `timestamp`
        corrected_timestamps = fill_missing_timestamps(timestamp_measure_offline)
        metadata['latency_measured'] = np.nanmean(corrected_timestamps - timestamp_measure_offline) - measure_latency_estimate
        metadata['n_missing_markers'] = np.count_nonzero(np.isnan(timestamp_measure_offline))
        n_consecutive_missing_cycles = max_repeated_nans(timestamp_measure_offline)
        if n_consecutive_missing_cycles < max_consecutive_missing_cycles:
            corrected_clock['timestamp'] = corrected_timestamps
            metadata['has_measured_timestamps'] = True
        else:
            print(f"Analog screen sensor missing too many markers ({n_consecutive_missing_cycles}/{max_consecutive_missing_cycles}). Ignoring")

    # Create a 'trials' table if it doesn't exist
    if not 'bmi3d_trials' in data:
        try:
            trial_idx = task['trial']
            n_trials = len(np.unique(trial_idx))
        except:
            print("Warning: trials missing. Re-export the hdf to include trials!")
            start_events = [b'TRIAL_START', b'TARGET_ON']
            end_events = [b'TRIAL_END']
            trial_events, trial_cycles = get_trial_segments(corrected_events['event'], corrected_events['time'], start_events, end_events)
            n_trials = len(trial_events)
            trial_idx = [np.where(trial_cycles >= idx)[0] for idx in range(len(task))] # needs testing

        corrected_trials = get_unique_conditions(trial_idx, task['target_location'])
    else:

        # TODO maybe should check if the last trial is incomplete
        corrected_trials = data['bmi3d_trials']
        n_trials = len(np.unique(corrected_trials['trial']))

    # Adjust for bmi3d time
    corrected_clock['timestamp'] -= bmi3d_start_time

    # Trim / pad everything to the same length
    n_cycles = corrected_clock['time'][-1]
    if metadata['sync_protocol_version'] >= 3 and metadata['sync_protocol_version'] < 6:

        # Due to the "sync" state at the beginning of the experiment, we need 
        # to add some (meaningless) cycles to the beginning of the clock
        state_log = data['bmi3d_state']
        n_sync_cycles = state_log['time'][1] # 120, approximately
        n_sync_clocks = np.count_nonzero(corrected_clock['time'] < n_sync_cycles)

        padded_clock = np.zeros((n_cycles,), dtype=corrected_clock.dtype)
        padded_clock[n_sync_cycles:] = corrected_clock[n_sync_clocks:]
        padded_clock['time'][:n_sync_cycles] = range(n_sync_cycles)
        # padded_clock['timestamp'][:n_sync_cycles] = np.arange(n_sync_cycles)/metadata['fps']
        corrected_clock = padded_clock
        
    # Update the event timestamps according to the corrected clock    
    if not metadata['has_measured_timestamps']:
        corrected_clock = corrected_clock[ [ name for name in corrected_clock.dtype.names if name not in 'timestamp' ] ] # remove 'timestamp'
    if 'timestamp_measure_offline' in corrected_clock.dtype.names:
        corrected_events = rfn.append_fields(corrected_events, 'timestamp_measure', corrected_clock['timestamp_measure_offline'][corrected_events['time']], dtypes='f8')
        corrected_events = rfn.append_fields(corrected_events, 'timestamp', corrected_events['timestamp_measure'], dtypes='f8')
    elif 'timestamp_sync' in corrected_events.dtype.names:
        corrected_events = rfn.append_fields(corrected_events, 'timestamp', corrected_events['timestamp_sync'], dtypes='f8')
    else:
        corrected_events = rfn.append_fields(corrected_events, 'timestamp', corrected_events['timestamp_bmi3d'], dtypes='f8')

    # Also put the reward system data into bmi3d time?
    if 'reward_system' in data and 'reward_system' in metadata['features']:
        metadata['has_reward_system'] = True
    else:
        metadata['has_reward_system'] = False

    data.update({
        'task': task,
        'state': state,
        'clock': corrected_clock,
        'events': corrected_events,
        'trials': corrected_trials,

    })
    metadata.update({
        'bmi3d_start_time': bmi3d_start_time,  
        'n_cycles': n_cycles,      
        'n_trials': n_trials,

    })
    return data, metadata

def parse_optitrack(data_dir, files):
    '''
    Parser for optitrack data

    Args:
        data_dir (str): where to look for the data
        files (dict): dictionary of files for this experiment
    
    Returns:
        tuple: tuple containing:

            data (dict): optitrack data
            metadata (dict): optitrack metadata
    '''
    # Check that there is optitrack data in files
    if not 'optitrack' in files:
        raise ValueError('Cannot parse nonexistent optitrack data!')

    # Load frame data
    optitrack_filename = files['optitrack']
    optitrack_metadata = load_optitrack_metadata(data_dir, optitrack_filename)
    optitrack_pos, optitrack_rot = load_optitrack_data(data_dir, optitrack_filename)

    # Load timing data from the ecube if present
    if 'ecube' in files:

        # Get the appropriate analog channel from bmi3d metadata
        try:
            _, bmi3d_event_metadata = load_bmi3d_hdf_table(data_dir, files['hdf'], 'sync_events')
            optitrack_strobe_channel = bmi3d_event_metadata['optitrack_sync_dch']
        except:
            optitrack_strobe_channel = 0

        # Load and parse the optitrack strobe signal
        digital_data, metadata = load_ecube_digital(data_dir, files['ecube'])
        samplerate = metadata['samplerate']
        optitrack_bit_mask = 1 << optitrack_strobe_channel
        optitrack_strobe = mask_and_shift(digital_data, optitrack_bit_mask)
        optitrack_strobe_timestamps, _ = detect_edges(optitrack_strobe, samplerate, rising=True, falling=False)
        # - check that eCube captured the same number of timestamps from esync as there are positions/rotations in the file
        if len(optitrack_pos) == len(optitrack_strobe_timestamps):
            optitrack_timestamps = optitrack_strobe_timestamps
            print("Optitrack strobes match exactly")
        # - otherwise assume they started at the same point, throw away or add zeros on the end if needed (throw a warning!)
        elif len(optitrack_pos) > len(optitrack_strobe_timestamps):
            n_extra = len(optitrack_pos) - len(optitrack_strobe_timestamps)
            print("{} too many optitrack positions recorded, truncating. Less than 50 is normal".format(n_extra))
            optitrack_pos = optitrack_pos[:len(optitrack_strobe_timestamps)]
            optitrack_rot = optitrack_rot[:len(optitrack_strobe_timestamps)]
            optitrack_timestamps = optitrack_strobe_timestamps
        # - optitrack has said they have issues getting the end of the recording to line up perfectly and to not worry about it :/
        else:
            n_extra = len(optitrack_strobe_timestamps) - len(optitrack_pos)
            print("{} too many optitrack strobe timestamps recorded, truncating. Less than 50 is normal".format(n_extra))
            optitrack_timestamps = optitrack_strobe_timestamps[:len(optitrack_pos)]
    
    # Otherwise just use the frame timing from optitrack
    else:
        print("Warning: using optitrack's internal timing")
        optitrack_timestamps = load_optitrack_time(data_dir, optitrack_filename)

    # Organize everything into dictionaries
    optitrack = np.empty((len(optitrack_timestamps),), dtype=[('timestamp', 'f8'), ('position', 'f8', (3,)), ('rotation', 'f8', (4,))])
    optitrack['timestamp'] = optitrack_timestamps
    optitrack['position'] = optitrack_pos
    optitrack['rotation'] = optitrack_rot
    data_dict = {
        'data': optitrack,
    }
    optitrack_metadata.update({
        'source_dir': data_dir,
        'source_files': files,
    }) 
    # TODO: add metadata about where the timestamps came from
    return data_dict, optitrack_metadata

def proc_exp(data_dir, files, result_dir, result_filename, overwrite=False):
    '''
    Process experiment data files: 
        Loads 'hdf' and 'ecube' (if present) data
        Parses 
    The above data is prepared into structured arrays:
        exp_data:
            task ([('cursor', '<f8', (3,)), ('trial', 'u8', (1,)), ('time', 'u8', (1,)), ...])
            state ([('msg', 'S', (1,)), ('time', 'u8', (1,))])
            clock ([('timestamp', 'f8', (1,)), ('time', 'u8', (1,))])
            events ([('timestamp', 'f8', (1,)), ('time', 'u8', (1,)), ('event', 'S32', (1,)), 
                ('data', 'u2', (1,)), ('code', 'u2', (1,))])
            trials ([('trial', 'u8'), ('index', 'u8'), (condition_name, 'f8', (3,)))])
        exp_metadata:
            source_dir (str)
            source_files (dict)
            bmi3d_start_time (float)
            n_cycles (int)
            n_trials (int)
            <other metadata from bmi3d>
    
    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool): whether to remove existing processed files if they exist

    Returns:
        None
    '''   
    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "exp_data" in contents or "exp_metadata" in contents:
            print("File {} already preprocessed, doing nothing.".format(result_filename))
            return
    
    # Prepare the BMI3D data
    if 'hdf' in files:
        bmi3d_data, bmi3d_metadata = parse_bmi3d(data_dir, files)
        save_hdf(result_dir, result_filename, bmi3d_data, "/exp_data", append=True)
        save_hdf(result_dir, result_filename, bmi3d_metadata, "/exp_metadata", append=True)

def proc_mocap(data_dir, files, result_dir, result_filename, overwrite=False):
    '''
    Process motion capture files:
        Loads metadata, position data, and rotation data from 'optitrack' files
        If present, reads 'hdf' metadata to find appropriate strobe channel
        If present, loads 'ecube' analog data representing optitrack camera strobe
    The data is prepared along with timestamps into HDF datasets:
        mocap_data:
            optitrack [('position', 'f8', (3,)), ('rotation', 'f8', (4,)), ('timestamp', 'f8', (1,)]
        mocap_metadata:
            source_dir (str)
            source_files (dict)
            samplerate (float)
            <other metadata from motive>
    
    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool): whether to remove existing processed files if they exist

    Returns:
        None
    '''  
    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "mocap_data" in contents or "mocap_metadata" in contents:
            print("File {} already preprocessed, doing nothing.".format(result_filename))
            return

    # Parse Optitrack data
    if 'optitrack' in files:
        optitrack_data, optitrack_metadata = parse_optitrack(data_dir, files)
        save_hdf(result_dir, result_filename, optitrack_data, "/mocap_data", append=True)
        save_hdf(result_dir, result_filename, optitrack_metadata, "/mocap_metadata", append=True)

def proc_lfp(data_dir, files, result_dir, result_filename, overwrite=False):
    '''
    Process lfp data:
        Loads 'ecube' headstage data and metadata
    Saves broadband data into the HDF datasets:
        Headstages (nt, nch)
    
    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool): whether to remove existing processed files if they exist

    Returns:
        None
    '''  
    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "Headstages" in contents:
            print("File {} already preprocessed, doing nothing.".format(result_filename))
            return

    # Preprocess neural data into lfp
    if 'ecube' in files:
        data_path = os.path.join(data_dir, files['ecube'])
        broadband = proc_ecube_data(data_path, 'Headstages', result_path)
        # TODO filter broadband data into LFP
