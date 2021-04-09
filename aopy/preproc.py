# preproc.py
# code for preprocessing neural data
import numpy as np

'''
Digital calc
'''
def convert_analog_to_digital(analog_data, thresh=.3):
    '''
        This function takes analog data and converts it to digital data given a 
        threshold. It scales the analog to between 0 and 1 and uses thres as a 

        Inputs: 
        analog_data [nt, nch]: Time series array of analog data
        (opt) thresh [float]: Minimum threshold value to use in conversion

        Outputs:
        digital_data [nt, nch]: Array of 1's or 0's indicating if the analog input 
            was above threshold  
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

def detect_edges(digital_data, samplerate, lowhi=True, hilow=True):
    '''
    Finds the timestamp and corresponding value of all the bit flips in data. Assume 
    the first element in data isn't a transition

    Inputs:
        digital_data [ntime x 1]: masked binary data array
        samplerate [int]: sampling rate of the data used to calculate timestamps
        lowhi [bool]: include low to high transitions
        hilow [bool]: include high to low transitions

    Output:
        timestamps [nbitflips]: when the bits flipped
        values [nbitflips]: corresponding values for each change
    '''

    digital_data = np.squeeze(np.uint64(digital_data)) # important conversion for binary math
    logical_idx = np.zeros((len(digital_data)-1,), dtype='?')
    if lowhi:
        logical_idx |= (~digital_data[:-1] & digital_data[1:]) > 0 # find low->high transitions
    if hilow:
        logical_idx |= (~digital_data[1:] & digital_data[:-1]) > 0
    logical_idx = np.insert(logical_idx, 0, False) # first element never a transition
    time = np.arange(np.size(digital_data))/samplerate
    return time[logical_idx], digital_data[logical_idx]

def mask_and_shift(data, bit_mask):
    '''
    Apply bit mask and shift data to the least significant set bit in the mask. 
    For example,
    mask_and_shift(0001000011110000, 1111111100000000) => 00010000
    mask_and_shift(0001000011110000, 0000000011111111) => 11110000

    Inputs:
        data [ntime]: digital data
        bit_mask [int]: which bits to filter

    Output:
        data_ [ntime]: masked and shifted data
    '''

    return np.bitwise_and(data, bit_mask) >> find_first_significant_bit(bit_mask)

def find_first_significant_bit(x):
    '''
    Find first significant big. Returns the index, counting from 0, of the
    least significant set bit in x. Helper function for mask_and_shift

    Input:
        x [int]: a number

    Output:
        idx_fsb [int]: index of first significant nonzero bit
    '''
    return (x & -x).bit_length() - 1 # no idea how it works! thanks stack overflow --LRS

def convert_digital_to_channels(data_64_bit):
    '''
    Converts 64-bit digital data from eCube into channels.

    Inputs:
        data_64_bit [n]: masked 64-bit data, little-endian

    Outputs:
        unpacked [n, 64]: where channel 0 is least significant bit
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

    Inputs: 
        timestamp [float]: given timestamp
        sequence [nt]: sequence to search for closest value
        radius [float]: distance from timestamp to search for closest value

    Output:
        closest_value [float]: value within sequence that is closest to timestamp
        closest_idx [int]: index of the closest_value in the sequence
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
    Inputs:
        approx_times [nt]: array of approximate event timestamps
        measured_times [nt']: array of measured timestamps that might correspond to the approximate timestamps
        search_radius [float]: distance before and after each approximate time to search for a measured time 
        
    Output:
        parsed_ts	[nt]: array of the same length as approximate timestamps, 
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

def get_measured_frame_timestamps(estimated_timestamps, measured_timestamps, latency_estimate=0.01, search_radius=1./100):
    '''
    Takes estimated frame times and measured frame times and returns a time for each frame

    Inputs:
    estimated_timestamps [nframes]: timestamps when frames were thought to be displayed
    measured_timestamps [nt]: timestamps when frames actually appeared on screen
    (opt) latency_estimate [float]: how long the display takes normally to update
    (opt) search_radius [float]: how far away to look for a measurement before giving up

    Output:
    corrected_timestamps [nframes]: some of which will be empty if they were not displayed
    '''

    approx_timestamps = estimated_timestamps + latency_estimate
    uncorrected_timestamps = find_measured_event_times(approx_timestamps, measured_timestamps, search_radius)

    # For any missing timestamps, the time at which they occurred is the next non-nan value
    missing = np.isnan(uncorrected_timestamps)
    corrected_timestamps = uncorrected_timestamps.copy()
    if missing.any():
        idx_missing = np.where(missing)[0]
        for idx in idx_missing:
            next_non_nan = idx + 1
            while np.isnan(corrected_timestamps[next_non_nan]):
                if next_non_nan + 1 == len(corrected_timestamps):
                    # cannot correct the very last timestamp
                    break
                next_non_nan += 1
            corrected_timestamps[idx] = corrected_timestamps[next_non_nan]
    return corrected_timestamps, uncorrected_timestamps


'''
Event filtering
'''
def get_matching_events(event_log, event_to_match):
    '''
    Given a list of tuple of (events, timestamps), find the matched event and the timestamps
    
    INPUT:
    event_log [list]: a list of tuples (event[string or int] , time stamp: float)
    event_to_match [int or str]: event to be matched to
    
    OUTPUT
    returns a list of matched events and their time stamps
    '''
    #use python filter function to speed up the searching
    return list(filter(lambda k: k[0] == event_to_match, event_log) )

def get_event_occurrences(event_log, event_to_count):
    '''
    Given event_log, count the number of occurances of event_to_count

    Input:
    event_log [a list of tuples (event:string or int , time_stamp: float)]:
    event_to_count [int or str]: event to be matched to

    Output:
    num_occurances [int]
    '''
    matched_events_in_list = get_matching_events(event_log, event_to_count)
    num_occurances = len(matched_events_in_list)
    return num_occurances

def calc_events_duration(event_log):
    '''
    given an event_log and succuss_event,
    calculate the succuss rate

    INPUT:
    event_log [a list of tuples (event: str or int, time_stamp: float)]
    
    OUTPUT:
    events_duration [float]: 
    '''

    # Unpack the first and last events
    first_event_name, first_event_timestamp = event_log[0]
    last_event_name, last_event_timestamp = event_log[-1]

    # Take the difference between the timestamps
    events_duration = last_event_timestamp - first_event_timestamp
    return events_duration

def calc_event_rate(event_log, event_name):
    '''
    Given an event_log and event_name, calculate the rate of that event

    INPUT:
    event_log[a list of tuples (event: str or int, timestamp: float)]
    event_name[str or int]: event to be matched to

    OUTPUT:
    event_rate[float]: fraction of matching events divided by total events
    '''
    events_duration = calc_events_duration(event_log)
    num_of_events = get_event_occurrences(event_log, event_name)

    event_rate = float(num_of_events) / float(events_duration)
    return event_rate

def calc_reward_rate(event_log, event_name='REWARD'):
    '''
    A wrapper for calc_event_rate
    event_name defauls to be 'REWARD'

    INPUTS:
        event_log [a list of tuples (event: str or int, timestamp: float)]
        event_name [str or int]: event to be matched to

    OUTPUT:
        reward_rate

    '''
    return calc_event_rate(event_log, event_name)

def trial_separate(events, times, evt_start, n_events=8):
    '''
    Compute the 2D matrices contaning events per trial and timestamps per trial

    INPUT
        events [nt]: events vector
        times [nt]: times vector
        evt_start [int or str]: event marking the start of a trial
        n_events [int]: number of events in a trial
    
    OUTPUT:
        trial_events [n_trial, n_events]: events per trial
        trial_times [n_trial, n_events]: timestamps per trial
    '''

    # Find the indices in events that correspond to evt_start 
    evt_start_idx = np.where(events == evt_start)[0]

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
    Compute a new trial_times matrix with offset timestamps for the given event_to_align
    
    INPUT
        aligned_events [n_trial, n_event]: events per trial
        aligned_times [n_trial, n_event]: timestamps per trial
        event_to_align [int or str]: event to align to

    OUTPUT:
        trial_aligned_times [n_trial, n_event]: number of trials by number of events
    '''

    # For each row, find the column that matches the given event, 
    # then subtract its timestamps from the entire row
    trial_aligned_times = np.zeros(aligned_times.shape)
    for idx_trial in range(len(aligned_events)):
        idx_time = np.where(aligned_events[idx_trial,:] == event_to_align)[0]
        time_offset = aligned_times[idx_trial, idx_time]
        offset_row = aligned_times[idx_trial, :] - time_offset
        trial_aligned_times[idx_trial] = offset_row

    return trial_aligned_times

def trial_align_data(data, trigger_times, time_before, time_after, samplerate):
    '''
    Transform data into chunks of data triggered by trial start times

    Inputs:
        data [nt, nch]: arbitrary data, can be multidimensional
        trigger_times [ntrial]: start time of each trial
        time_before [float]: amount of time to include before the start of each trial
        time_after [float]: time to include after the start of each trial
        samplerate [int]: sampling rate of data
    
    Output:
        trial_aligned [ntrial, nt, nch]: trial aligned data
    '''
    dur = time_after + time_before
    n_samples = int(np.floor(dur * samplerate))

    if data.ndim > 1:
        trial_aligned = np.zeros((len(trigger_times), n_samples, *data.shape[1:]))
    else:
        trial_aligned = np.zeros((len(trigger_times), n_samples))
    for t in range(len(trigger_times)):
        t0 = trigger_times[t] - time_before
        if np.isnan(t0):
            continue
        sub = subvec(data, t0, n_samples, samplerate)
        if data.ndim > 1:
            trial_aligned[t,:min(len(sub),n_samples),:] = sub[:min(len(sub),n_samples)]
        else:
            trial_aligned[t,:min(len(sub),n_samples)] = sub[:min(len(sub),n_samples)]
    return trial_aligned

def trial_align_times(timestamps, trigger_times, time_before, time_after, subtract=True):
    '''
    Takes timestamps and splits them into chunks triggered by trial start times

    Inputs:
        timestamps [nt]: events in time to be trial aligned
        trigger_times [ntrial]: start time of each trial
        time_before [float]: amount of time to include before the start of each trial
        time_after [float]: time to include after the start of each trial
        (opt) subtract [bool]: whether the start of each trial should be set to 0
    
    Output:
        trial_aligned [ntrial, nt]: trial aligned timestamps
        trial_indices [ntrial, nt]: indices into timestamps in the same shape as trial_aligned
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

def subvec(vector, t0, n_samples, samplerate):
    '''
    Sub-vector helper function
    
    Input:
        vector [nt]: that you want to slice
        t0 [float]: start time
        n_samples [int]: number of samples to extract
        samplerate [int]: sampling rate of the vector

    Output:
        sub [n_samples]: vector of length n_samples starting at t0
    '''
    sub = np.empty((n_samples,))
    sub[:] = np.nan
    idx_start = int(np.floor(t0*samplerate))
    idx_end = min(len(vector), idx_start+n_samples)
    sub[:idx_end-idx_start] = vector[idx_start:idx_end]
    return sub
