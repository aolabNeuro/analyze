# preproc.py
# code for preprocessing neural data
import numpy as np
from .data import *
import pickle 

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


def get_trial_segments(events, times, start_events, end_events):
    '''
    Gets times for the start and end of each trial according to the given set of start_events and end_events

    Inputs:
        events [nt]: events vector
        times [nt]: times vector
        start_events [list]: set of start events to match
        end_events [list]: set of end events to match
    Output:
        segments [list of list of events]: a segment of each trial
        times [ntrials, 2]: list of 2 timestamps for each trial corresponding to the start and end events

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

    Inputs:
        data [nt, ndim]: arbitrary timeseries data that needs to segmented
        segment_times [nseg, 2] pairs of start and end times for each segment
        samplerate [int]: sampling rate of the data

    Output:
        segments [list of 1d arrays [nt]]: nt is the length of each segment (can be different for each)
    '''
    segments = []
    for idx_seg in range(segment_times.shape[0]):
        idx_data_start = int(segment_times[idx_seg,0]*samplerate)
        idx_data_end = int(segment_times[idx_seg,1]*samplerate)
        seg = data[idx_data_start:idx_data_end]
        segments.append(seg)
    return segments


'''
Prepare experiment files
'''
def parse_bmi3d(data_dir, files):
    '''
    Wrapper around version-specific bmi3d parsers
    '''
    # Check that there is hdf data in files
    if not 'bmi3d' in files:
        raise ValueError('Cannot parse nonexistent data!')

    # Load bmi3d data to see which sync protocol is used
    try:
        events, event_metadata = load_bmi3d_sync_events(data_dir, files['bmi3d'])
        sync_version = event_metadata['sync_protocol_version']
    except:
        sync_version = 0

    # Pass files onto the appropriate parser
    if sync_version == 0:
        return _parse_bmi3d_v0(data_dir, files)
    elif sync_version < 3:
        return _parse_bmi3d_v1(data_dir, files)
    else:
        raise NotImplementedError()

def _parse_bmi3d_v0(data_dir, files):
    '''
    Parser for v0 BMI3D data, which basically ignores timing from the eCube
    '''
    bmi3d_hdf_filename = files['bmi3d']
    bmi3d_metadata = {}

    # Load bmi3d data
    bmi3d_task, bmi3d_task_metadata = load_bmi3d_task(data_dir, bmi3d_hdf_filename)
    bmi3d_events, bmi3d_event_metadata = load_bmi3d_sync_events(data_dir, bmi3d_hdf_filename)
    with h5py.File(os.path.join(data_dir, bmi3d_hdf_filename), 'r') as f:
        bmi3d_root_metadata = f['/'].attrs
        for name in bmi3d_root_metadata:
            bmi3d_metadata['bmi3d_'+name] = bmi3d_root_metadata[name]
    for name in bmi3d_task_metadata:
        bmi3d_metadata['bmi3d_'+name] = bmi3d_task_metadata[name]
    for name in bmi3d_event_metadata:
        bmi3d_metadata['bmi3d_'+name] = bmi3d_event_metadata[name]

    # Estimate timestamps
    bmi3d_timestamps = np.arange(len(bmi3d_task))/bmi3d_task_metadata['fps']
    bmi3d_clock = np.array(bmi3d_timestamps, dtype=[('timestamp', 'f8')])

    # Rename 'time' to 'cycle_count'
    if bmi3d_events.dtype.names == ('time', 'event', 'data', 'code'):
        bmi3d_events.dtype.names = ('cycle_count', 'event', 'data', 'code')

    import numpy.lib.recfunctions as rfn
    bmi3d_cycles = rfn.merge_arrays((bmi3d_clock, bmi3d_task), flatten=True)
    bmi3d_data = dict(
        bmi3d_cycles=bmi3d_cycles,
        bmi3d_events=bmi3d_events,
    )
    return bmi3d_data, bmi3d_metadata

def _parse_bmi3d_v1(data_dir, files):
    '''
    Parser for v1 and v2 BMI3D data
    '''
    bmi3d_hdf_filename = files['bmi3d']
    ecube_filename = files['ecube']

    data_dict = {}
    metadata_dict = {}

    # Load bmi3d data
    bmi3d_task, bmi3d_task_metadata = load_bmi3d_task(data_dir, bmi3d_hdf_filename)
    bmi3d_params_dict = bmi3d_task_metadata
    bmi3d_events, bmi3d_event_metadata = load_bmi3d_sync_events(data_dir, bmi3d_hdf_filename)
    bmi3d_sync_version = bmi3d_event_metadata['sync_protocol_version']
    bmi3d_sync_clock, bmi3d_sync_clock_metadata = load_bmi3d_sync_clock(data_dir, bmi3d_hdf_filename) # there isn't any clock metadata
    bmi3d_trials, trials_metadata = load_bmi3d_trials(data_dir, bmi3d_hdf_filename) # there isn't any trial metadata
    bmi3d_event_metadata['event_sync_dict_pickled'] = str(pickle.dumps(bmi3d_event_metadata['event_sync_dict']))

    with h5py.File(os.path.join(data_dir, bmi3d_hdf_filename), 'r') as f:
        bmi3d_root_metadata = f['/'].attrs
        for name in bmi3d_root_metadata:
            metadata_dict['bmi3d_'+name] = bmi3d_root_metadata[name]
    for name in bmi3d_task_metadata:
        metadata_dict['bmi3d_'+name] = bmi3d_task_metadata[name]
    for name in bmi3d_event_metadata:
        metadata_dict['bmi3d_'+name] = bmi3d_event_metadata[name]

    # Load ecube digital data to find the strobe and events from bmi3d
    digital_data, metadata = load_ecube_digital(data_dir, ecube_filename)
    samplerate = metadata['samplerate']

    def convert_channels_to_mask(channels):
        '''
        Helper function to take a range of channels into a bitmask
        '''
        try:
            # Range of channels
            _ = iter(channels)
            flags = np.zeros(64, dtype=int)
            flags[channels] = 1
            return int(np.dot(np.array([2**i for i in range(1, 65)]), flags))
        except:
            
            # Single channel
            return int(1 << channels)

    # Load ecube analog data for the strobe and reward system
    analog_channels = [bmi3d_event_metadata['screen_measure_ach'], bmi3d_event_metadata['reward_measure_ach']] # [5, 0]
    ecube_analog, metadata = load_eCube_analog(data_dir, ecube_filename, channels=analog_channels)
    screen_strobe_analog = ecube_analog[0,:]
    reward_system_analog = ecube_analog[1,:]

    # Mask and detect BMI3D computer events from ecube
    event_bit_mask = convert_channels_to_mask(bmi3d_event_metadata['event_sync_dch']) # 0xff0000 
    ecube_sync_data = mask_and_shift(digital_data, event_bit_mask)
    ecube_sync_timestamps, ecube_sync_events = detect_edges(ecube_sync_data, samplerate, lowhi=True, hilow=False)
    bmi3d_clock_mask = convert_channels_to_mask(bmi3d_event_metadata['screen_sync_dch']) # 0x1000000
    bmi3d_clock_data = mask_and_shift(digital_data, bmi3d_clock_mask)
    bmi3d_clock_timestamps, _ = detect_edges(bmi3d_clock_data, samplerate, lowhi=True, hilow=False)

    # Mask and detect screen sensor events (A5 and D5)
    screen_strobe_mask = convert_channels_to_mask(bmi3d_event_metadata['screen_measure_dch']) # 1 << 5
    screen_strobe = mask_and_shift(digital_data, screen_strobe_mask)
    screen_strobe_timestamps, screen_strobe_values = detect_edges(screen_strobe, samplerate, lowhi=True, hilow=True)
    screen_strobe_digitized = convert_analog_to_digital(screen_strobe_analog)
    screen_strobe_timestamps_offline, screen_strobe_values_offline = detect_edges(screen_strobe_digitized, samplerate, lowhi=True, hilow=True)

    # And reward system (A0)
    reward_system_digitized = convert_analog_to_digital(reward_system_analog)
    samplerate = metadata['samplerate']
    reward_system_timestamps, reward_system_values = detect_edges(reward_system_digitized, samplerate, lowhi=True, hilow=True)
    
    # Error checking / fixing some missing data

    # Find the (correct) timestamps for each cycle of bmi3d's state machine --> use to sync cursor position
    # - Check that the events are all present
    if ecube_sync_events[0] != bmi3d_events['code'][0]:
        print("Warning: first event ({}) doesn't match bmi3d records ({})".format(ecube_sync_events[0], bmi3d_events['code'][0]))
    if ecube_sync_events[-1] != bmi3d_events['code'][-1]:
        print("Warning: last event ({}) doesn't match bmi3d records ({})".format(ecube_sync_events[-1], bmi3d_events['code'][-1]))
    # - Check that the number of frames is consistent with the clock
    bmi3d_internal_clock_timestamps = bmi3d_sync_clock['timestamp']
    if len(bmi3d_clock_timestamps) < len(bmi3d_internal_clock_timestamps):
        print("Warning: length of clock timestamps on eCube ({}) doesn't match bmi3d record ({})".format(len(bmi3d_clock_timestamps), len(bmi3d_task)))
        print("Adding internal clock timestamps to the end of the recording...")
        tmp = bmi3d_clock_timestamps.copy()
        bmi3d_clock_timestamps = bmi3d_internal_clock_timestamps.copy()
        bmi3d_clock_timestamps[:len(tmp)] = tmp
    elif len(bmi3d_clock_timestamps) > len(bmi3d_task):
        raise NotImplementedError()
    n_cycles = len(bmi3d_clock_timestamps)
    # - Although most of the time the screen should update when the clock cycles, it won't always
    latency_estimate = 0.01
    search_radius = 0.01
    bmi3d_external_clock_timestamps, bmi3d_external_clock_timestamps_uncorrected = get_measured_frame_timestamps(
        bmi3d_clock_timestamps, screen_strobe_timestamps_offline, 
        latency_estimate, search_radius)
    # - Add some statistics about the timing
    n_missing_markers = np.count_nonzero(np.isnan(bmi3d_external_clock_timestamps_uncorrected))
    fraction_missing = n_missing_markers/n_cycles
    print("Fraction missing markers: {:.2}".format(fraction_missing))
    measured_diff = bmi3d_external_clock_timestamps_uncorrected - bmi3d_clock_timestamps
    latency = np.mean(measured_diff[~np.isnan(measured_diff)])
    print("Estimated display latency: {:.2} s (refresh period is {:.2} s)".format(latency, 1./bmi3d_task_metadata['fps']))

    # Organize the data a bit
    # - Subtract bmi3d time 0 from all the timestamps
    event_exp_start = bmi3d_events[bmi3d_events['event'] == b'EXP_START']
    bmi3d_start_time = ecube_sync_timestamps[ecube_sync_events == event_exp_start['code']]
    if len(bmi3d_start_time) == 0:
        bmi3d_start_time = bmi3d_external_clock_timestamps[0]
    elif len(bmi3d_start_time) > 1:
        bmi3d_start_time = bmi3d_start_time[0] # why are there sometimes two????
    bmi3d_timestamps = bmi3d_external_clock_timestamps - bmi3d_start_time
    # - Rename 'time' to 'cycle_count'
    if bmi3d_events.dtype.names == ('time', 'event', 'data', 'code'):
        bmi3d_events.dtype.names = ('cycle_count', 'event', 'data', 'code')
    if bmi3d_sync_clock.dtype.names == ('time', 'timestamp', 'prev_tick'):
        bmi3d_sync_clock.dtype.names = ('cycle_count', 'timestamp', 'prev_tick')
    # - Add the timestamps to the task data array
    import numpy.lib.recfunctions as rfn
    bmi3d_sync_clock['timestamp'] = bmi3d_timestamps
    bmi3d_cycles = rfn.merge_arrays((bmi3d_sync_clock, bmi3d_task), flatten=True)
    # - Also put the reward system data into bmi3d time
    reward_system = np.empty((len(reward_system_timestamps),), dtype=[('timestamp', 'f8'), ('state', '?')])
    reward_system['timestamp'] = reward_system_timestamps
    reward_system['state'] = reward_system_values

    # Wrap everything up in dictionaries
    data_dict.update({
        'bmi3d_cycles': bmi3d_cycles,
        'bmi3d_events': bmi3d_events,
        'bmi3d_trials': bmi3d_trials,
        'reward_system': reward_system,
    })
    metadata_dict.update({
        'measured_display_latency': latency,
        'missing_markers': n_missing_markers,
        'marker_search_radius': search_radius,
        'marker_latency_estimate': latency_estimate,
        'bmi3d_n_cycles': n_cycles,
        'bmi3d_start_time': bmi3d_start_time,
    })
    return data_dict, metadata_dict

def parse_optitrack(data_dir, files):
    '''
    Parser for optitrack data

    Inputs:
        data_dir [str]: where to look for the data
        files [dict]: dictionary of files for this experiment
    
    Outputs:
        data [dict]: optitrack data
        metadata [dict]: optitrack metadata
    '''
    # Check that there is hdf data in files
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
            bmi3d_events, bmi3d_event_metadata = load_bmi3d_sync_events(data_dir, files['bmi3d'])
            optitrack_strobe_channel = bmi3d_event_metadata['optitrack_sync_dch']
        except:
            optitrack_strobe_channel = 0

        # Load and parse the optitrack strobe signal
        digital_data, metadata = load_ecube_digital(data_dir, files['ecube'])
        samplerate = metadata['samplerate']
        optitrack_bit_mask = 1 << optitrack_strobe_channel
        optitrack_strobe = mask_and_shift(digital_data, optitrack_bit_mask)
        optitrack_strobe_timestamps, optitrack_strobe_values = detect_edges(optitrack_strobe, samplerate, lowhi=True, hilow=False)
        # - check that eCube captured the same number of timestamps from esync as there are positions/rotations in the file
        if len(optitrack_pos) == len(optitrack_strobe_timestamps):
            optitrack_timestamps = optitrack_strobe_timestamps
            print("Optitrack strobes match exactly")
        # - otherwise assume they started at the same point, throw away or add zeros on the end if needed (throw a warning!)
        elif len(optitrack_pos) > len(optitrack_strobe_timestamps):
            print("Too many optitrack positions recorded, truncating. This is normal")
            optitrack_pos = optitrack_pos[:len(optitrack_strobe_timestamps)]
            optitrack_rot = optitrack_rot[:len(optitrack_strobe_timestamps)]
            optitrack_timestamps = optitrack_strobe_timestamps
        # - optitrack has said they have issues getting the end of the recording to line up perfectly and to not worry about it :/
        else:
            print("Too many optitrack strobe timestamps recorded, truncating. This is normal")
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
    data_dict = {'optitrack': optitrack}
    metadata_dict = dict([('optitrack_'+k, v) for k, v in optitrack_metadata.items()])
    metadata_dict['has_optitrack_data'] = True
    return data_dict, metadata_dict

def proc_exp(data_dir, files, result_dir, result_filename, overwrite=True):
    '''
    Process experiment data files
    
    Inputs:
        files [dict]: dictionary of filenames indexed by system
        hdf_filename [str]: where to store the prepared data

    Output:
        None
    '''   
    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if overwrite and os.path.exists(filepath):
        os.remove(filepath)
    elif os.path.exists(filepath):
        print("File {} already exists, doing nothing.".format(result_filename))
        return
    
    # Then process the data if needed
    data_dict = {}
    metadata_dict = {}

    # Prepare the BMI3D data
    if 'bmi3d' in files:
        bmi3d_data, bmi3d_metadata = parse_bmi3d(data_dir, files)
        data_dict.update(bmi3d_data)
        metadata_dict.update(bmi3d_metadata)

    # Parse Optitrack data
    if 'optitrack' in files:
        optitrack_data, optitrack_metadata = parse_optitrack(data_dir, files)
        data_dict.update(optitrack_data)
        metadata_dict.update(optitrack_metadata)

    # Save everything to an HDF file
    save_hdf(result_dir, result_filename, data_dict, metadata_dict, append=False)
