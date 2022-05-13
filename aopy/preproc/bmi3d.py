# bmi3d.py
# Code for parsing and preparing data from BMI3D

import numpy as np
import numpy.lib.recfunctions as rfn
from .base import get_measured_clock_timestamps, fill_missing_timestamps, get_trial_segments, get_unique_conditions
from .. import data as aodata
from .. import utils
import os

def parse_bmi3d(data_dir, files):
    '''
    Wrapper around version-specific bmi3d parsers

    Args:
        data_dir (str): where to look for the data
        files (dict): dictionary of files for this experiment
    
    Returns:
        tuple: tuple containing:
            | **data (dict):** bmi3d data
            | **metadata (dict):** bmi3d metadata
    '''
    # Check that there is hdf data in files
    if not 'hdf' in files:
        raise ValueError('Cannot parse nonexistent data!')

    # Load bmi3d data to see which sync protocol is used
    try:
        events, event_metadata = aodata.load_bmi3d_hdf_table(data_dir, files['hdf'], 'sync_events')
        sync_version = event_metadata['sync_protocol_version']
    except:
        sync_version = -1

    # Pass files onto the appropriate parser
    if sync_version < 7:
        data, metadata = _parse_bmi3d_v0(data_dir, files)
        metadata['bmi3d_parser'] = 0
        metadata['sync_protocol_version'] = sync_version

    elif sync_version < 11:
        data, metadata = _parse_bmi3d_v1(data_dir, files)
        metadata['bmi3d_parser'] = 1
    else:
        print("Warning: this bmi3d sync version is untested!")
        data, metadata = _parse_bmi3d_v1(data_dir, files)
        metadata['bmi3d_parser'] = 1

    # Standardize the parsed variable names and perform some error checking
    metadata['bmi3d_source'] = os.path.join(data_dir, files['hdf'])
    
    if sync_version >= 7:
        return _prepare_bmi3d_v1(data, metadata)
    else:
        return _prepare_bmi3d_v0(data, metadata)

def _parse_bmi3d_v0(data_dir, files):
    '''
    Simple parser for BMI3D data which basically ignores timing from the eCube.

    Args:
        data_dir (str): where to look for the data
        files (dict): dictionary of files for this experiment
    
    Returns:
        tuple: tuple containing:
            | **data (dict):** bmi3d data
            | **metadata (dict):** bmi3d metadata
    '''
    bmi3d_hdf_filename = files['hdf']
    bmi3d_hdf_full_filename = os.path.join(data_dir, bmi3d_hdf_filename)
    metadata = {}

    # Load bmi3d data
    bmi3d_task, bmi3d_task_metadata = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task')
    bmi3d_state, _ = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task_msgs')
    bmi3d_events, bmi3d_event_metadata = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'sync_events')
    bmi3d_root_metadata = aodata.load_bmi3d_root_metadata(data_dir, bmi3d_hdf_filename)
    if aodata.is_table_in_hdf('clda', bmi3d_hdf_full_filename): 
        bmi3d_clda, bmi3d_clda_meta = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'clda')
        metadata.update(bmi3d_clda_meta)

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

    if aodata.is_table_in_hdf('clda', bmi3d_hdf_full_filename): bmi3d_data.update(bmi3d_clda)
    return bmi3d_data, metadata

def _parse_bmi3d_v1(data_dir, files):
    '''
    Parser for BMI3D data which incorporates ecube data. Only compatible with sync versions > 0

    Args:
        data_dir (str): where to look for the data
        files (dict): dictionary of files for this experiment
    
    Returns:
        tuple: tuple containing:
            | **data_dict (dict):** bmi3d data
            | **metadata_dict (dict):** bmi3d metadata
    '''

    data_dict = {}
    metadata_dict = {}

    # Load bmi3d data
    bmi3d_hdf_filename = files['hdf']
    bmi3d_hdf_full_filename = os.path.join(data_dir, bmi3d_hdf_filename)
    
    bmi3d_task, bmi3d_task_metadata = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task')
    bmi3d_events, bmi3d_event_metadata = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'sync_events')

    sync_protocol_version = bmi3d_event_metadata['sync_protocol_version']
    bmi3d_sync_clock, _ = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'sync_clock') # there isn't any clock metadata
    bmi3d_root_metadata = aodata.load_bmi3d_root_metadata(data_dir, bmi3d_hdf_filename)

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
        'bmi3d_clock': bmi3d_sync_clock,
        'bmi3d_events': bmi3d_events,
    })  

    # Some data/metadata isn't always present
    if aodata.is_table_in_hdf('clda', bmi3d_hdf_full_filename): 
        bmi3d_clda, bmi3d_clda_meta = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'clda')
        metadata_dict.update(bmi3d_clda_meta)
        data_dict.update(
            {'bmi3d_clda': bmi3d_clda}
        )
    if aodata.is_table_in_hdf('task_msgs', bmi3d_hdf_full_filename): 
        bmi3d_state, _ = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task_msgs')
        data_dict.update(
            {'bmi3d_state': bmi3d_state}
        )
    if aodata.is_table_in_hdf('trials', bmi3d_hdf_full_filename): 
        bmi3d_trials, _ = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'trials')
        data_dict.update(
            {'bmi3d_trials': bmi3d_trials}
        )

    if 'ecube' in files:
        ecube_filename = files['ecube']
    
        # Load ecube digital data to find the strobe and events from bmi3d
        digital_data, metadata = aodata.load_ecube_digital(data_dir, ecube_filename)
        digital_samplerate = metadata['samplerate']

        # Load ecube analog data for the strobe and reward system
        analog_channels = [bmi3d_event_metadata['screen_measure_ach'], bmi3d_event_metadata['reward_measure_ach']] # [5, 0]
        ecube_analog, metadata = aodata.load_ecube_analog(data_dir, ecube_filename, channels=analog_channels)
        clock_measure_analog = ecube_analog[:,0]
        reward_system_analog = ecube_analog[:,1]
        analog_samplerate = metadata['samplerate']

        # Mask and detect BMI3D computer events from ecube
        event_bit_mask = utils.convert_channels_to_mask(bmi3d_event_metadata['event_sync_dch']) # 0xff0000
        ecube_sync_data = utils.mask_and_shift(digital_data, event_bit_mask)
        ecube_sync_timestamps, ecube_sync_events = utils.detect_edges(ecube_sync_data, digital_samplerate, rising=True, falling=False)
        sync_events = np.empty((len(ecube_sync_timestamps),), dtype=[('timestamp', 'f8'), ('code', 'u1')])
        sync_events['timestamp'] = ecube_sync_timestamps
        sync_events['code'] = ecube_sync_events
        if sync_protocol_version < 3:
            clock_sync_bit_mask = 0x1000000 # wrong in 1 and 2
        else:
            clock_sync_bit_mask = utils.convert_channels_to_mask(bmi3d_event_metadata['screen_sync_dch']) 
        clock_sync_data = utils.mask_and_shift(digital_data, clock_sync_bit_mask)
        clock_sync_timestamps, _ = utils.detect_edges(clock_sync_data, digital_samplerate, rising=True, falling=False)
        sync_clock = np.empty((len(clock_sync_timestamps),), dtype=[('timestamp', 'f8')])
        sync_clock['timestamp'] = clock_sync_timestamps

        # Mask and detect screen sensor events (A5 and D5)
        clock_measure_bit_mask = utils.convert_channels_to_mask(bmi3d_event_metadata['screen_measure_dch']) # 1 << 5
        clock_measure_data_online = utils.mask_and_shift(digital_data, clock_measure_bit_mask)
        clock_measure_timestamps_online, clock_measure_values_online = utils.detect_edges(clock_measure_data_online, digital_samplerate, rising=True, falling=True)
        measure_clock_online = np.empty((len(clock_measure_timestamps_online),), dtype=[('timestamp', 'f8'), ('value', 'f8')])
        measure_clock_online['timestamp'] = clock_measure_timestamps_online
        measure_clock_online['value'] = clock_measure_values_online
        clock_measure_digitized = utils.convert_analog_to_digital(clock_measure_analog, thresh=0.5)
        clock_measure_timestamps_offline, clock_measure_values_offline = utils.detect_edges(clock_measure_digitized, analog_samplerate, rising=True, falling=True)
        measure_clock_offline = np.empty((len(clock_measure_timestamps_offline),), dtype=[('timestamp', 'f8'), ('value', 'f8')])
        measure_clock_offline['timestamp'] = clock_measure_timestamps_offline
        measure_clock_offline['value'] = clock_measure_values_offline

        # And reward system (A0)
        reward_system_digitized = utils.convert_analog_to_digital(reward_system_analog)
        reward_system_timestamps, reward_system_values = utils.detect_edges(reward_system_digitized, analog_samplerate, rising=True, falling=True)
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

        metadata_dict.update({
            'digital_samplerate': digital_samplerate,
            'analog_samplerate': analog_samplerate,
        })
    return data_dict, metadata_dict

def _prepare_bmi3d_v0(data, metadata):
    '''
    Organizes the bmi3d data and metadata and computes some automatic conversions. Works on sync protocol
    versions 0 through 6. In these protocols there was a sync state at the beginning of each recording
    that contained a single long-duration clock pulse and matching screen pulse for measuring display
    latency. After this sync period, the digital clock arriving from BMI3D is reliable but often 
    truncated at the end of the recording.

    Args:
        data (dict): bmi3d data
        metadata (dict): bmi3d metadata

    Returns:
        tuple: tuple containing:
            | **data (dict):** prepared bmi3d data
            | **metadata (dict):** prepared bmi3d metadata
    '''
    assert metadata['sync_protocol_version'] < 7, \
        f"Sync protocol version {metadata['sync_protocol_version']} not supported"

    parser_version = metadata['bmi3d_parser']
    internal_clock = data['bmi3d_clock']
    internal_events = data['bmi3d_events']
    task = data['bmi3d_task']

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
        # This was changed in version 7 - now the sync period is just a slower clock rate,
        # so we can use those too.
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
        approx_clock['timestamp'] = corrected_clock['timestamp_sync'] # update the estimate using the sync clock
    else:
        print("Warning: no sync clock connected! This will usually result in problems.")

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
        n_consecutive_missing_cycles = utils.max_repeated_nans(timestamp_measure_online)
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
        n_consecutive_missing_cycles = utils.max_repeated_nans(timestamp_measure_offline)
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

    # Trim / pad everything to the same length
    n_cycles = int(corrected_clock['time'][-1]) + 1
    if metadata['sync_protocol_version'] >= 3 and metadata['sync_protocol_version'] < 6:

        # Due to the "sync" state at the beginning of the experiment, we need 
        # to add some (meaningless) cycles to the beginning of the clock
        state_log = data['bmi3d_state']
        n_sync_cycles = state_log['time'][1] # 120, approximately
        n_sync_clocks = np.count_nonzero(corrected_clock['time'] < n_sync_cycles)

        padded_clock = np.zeros((n_cycles,), dtype=corrected_clock.dtype)
        padded_clock[n_sync_cycles+1:] = corrected_clock[n_sync_clocks:]
        padded_clock['time'][:n_sync_cycles+1] = range(n_sync_cycles+1)
        corrected_clock = padded_clock

    # Update the clock to have a default 'timestamps' field
    if not metadata['has_measured_timestamps'] and 'timestamp_sync' in corrected_clock.dtype.names:
        corrected_clock['timestamp'] = corrected_clock['timestamp_sync']
    elif not metadata['has_measured_timestamps']:
        corrected_clock['timestamp'] = corrected_clock['timestamp_bmi3d']

    # Update the event timestamps according to the corrected clock    
    if metadata['has_measured_timestamps']:
        corrected_events = rfn.append_fields(corrected_events, 'timestamp_measure', corrected_clock['timestamp'][corrected_events['time']], dtypes='f8')
        corrected_events = rfn.append_fields(corrected_events, 'timestamp', corrected_events['timestamp_measure'], dtypes='f8')
    elif 'timestamp_sync' in corrected_events.dtype.names:
        corrected_events = rfn.append_fields(corrected_events, 'timestamp', corrected_events['timestamp_sync'], dtypes='f8')
    else:
        corrected_events = rfn.append_fields(corrected_events, 'timestamp', corrected_events['timestamp_bmi3d'], dtypes='f8')

    # Also put some information about the reward system
    if 'reward_system' in data and 'reward_system' in metadata['features']:
        metadata['has_reward_system'] = True
    else:
        metadata['has_reward_system'] = False

    data.update({
        'task': task,
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

def _prepare_bmi3d_v1(data, metadata):
    '''
    Organizes the bmi3d data and metadata and computes some automatic conversions. Version 1 for 
    bmi3d sync protocol 7 up to sync protocol 10. In these versions, the sync clock signal was 
    very unreliable, thus we do extra error correction to approximate accurate clock and event
    timestamps.

    Args:
        data (dict): bmi3d data
        metadata (dict): bmi3d metadata

    Returns:
        tuple: tuple containing:
            | **data (dict):** prepared bmi3d data
            | **metadata (dict):** prepared bmi3d metadata
    '''
    internal_clock = data['bmi3d_clock']
    internal_events = data['bmi3d_events']
    task = data['bmi3d_task']

    assert metadata['sync_protocol_version'] >= 7 and metadata['sync_protocol_version'] < 11, \
        f"Sync protocol version {metadata['sync_protocol_version']} not supported"

    # Estimate display latency
    if 'sync_clock' in data and 'measure_clock_offline' in data and len(data['sync_clock']) > 0:

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

        # Guess 10 ms
        measure_latency_estimate = 0.01
    metadata['measure_latency_estimate'] = measure_latency_estimate

    # By default use the internal clock and events.
    event_cycles = internal_events['time']
    event_timestamps = internal_clock['timestamp'][event_cycles]
    corrected_events = rfn.append_fields(internal_events, 'timestamp_bmi3d', event_timestamps, dtypes='f8')

    # Correct the clock
    corrected_clock = internal_clock.copy()
    corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_bmi3d', corrected_clock['timestamp'], dtypes='f8')
    approx_clock = corrected_clock['timestamp']
    valid_clock_cycles = len(corrected_clock)

    # 1. Digital clock from BMI3D via NI DIO card
    sync_search_radius = 1.5/metadata['fps']
    if 'sync_clock' in data and len(data['sync_clock']) > 0:
        sync_clock = data['sync_clock']
        if len(sync_clock) == 0:
            print("Warning: no clock timestamps on the eCube. Maybe something was unplugged?")
            print("Using internal clock timestamps")
        elif len(sync_clock) < len(internal_clock):
            print("Warning: length of clock timestamps on eCube ({}) doesn't match bmi3d record ({})".format(len(sync_clock), len(internal_clock)))
            valid_clock_cycles = len(sync_clock)
        elif len(sync_clock) > len(internal_clock):
            raise RuntimeError("Extra timestamps detected, something has gone horribly wrong.")

        # Adjust the internal clock so that it starts at the same time as the sync clock
        approx_clock = corrected_clock['timestamp'] + sync_clock['timestamp'][0] - corrected_clock['timestamp'][0]

        # Find sync clock pulses that match up to the expected internal clock timestamps within 1 radius
        timestamp_sync = get_measured_clock_timestamps(
            approx_clock, sync_clock['timestamp'], 0, sync_search_radius) # assume no latency between bmi3d and ecube via nidaq
        nanmask = np.isnan(timestamp_sync)
        # print(f"this many are NaN: {np.count_nonzero(nanmask)} out of {len(timestamp_sync)}")
        timestamp_sync[nanmask] = approx_clock[nanmask] # if nothing, then use the approximated value
        corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_sync', timestamp_sync, dtypes='f8')
    else:
        print("Warning: no sync clock connected! This will usually result in problems.")

    # 2. Screen photodiode measurements, digitized online by NXP microcontroller
    measure_search_radius = 1.5/metadata['fps']
    max_consecutive_missing_cycles = metadata['fps'] # maximum 1 second missing
    metadata['has_measured_timestamps'] = False
    if 'measure_clock_online' in data and len(data['measure_clock_online']) > 0:
        # Find the timestamps for each cycle of bmi3d's state machine from all the clock sources
        timestamp_measure_online = get_measured_clock_timestamps(
            approx_clock, data['measure_clock_online']['timestamp'], 
                measure_latency_estimate, measure_search_radius)
        corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_measure_online', timestamp_measure_online, dtypes='f8')

        # If there are few missing measurements, include this as the default `timestamp`
        corrected_timestamps = fill_missing_timestamps(timestamp_measure_online)
        metadata['latency_measured'] = np.nanmean(corrected_timestamps - timestamp_measure_online) - measure_latency_estimate
        metadata['n_missing_markers'] = np.count_nonzero(np.isnan(timestamp_measure_online[:valid_clock_cycles]))
        n_consecutive_missing_cycles = utils.max_repeated_nans(timestamp_measure_online[:valid_clock_cycles])
        if n_consecutive_missing_cycles < max_consecutive_missing_cycles:
            metadata['has_measured_timestamps'] = True
            corrected_clock['timestamp_measure_online'] = corrected_timestamps
        else:
            print(f"Digital screen sensor missing too many markers ({n_consecutive_missing_cycles}/{max_consecutive_missing_cycles}). Ignoring")

    # 3. Screen photodiode measurements, raw voltage digitized offline
    if 'measure_clock_offline' in data and len(data['measure_clock_offline']) > 0:
        timestamp_measure_offline = get_measured_clock_timestamps(
            approx_clock, data['measure_clock_offline']['timestamp'], 
                measure_latency_estimate, measure_search_radius)
        corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_measure_offline', timestamp_measure_offline, dtypes='f8')
        
        # If there are few missing measurements, include this as the default `timestamp`
        corrected_timestamps = fill_missing_timestamps(timestamp_measure_offline)
        metadata['latency_measured'] = np.nanmean(corrected_timestamps - timestamp_measure_offline) - measure_latency_estimate
        metadata['n_missing_markers'] = np.count_nonzero(np.isnan(timestamp_measure_offline[:valid_clock_cycles]))
        n_consecutive_missing_cycles = utils.max_repeated_nans(timestamp_measure_offline[:valid_clock_cycles])
        if n_consecutive_missing_cycles < max_consecutive_missing_cycles:
            corrected_clock['timestamp_measure_offline'] = corrected_timestamps
            metadata['has_measured_timestamps'] = True
        else:
            print(f"Analog screen sensor missing too many markers ({n_consecutive_missing_cycles}/{max_consecutive_missing_cycles}). Ignoring")

    # Update the clock to have a default 'timestamps' field
    if not metadata['has_measured_timestamps'] and 'timestamp_sync' in corrected_clock.dtype.names:
        corrected_clock['timestamp'] = corrected_clock['timestamp_sync']
    elif not metadata['has_measured_timestamps']:
        corrected_clock['timestamp'] = corrected_clock['timestamp_bmi3d']
    elif 'timestamp_measure_offline' in corrected_clock.dtype.names:
        corrected_clock['timestamp'] = corrected_clock['timestamp_measure_offline']
    elif 'timestamp_measure_online' in corrected_clock.dtype.names:
        corrected_clock['timestamp'] = corrected_clock['timestamp_measure_online']

    # Update the event timestamps according to the corrected clock    
    if metadata['has_measured_timestamps']:
        corrected_events = rfn.append_fields(corrected_events, 'timestamp_measure', corrected_clock['timestamp'][corrected_events['time']], dtypes='f8')
        corrected_events = rfn.append_fields(corrected_events, 'timestamp', corrected_events['timestamp_measure'], dtypes='f8')
    elif 'timestamp_sync' in corrected_clock.dtype.names:
        corrected_events = rfn.append_fields(corrected_events, 'timestamp_sync', corrected_clock['timestamp_sync'][corrected_events['time']], dtypes='f8')
        corrected_events = rfn.append_fields(corrected_events, 'timestamp', corrected_events['timestamp_sync'], dtypes='f8')
    else:
        corrected_events = rfn.append_fields(corrected_events, 'timestamp_approx', approx_clock[corrected_events['time']])
        corrected_events = rfn.append_fields(corrected_events, 'timestamp', corrected_events['timestamp_approx'], dtypes='f8')

    # Also put some information about the reward system
    if 'reward_system' in data and 'reward_system' in metadata['features']:
        metadata['has_reward_system'] = True
    else:
        metadata['has_reward_system'] = False

    data.update({
        'task': task,
        'clock': corrected_clock,
        'events': corrected_events,
    })
    return data, metadata