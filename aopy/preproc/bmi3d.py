# bmi3d.py
# Code for parsing and preparing data from BMI3D

import warnings
from matplotlib import pyplot as plt
import numpy as np
import os
from datetime import datetime
from importlib.metadata import version
import pandas as pd
import json
import sympy

from .. import precondition
from .. import data as aodata
from .. import utils
from .. import analysis
from .. import visualization
from . import base
from . import laser

def decode_event(dictionary, value):
    '''
    Decode a integer event code into a event name and data

    Args:
        dictionary (dict): dictionary of (event_name, event_code) event definitions
        value (int): number to decode

    Returns:
        tuple: 2-tuple containing (event_name, data) for the given value
    '''

    # Sort the dictionary in order of value
    ordered_list = sorted(dictionary.items(), key=lambda x: x[1])

    # Find a matching event (greatest value that is lower than the given value)
    for i, event in enumerate(ordered_list[1:]):
        if value < event[1]:
            event_name = ordered_list[i][0]
            event_data = value - ordered_list[i][1]
            return event_name, event_data

     # Check last value
    if value == ordered_list[-1][1]:
        return ordered_list[-1][0], 0

    # Return none if no matching events
    return None

def decode_events(dictionary, values):
    '''
    Decode a list of integer event code into a event names and data

    Args:
        dictionary (dict): dictionary of (event_name, event_code) event definitions
        values (n_values): list of integer numbers to decode

    Returns:
        tuple: 2-tuple containing (event_names, data) for the given values
    '''
    tuples = [decode_event(dictionary, value) for value in values]
    return list(zip(*tuples))

def _correct_hand_traj(hand_position, cursor_position):
    '''
    This function removes hand position data points when the cursor is simultaneously stationary in all directions.
    These hand position data points are artifacts. 
        
    Args:
        hand_position (nt, 3): Uncorrected hand position
        cursor_position (nt, 3): Cursor position from the same experiment, used to find where the hand position is invalid.
    
    Returns:
        hand_position (nt, 3): Corrected hand position
    '''

    # Set hand position to np.nan if the cursor position doesn't update. This indicates an optitrack error moved the hand outside the boundary.
    bad_pt_mask = np.zeros(cursor_position.shape, dtype=bool) 
    bad_pt_mask[1:,0] = (np.diff(cursor_position, axis=0)==0)[:,0] & (np.diff(cursor_position, axis=0)==0)[:,1] & (np.diff(cursor_position, axis=0)==0)[:,2]
    bad_pt_mask[1:,1] = (np.diff(cursor_position, axis=0)==0)[:,0] & (np.diff(cursor_position, axis=0)==0)[:,1] & (np.diff(cursor_position, axis=0)==0)[:,2]
    bad_pt_mask[1:,2] = (np.diff(cursor_position, axis=0)==0)[:,0] & (np.diff(cursor_position, axis=0)==0)[:,1] & (np.diff(cursor_position, axis=0)==0)[:,2]
    hand_position[bad_pt_mask] = np.nan

    return hand_position

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
    # Load bmi3d data to see which sync protocol is used
    try:
        events, event_metadata = aodata.load_bmi3d_hdf_table(data_dir, files['hdf'], 'sync_events')
        sync_version = event_metadata['sync_protocol_version']
    except:
        sync_version = -1

    # Pass files onto the appropriate parser
    if sync_version <= 0:
        data, metadata = _parse_bmi3d_v0(data_dir, files)
        metadata['bmi3d_parser'] = 0
        metadata['sync_protocol_version'] = sync_version

    elif sync_version < 14:
        data, metadata = _parse_bmi3d_v1(data_dir, files)
        metadata['bmi3d_parser'] = 1

    else:
        print("Warning: this bmi3d sync version is untested!")
        data, metadata = _parse_bmi3d_v1(data_dir, files)
        metadata['bmi3d_parser'] = 1
    
    # Keep track of the software version
    metadata['bmi3d_preproc_date'] = datetime.now()
    try:
        metadata['bmi3d_preproc_version'] = version('aolab-aopy')
    except:
        metadata['bmi3d_preproc_version'] = 'unknown'
    
    # And where the data came from
    try:
        metadata['bmi3d_source'] = os.path.join(data_dir, files['hdf'])
    except:
        metadata['bmi3d_source'] = None

    # Standardize the parsed variable names and perform some error checking
    return _prepare_bmi3d_v1(data, metadata)

def _parse_bmi3d_v0(data_dir, files):
    '''
    Simple parser for BMI3D data. Ignores eCube data.

    Args:
        data_dir (str): where to look for the data
        files (dict): dictionary of files for this experiment
    
    Returns:
        tuple: tuple containing:
            | **data (dict):** bmi3d data
            | **metadata (dict):** bmi3d metadata
    '''
    metadata = {}
    metadata['source_dir'] = data_dir
    metadata['source_files'] = files

    if 'hdf' not in files:
        warnings.warn("No hdf file found, cannot parse bmi3d data")
        return {}, metadata

    bmi3d_hdf_filename = files['hdf']
    bmi3d_hdf_full_filename = os.path.join(data_dir, bmi3d_hdf_filename)

    # Load bmi3d data
    bmi3d_task, bmi3d_task_metadata = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task')
    bmi3d_root_metadata = aodata.load_bmi3d_root_metadata(data_dir, bmi3d_hdf_filename)

    # Copy metadata
    metadata.update(bmi3d_task_metadata)
    metadata.update(bmi3d_root_metadata)

    # Put data into dictionary
    bmi3d_data = dict(
        bmi3d_task=bmi3d_task,
    )

    # Some data/metadata isn't always present
    if aodata.is_table_in_hdf('sync_events', bmi3d_hdf_full_filename):
        bmi3d_events, bmi3d_event_metadata = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'sync_events') # exists in tablet data
        metadata.update(bmi3d_event_metadata)
        bmi3d_data['bmi3d_events'] = bmi3d_events
    if aodata.is_table_in_hdf('clda', bmi3d_hdf_full_filename): 
        bmi3d_clda, bmi3d_clda_meta = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'clda')
        metadata.update(bmi3d_clda_meta)
        bmi3d_data['bmi3d_clda'] = bmi3d_clda
    if aodata.is_table_in_hdf('task_msgs', bmi3d_hdf_full_filename): 
        bmi3d_state, _ = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task_msgs')
        bmi3d_data['bmi3d_state'] = bmi3d_state
    if aodata.is_table_in_hdf('trials', bmi3d_hdf_full_filename): 
        bmi3d_trials, _ = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'trials')
        bmi3d_data['bmi3d_trials'] = bmi3d_trials
    if aodata.is_table_in_hdf('sync_clock', bmi3d_hdf_full_filename) and len(aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'sync_clock')[0])>0: # exists but empty in tablet data
        bmi3d_clock, _ = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'sync_clock') # there isn't any clock metadata
    else:
        # Estimate timestamps
        bmi3d_cycles = np.arange(len(bmi3d_task))
        bmi3d_timestamps = bmi3d_cycles/bmi3d_task_metadata['fps']
        bmi3d_clock = np.empty((len(bmi3d_task),), dtype=[('time', 'u8'), ('timestamp', 'f8')])
        bmi3d_clock['time'] = bmi3d_cycles
        bmi3d_clock['timestamp'] = bmi3d_timestamps
    bmi3d_data['bmi3d_clock'] = bmi3d_clock

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

    # Start by loading bmi3d data using the v0 parser
    data_dict, metadata_dict = _parse_bmi3d_v0(data_dir, files)

    if 'ecube' in files: # if not, there will be no sync_events or sync_clock in preproc data
        ecube_filename = files['ecube']
    
        # Load ecube digital data to find the strobe and events from bmi3d
        digital_data, metadata = aodata.load_ecube_digital(data_dir, ecube_filename)
        digital_samplerate = metadata['samplerate']

        # Load ecube analog data
        ecube_analog, metadata = aodata.load_ecube_analog(data_dir, ecube_filename)
        analog_samplerate = metadata['samplerate']

        # Mask and detect BMI3D computer events from ecube
        event_bit_mask = utils.convert_channels_to_mask(metadata_dict['event_sync_dch']) # 0xff0000
        ecube_sync_data = utils.extract_bits(digital_data, event_bit_mask)
        ecube_sync_timestamps, ecube_sync_events = utils.detect_edges(ecube_sync_data, digital_samplerate, 
            rising=True, falling=False)
        if len(ecube_sync_timestamps) > 2 and np.min(np.diff(ecube_sync_timestamps)) < metadata_dict['sync_pulse_width']:
            print(f"Correcting sync pulse width in {ecube_filename}")
            metadata['corrected_sync_pulse_width'] = True
            # There can occasionally be a compression of the pause event that smears it across multiple 
            # digital lines _-‾ and it shows up as multiple events very close together.
            ecube_sync_timestamps, ecube_sync_events = utils.detect_edges(ecube_sync_data, digital_samplerate, 
                rising=True, falling=False, min_pulse_width=metadata_dict['sync_pulse_width'])
        sync_event_names, sync_event_data = decode_events(metadata_dict['event_sync_dict'], ecube_sync_events)
        sync_events = np.empty((len(ecube_sync_timestamps),), dtype=[('timestamp', 'f8'), ('code', 'u1'), ('event', 'S32'), ('data', 'u4')])
        sync_events['timestamp'] = ecube_sync_timestamps
        sync_events['code'] = ecube_sync_events
        sync_events['event'] = sync_event_names
        sync_events['data'] = sync_event_data
        if metadata_dict['sync_protocol_version'] < 3:
            clock_sync_bit_mask = 0x1000000 # wrong in 1 and 2
        else:
            clock_sync_bit_mask = utils.convert_channels_to_mask(metadata_dict['screen_sync_dch']) 
        clock_sync_data = utils.extract_bits(digital_data, clock_sync_bit_mask)
        clock_sync_timestamps, _ = utils.detect_edges(clock_sync_data, digital_samplerate, rising=True, falling=False)
        sync_clock = np.empty((len(clock_sync_timestamps),), dtype=[('timestamp', 'f8')])
        sync_clock['timestamp'] = clock_sync_timestamps

        # Mask and detect screen sensor events (A5 and D5)
        measure_clock_online = base.get_dch_data(digital_data, digital_samplerate, metadata_dict['screen_measure_dch'])
        clock_measure_analog = ecube_analog[:, metadata_dict['screen_measure_ach']] # 5
        clock_measure_digitized = utils.convert_analog_to_digital(clock_measure_analog, thresh=0.5)
        measure_clock_offline = base.get_dch_data(clock_measure_digitized, analog_samplerate, 0)

        # And reward system (A0)
        reward_system_analog = ecube_analog[:, metadata_dict['reward_measure_ach']] # 0
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

        # Analog cursor out (A3, A4) since version 11
        if 'cursor_x_ach' in metadata_dict and 'cursor_z_ach' in metadata_dict:
            cursor_analog = ecube_analog[:, [metadata_dict['cursor_x_ach'], metadata_dict['cursor_z_ach']]]
            cursor_analog, _ = precondition.filter_kinematics(cursor_analog, samplerate=analog_samplerate)
            cursor_analog_samplerate = 1000
            cursor_analog = precondition.downsample(cursor_analog, analog_samplerate, cursor_analog_samplerate)
            max_voltage = 3.34 # using teensy 3.6
            cursor_analog_cm = ((cursor_analog * metadata['voltsperbit']) - max_voltage/2) / metadata_dict['cursor_out_gain']
            data_dict.update({
                'cursor_analog_volts': cursor_analog,
                'cursor_analog_cm': cursor_analog_cm,
            })
            metadata_dict['cursor_analog_samplerate'] = cursor_analog_samplerate

        metadata_dict.update({
            'digital_samplerate': digital_samplerate,
            'analog_samplerate': analog_samplerate,
            'analog_voltsperbit': metadata['voltsperbit']
        })

        # Laser sensors
        possible_ach = ['qwalor_sensor_ach', 'qwalor_ch1_sensor_ach', 'qwalor_ch2_sensor_ach', 
                        'qwalor_ch3_sensor_ach', 'qwalor_ch4_sensor_ach']
        possible_dch = ['qwalor_trigger_dch', 'qwalor_ch1_trigger_dch', 'qwalor_ch2_trigger_dch', 
                        'qwalor_ch3_trigger_dch', 'qwalor_ch4_trigger_dch']
        for ach in possible_ach:
            if ach in metadata_dict:
                sensor_name = ach[:-4]
                laser_sensor_data = ecube_analog[:, metadata_dict[ach]]
                data_dict[sensor_name] = laser_sensor_data
        for dch in possible_dch:
            if dch in metadata_dict:
                trigger_name = dch[:-4]
                data_dict[trigger_name] = base.get_dch_data(digital_data, digital_samplerate, metadata_dict[dch])

        # Optical switch
        if 'qwalor_switch_rdy_dch' in metadata_dict: 
            digital_samplerate = metadata['samplerate']
            switch_rdy_mask = utils.convert_channels_to_mask(metadata_dict['qwalor_switch_rdy_dch']) 
            ecube_switch_moving = utils.extract_bits(digital_data, switch_rdy_mask)
            switch_bit_mask = utils.convert_channels_to_mask(metadata_dict['qwalor_switch_data_dch']) # 0xff0000
            ecube_switch_data = utils.extract_bits(digital_data, switch_bit_mask) + 1 # change to 1-index
            masked_ecube_switch_data = ecube_switch_data.copy()
            masked_ecube_switch_data[ecube_switch_moving == 1] = 0 # mask data when the switch isn't ready
            ecube_switch_timestamps, ecube_switch_channel = utils.detect_edges(masked_ecube_switch_data, digital_samplerate, 
                rising=True, falling=True, check_alternating=False)
            optical_switch = np.empty((len(ecube_switch_timestamps),), dtype=[('timestamp', 'f8'), ('channel', 'u1')])
            optical_switch['timestamp'] = ecube_switch_timestamps
            optical_switch['channel'] = ecube_switch_channel # 1-indexed; positive is rising edge, zero is falling edge            
            data_dict['optical_switch'] = optical_switch

    return data_dict, metadata_dict

def _prepare_bmi3d_v1(data, metadata):
    '''
    Organizes the bmi3d data and metadata and computes some automatic conversions. Corrects for
    unreliable sync clock signal, finds measured timestamps, and pads the clock for versions
    with a sync period at the beginning of the experiment.

    Args:
        data (dict): bmi3d data
        metadata (dict): bmi3d metadata

    Returns:
        tuple: tuple containing:
            | **data (dict):** prepared bmi3d data
            | **metadata (dict):** prepared bmi3d metadata
    '''
    # Must be present: clock, task
    if 'bmi3d_clock' in data and 'bmi3d_task' in data:
        internal_clock = data['bmi3d_clock']
        task = data['bmi3d_task']
    elif 'sync_clock' in data:
        warnings.warn('Critical error! No internal clock found, using sync clock instead. Task data will be missing')
        internal_clock = data['sync_clock']
        task = None
    else:
        warnings.warn("No clock or task data found! Cannot prepare bmi3d data")
        return data, metadata
    
    # Estimate display latency
    if 'sync_clock' in data and 'measure_clock_offline' in data and len(data['sync_clock']) > 0:

        # Estimate the latency based on the "sync" state at the beginning of the experiment
        sync_impulse = data['sync_clock']['timestamp'][1:3]
        measure_impulse = base.get_measured_clock_timestamps(sync_impulse, data['measure_clock_offline']['timestamp'],
            latency_estimate=0.01, search_radius=0.1)
        if np.count_nonzero(np.isnan(measure_impulse)) > 0:
            warnings.warn("Warning: sync failed. Using latency estimate 0.01")
            measure_latency_estimate = 0.01
        else:
            measure_latency_estimate = np.mean(measure_impulse - sync_impulse)
            print("Sync latency estimate: {:.4f} s".format(measure_latency_estimate))
    else:
        measure_latency_estimate = 0.01 # Guess 10 ms
    metadata['measure_latency_estimate'] = measure_latency_estimate

    # Correct the clock
    cycle_bmi3d = internal_clock['time'].copy()
    timestamp_bmi3d = internal_clock['timestamp'].copy()
    corrected_clock = {
        'time': cycle_bmi3d,
        'timestamp_bmi3d': timestamp_bmi3d,
    }
    approx_clock = timestamp_bmi3d.copy()
    valid_clock_cycles = len(approx_clock)

    # 1. Digital clock from BMI3D via NI DIO card
    sync_search_radius = 1.5/metadata['fps']
    if 'sync_clock' in data and len(data['sync_clock']) > 0:
        sync_clock = data['sync_clock']
        if len(sync_clock) == 0:
            warnings.warn("Warning: no clock timestamps on the eCube. Maybe something was unplugged?")
            print("Using internal clock timestamps")
        elif len(sync_clock) < len(internal_clock):
            warnings.warn("Warning: length of clock timestamps on eCube ({}) doesn't match bmi3d record ({})".format(len(sync_clock), len(internal_clock)))
            valid_clock_cycles = len(sync_clock)
        elif len(sync_clock) > len(internal_clock):
            raise RuntimeError("Extra timestamps detected, something has gone horribly wrong.")

        # Adjust the internal clock so that it starts at the same time as the sync clock
        approx_clock = approx_clock + sync_clock['timestamp'][0] - approx_clock[0]

        # Find sync clock pulses that match up to the expected internal clock timestamps within 1 radius
        timestamp_sync = base.get_measured_clock_timestamps(
            approx_clock, sync_clock['timestamp'], 0, sync_search_radius) # assume no latency between bmi3d and ecube via nidaq
        nanmask = np.isnan(timestamp_sync)
        # print(f"this many are NaN: {np.count_nonzero(nanmask)} out of {len(timestamp_sync)}")
        timestamp_sync[nanmask] = approx_clock[nanmask] # if nothing, then use the approximated value
        corrected_clock['timestamp_sync'] = timestamp_sync
    else:
        warnings.warn("Warning: no sync clock connected! This will usually result in problems.")

    # 2. Screen photodiode measurements, digitized online by NXP microcontroller
    measure_search_radius = 1.5/metadata['fps']
    max_consecutive_missing_cycles = metadata['fps'] # maximum 1 second missing
    metadata['has_measured_timestamps'] = False
    if 'measure_clock_online' in data and len(data['measure_clock_online']) > 0:
        # Find the timestamps for each cycle of bmi3d's state machine from all the clock sources
        timestamp_measure_online = base.get_measured_clock_timestamps(
            approx_clock, data['measure_clock_online']['timestamp'], 
                measure_latency_estimate, measure_search_radius)
        corrected_clock['timestamp_measure_online'] = timestamp_measure_online

        # If there are few missing measurements, include this in the data
        metadata['latency_measured'] = np.nanmean(timestamp_measure_online - approx_clock)
        metadata['n_missing_markers'] = np.count_nonzero(np.isnan(timestamp_measure_online[:valid_clock_cycles]))
        n_consecutive_missing_cycles = utils.max_repeated_nans(timestamp_measure_online[:valid_clock_cycles])
        if n_consecutive_missing_cycles < max_consecutive_missing_cycles:
            metadata['has_measured_timestamps'] = True
        else:
            warnings.warn(f"Digital screen sensor missing too many markers ({n_consecutive_missing_cycles}/{max_consecutive_missing_cycles}). Ignoring")

    # 3. Screen photodiode measurements, raw voltage digitized offline
    if 'measure_clock_offline' in data and len(data['measure_clock_offline']) > 0:
        timestamp_measure_offline = base.get_measured_clock_timestamps(
            approx_clock, data['measure_clock_offline']['timestamp'], 
                measure_latency_estimate, measure_search_radius)
        
        # If there are few missing measurements, include this as the default `timestamp`
        metadata['latency_measured'] = np.nanmean(timestamp_measure_offline - approx_clock)
        metadata['n_missing_markers'] = np.count_nonzero(np.isnan(timestamp_measure_offline[:valid_clock_cycles]))
        n_consecutive_missing_cycles = utils.max_repeated_nans(timestamp_measure_offline[:valid_clock_cycles])
        if n_consecutive_missing_cycles < max_consecutive_missing_cycles:
            metadata['has_measured_timestamps'] = True
            corrected_clock['timestamp_measure_offline'] = timestamp_measure_offline
        else:
            warnings.warn(f"Analog screen sensor missing too many markers ({n_consecutive_missing_cycles}/{max_consecutive_missing_cycles}). Ignoring")

    # Assemble the corrected clock
    corrected_clock = pd.DataFrame.from_dict(corrected_clock).to_records(index=False)

    # Trim / pad the clock
    n_cycles = int(corrected_clock['time'][-1])
    if metadata['sync_protocol_version'] >= 3 and metadata['sync_protocol_version'] < 6:

        # Due to the "sync" state at the beginning of the experiment, we need 
        # to add some (meaningless) cycles to the beginning of the clock
        state_log = data['bmi3d_state']
        n_sync_cycles = state_log['time'][1] # 120, approximately
        n_sync_clocks = np.count_nonzero(corrected_clock['time'] < n_sync_cycles)

        padded_clock = np.zeros((n_cycles,), dtype=corrected_clock.dtype)
        padded_clock[n_sync_cycles:] = corrected_clock[n_sync_clocks:]
        padded_clock['time'][:n_sync_cycles] = range(n_sync_cycles)
        corrected_clock = padded_clock

    # By default use the internal events if they exist
    corrected_events = None
    if 'bmi3d_events' in data:
        corrected_events = np.empty((len(data['bmi3d_events']),), dtype=[('timestamp', 'f8'), ('code', 'u1'), ('event', 'S32'), ('data', 'u4')])
        corrected_events['timestamp'] =  np.asarray([timestamp_bmi3d[cycle] for cycle in data['bmi3d_events']['time']])
        corrected_events['code'] = data['bmi3d_events']['code']
        corrected_events['event'] = data['bmi3d_events']['event']
        try:
            corrected_events['data'] = data['bmi3d_events']['data']
        except:
            pass
        
    # But use the sync events if they exist and are valid
    if 'sync_events' in data and len(data['sync_events']) > 0:
        if not np.array_equal(data['sync_events']['code'], corrected_events['code']):
            warnings.warn("sync events don't match bmi3d events. This will probably cause problems.")
        corrected_events = data['sync_events']
    else:
        warnings.warn("No sync events present, using bmi3d events instead")

    data.update({
        'task': task,
        'clock': corrected_clock,
        'events': corrected_events,
    })

    # Also put some information about the reward system
    if 'reward_system' in data and 'reward_system' in metadata['features']:
        metadata['has_reward_system'] = True
    else:
        metadata['has_reward_system'] = False

    # In some versions of BMI3D, hand position contained erroneous data
    # caused by `np.empty()` instead of `np.nan`. The 'clean_hand_position' 
    # replaces these bad data with `np.nan`.
    if isinstance(task, np.ndarray) and 'manual_input' in task.dtype.names:
        clean_hand_position = _correct_hand_traj(task['manual_input'], task['cursor'])
        if np.count_nonzero(~np.isnan(clean_hand_position)) > 2*clean_hand_position.ndim:
            data['clean_hand_position'] = clean_hand_position

    # Interpolate clean hand kinematics
    if ('timestamp_sync' in corrected_clock.dtype.names and 
        'clean_hand_position' in data and
        len(data['clean_hand_position']) > 0):
        metadata['hand_interp_samplerate'] = 1000
        data['hand_interp'] = aodata.get_interp_task_data(data, metadata, datatype='hand', samplerate=metadata['hand_interp_samplerate'])

    # And interpolated cursor kinematics
    if ('timestamp_sync' in corrected_clock.dtype.names and 
        isinstance(task, np.ndarray) and 
        'cursor' in task.dtype.names and
        len(task['cursor']) > 0):
        metadata['cursor_interp_samplerate'] = 1000
        data['cursor_interp'] = aodata.get_interp_task_data(data, metadata, datatype='cursor', samplerate=metadata['cursor_interp_samplerate'])
        
    return data, metadata

def get_peak_power_mW(exp_metadata):
    """
    Estimate the peak power from the date

    Args:
        exp_metadata (dict): bmi3d metadata

    Returns:
        float: peak power in mW
    """
    date = datetime.fromisoformat(exp_metadata['date']).date()
    if 'qwalor_peak_watts' in exp_metadata:
        peak_power_mW = exp_metadata['qwalor_peak_watts']
    elif date < datetime(2022,5,31).date():
        if 'qwalor_channel' in exp_metadata and exp_metadata['qwalor_channel'] == 4:
            peak_power_mW = 1.5
        else:
            peak_power_mW = 20
    elif date < datetime(2022,9,30).date():
        peak_power_mW = 1.5
    elif date < datetime(2023,1,23).date():
        peak_power_mW = 20
    else:
        peak_power_mW = 25  

    return peak_power_mW
    
def _get_laser_trial_times_old_data(exp_data, exp_metadata, laser_sensor='qwalor_sensor', 
                                    calibration_file='qwalor_447nm_ch2.yaml', debug=False, **kwargs):
    '''
    Get the laser trial times, trial widths, and trial powers from the given experiment. Returned
    values are computed from the laser sensor in combination with the expected laser events from
    BMI3D's hdf records. Not recommended for use with experiments with sync_protocol_version > 12.

    Args:
        exp_data (dict): bmi3d data
        exp_metadata (dict): bmi3d metadata
        laser_sensor (str, optional): Specifies the name of the analog laser sensor
        calibration_file (str, optional): Specifies the name of the calibration file for the laser sensor
        debug (bool, optional): print a plot of the laser sensor aligned to the computed times
        kwargs (dict): to be passed to `:func:~aopy.preproc.laser.find_stim_times`

    Returns:
        tuple: tuple containing:
            | **corrected_times (nevent):** corrected laser timings (seconds)
            | **corrected_widths (nevent):** corrected laser widths (seconds)
            | **corrected_powers (nevent):** corrected laser powers (fraction of maximum)
            | **times_not_found (nevent):** boolean array of times without onset and offset sensor measurements
            | **widths_above_thr (nevent):** boolean array of widths above the given threshold from the expected width
            | **powers_above_thr (nevent):** boolean array of powers above the given threshold from the expected power
    '''
    if laser_sensor not in exp_data:
        raise ValueError(f"Could not find laser sensor data ({laser_sensor}). Try preprocessing the data first")

    # Some older experiments didn't have laser trigger data. Instead, we estimate
    # the laser event timing from BMI3D's sync events, then use the laser sensor to find
    # a more accurate time. However, in some cases the laser sensor data is too noisy so
    # we fall back on these inaccurate times.
    events = exp_data['sync_events']['event'] 
    event_times = exp_data['sync_events']['timestamp']
    times = event_times[events == b'TRIAL_START']
    min_isi = np.min(np.diff(times))
    if 'search_radius' not in kwargs:
        kwargs['search_radius'] = min_isi/2 # look for sensor measurements up to half the minimum ISI away

    # Get width and power from the 'trials' data
    if 'bmi3d_trials' in exp_data:
        trials = exp_data['bmi3d_trials']
        gains = trials['power'][:len(times)]
        edges = trials['edges'][:len(times)]
        widths = np.array([t[1] - t[0] for t in edges])
        thr_width=0.001
        thr_power=0.05
    else:
        # In the very old experiments, the laser width and power were not recorded. Since we
        # can't use the exp data as ground truth, so just trust the analog sensor data. Timing may be off.
        widths = np.zeros((len(times),))
        gains = np.ones((len(times),))
        thr_width = 999
        thr_power = 1

    # Correct the event timings using the sensor data
    sensor_data = exp_data[laser_sensor]
    sensor_voltsperbit = exp_metadata['analog_voltsperbit']
    samplerate = exp_metadata['analog_samplerate']   
    peak_power_mW = get_peak_power_mW(exp_metadata)
    (corrected_times, corrected_widths, corrected_powers, times_not_found, widths_above_thr, 
         powers_above_thr) = laser.find_stim_times(times, widths, gains, sensor_data, 
        samplerate, sensor_voltsperbit, peak_power_mW, thr_width=thr_width, 
        thr_power=thr_power, calibration_file=calibration_file, debug=debug, **kwargs)

    if np.sum(times_not_found) > 0:
        warnings.warn(f"{np.sum(times_not_found)} laser trials missing onset and/or offset sensor measurements")

    if np.sum(widths_above_thr) > 0:
        warnings.warn(f"{np.sum(widths_above_thr)} laser trials have widths above the given threshold")

    if np.sum(powers_above_thr) > 0:
        warnings.warn(f"{np.sum(powers_above_thr)} laser trials have powers above the given threshold")

    return corrected_times, corrected_widths, gains, corrected_powers

def _get_laser_trial_times(exp_data, exp_metadata, laser_trigger='qwalor_trigger', 
                           laser_sensor='qwalor_sensor', debug=False, **kwargs):
    '''
    Get the laser trial times, trial widths, and trial powers from the given experiment. Returned
    values are computed from the laser sensor in combination with the expected laser events from
    BMI3D's hdf records. Not recommended for use with experiments with sync_protocol_version > 12.

    Args:
        exp_data (dict): bmi3d data
        exp_metadata (dict): bmi3d metadata
        laser_trigger (str, optional): Specifies the name of the digital laser trigger
        laser_sensor (str, optional): Specifies the name of the analog laser sensor
        debug (bool, optional): print a plot of the laser sensor aligned to the computed times
        kwargs (dict): to be passed to `:func:~aopy.preproc.laser.find_stim_times`

    Returns:
        tuple: tuple containing:
            | **times (nevent):** laser timings (seconds)
            | **widths (nevent):** laser widths (seconds)
            | **gains (nevent):** laser gains (fraction)
            | **powers (nevent):** calibrated laser powers (mW)
    '''
    # Use the digital trigger as the ground truth of timing
    timestamps = exp_data[laser_trigger]['timestamp']
    values = exp_data[laser_trigger]['value']
    times = timestamps[values == 1]
    widths = timestamps[values == 0] - timestamps[values == 1]

    # Figure out the intended gain of each pulse
    if 'bmi3d_trials' in exp_data and 'power' in exp_data['bmi3d_trials'].dtype.names:
        trials = exp_data['bmi3d_trials']
        gains = trials['power'][:len(times)]
    elif 'laser_power' in exp_metadata:
        try:
            gains = np.ones((len(times),)) * exp_metadata['laser_power']
        except:
            gains = np.ones((len(times),))
    else:
        # In the very old experiments, the laser width and power were not recorded.
        gains = np.ones((len(times),))

    sensor_data = exp_data[laser_sensor]
    sensor_voltsperbit = exp_metadata['analog_voltsperbit']
    samplerate = exp_metadata['analog_samplerate']   

    laser_on_times = np.vstack([timestamps[values == 1], timestamps[values == 0]]).T
    laser_on_samples = (laser_on_times * samplerate).astype(int)

    laser_sensor_values = np.array([np.median(sensor_data[laser_on_samples[t,0]:laser_on_samples[t,1]]) 
                            for t in range(len(laser_on_samples))], dtype='float')
    
    # Estimate the peak power from the date
    peak_power_mW = get_peak_power_mW(exp_metadata)
    powers = laser.calibrate_sensor(laser_sensor_values * sensor_voltsperbit, peak_power_mW, **kwargs)  

    if debug:
        print(f"eCube recorded {len(times)} stims")

        plt.figure()
        visualization.plot_laser_sensor_alignment(sensor_data*sensor_voltsperbit, samplerate, times)

    return times, widths, gains, powers

def get_laser_trial_times(preproc_dir, subject, te_id, date, laser_trigger='qwalor_trigger', 
                          laser_sensor='qwalor_sensor', debug=False, **kwargs):
    '''
    Get the laser trial times, trial widths, and trial powers from the given experiment. Returned
    values are computed from the laser sensor in combination with the expected laser events from
    BMI3D's hdf records.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        laser_trigger (str, optional): Specifies the name of the digital laser trigger
        laser_sensor (str, optional): Specifies the name of the analog laser sensor
        kwargs (dict): to be passed to `:func:~aopy.preproc.laser.find_stim_times`

    Returns:
        tuple: tuple containing:
            | **times (nevent):** laser timings (seconds)
            | **widths (nevent):** laser widths (seconds)
            | **gains (nevent):** laser gains (fraction)
            | **powers (nevent):** calibrated laser powers (mW)
    '''
    exp_data, exp_metadata = aodata.load_preproc_exp_data(preproc_dir, subject, te_id, date)

    # Load the sensor data if it's not already in the bmi3d data
    if laser_sensor not in exp_data:
        files, data_dir = aodata.get_source_files(preproc_dir, subject, te_id, date)
        hdf_filepath = os.path.join(data_dir, files['hdf'])
        if not os.path.exists(hdf_filepath):
            raise FileNotFoundError(f"Could not find raw files for te {te_id} ({hdf_filepath})")
        exp_data, exp_metadata = parse_bmi3d(data_dir, files)
        
    # Experiments need to be handled differently depending on whether the trigger was recorded
    if laser_trigger in exp_data:

        # Return ground truth timestamps of when the laser should have been turned on
        return _get_laser_trial_times(exp_data, exp_metadata, laser_trigger=laser_trigger,
                                      laser_sensor=laser_sensor, debug=debug, **kwargs)
    else:

        # Use the bmi3d events as an estimate of timing, then locate the nearby sensor measurements
        return _get_laser_trial_times_old_data(exp_data, exp_metadata, laser_sensor=laser_sensor, 
                                               debug=debug, **kwargs)

def get_switched_stimulation_sites(preproc_dir, subject, te_id, date, trigger_timestamps, 
                                   return_switch_ch=False, debug=False):
    '''
    Get the stimulation sites at the given timestamps from an experiment where an optical switch was used.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object
        date (str): Date of recording
        trigger_timestamps (nt,): timestamps of interest
        return_switch_ch (bool, optional): also return the switch channel at the computed times
        debug (bool, optional): print a plot of the optical switch channel at the computed times
        
    Returns:
        (nt,): stimulation sites at the given timestamps, or np.nan if no site was selected.
    '''
    exp_data, exp_metadata = aodata.load_preproc_exp_data(preproc_dir, subject, te_id, date)

    # Check that the optical switch was present
    if 'qwalor_switch_rdy_dch' not in exp_metadata:
        raise ValueError("No optical switch data found in the experiment")

    # Load the optical switch data if it's not already in the bmi3d data
    if 'optical_switch' not in exp_data:
        files, data_dir = aodata.get_source_files(preproc_dir, subject, te_id, date)
        hdf_filepath = os.path.join(data_dir, files['hdf'])
        if not os.path.exists(hdf_filepath):
            raise FileNotFoundError(f"Could not find raw files for te {te_id} ({hdf_filepath})")
        exp_data, exp_metadata = parse_bmi3d(data_dir, files)

    optical_switch = exp_data['optical_switch']
    optical_switch_timestamps = optical_switch['timestamp']
    optical_switch_channels = optical_switch['channel']
    
    switch_channels = np.zeros((len(trigger_timestamps),), dtype='float')
    for i, t in enumerate(trigger_timestamps):

        # Find the most recent switch time before the trigger time
        idx = np.searchsorted(optical_switch_timestamps, t, side='right') - 1
        if idx < 0: # before the first switch
            switch_channels[idx] = optical_switch_channels[0]
        elif len(optical_switch_timestamps) > idx + 1 and optical_switch_timestamps[idx+1] < t:
            switch_channels[idx] = 0
        else:
            switch_channels[i] = optical_switch_channels[idx]
    switch_channels[switch_channels <= 0] = np.nan # no site selected
    switch_channels -= 1 # 1-indexed to 0-indexed

    stimulation_site = [exp_metadata['stimulation_site'][int(ch)] if not np.isnan(ch) else None for ch in switch_channels]

    if debug:
        plt.figure()
        plt.step(optical_switch_timestamps, optical_switch_channels, where='post')
        plt.plot(trigger_timestamps, switch_channels + 1, 'ro')
        for i, txt in enumerate(stimulation_site):
            plt.text(trigger_timestamps[i], switch_channels[i] + 1, txt, fontsize=6, ha='center', va='center', color='w')
        plt.xlabel('time (s)')
        plt.ylabel('switch channel (1-indexed)')

    if return_switch_ch:
        return stimulation_site, switch_channels
    else:
        return stimulation_site

def get_target_events(exp_data, exp_metadata):
    '''
    For target acquisition tasks, get an (n_event, n_target) array encoding the position
    of each target whenever an event is fired by BMI3D. The resulting sequence is used 
    to generate a sampled timeseries in :func:`~aopy.data.bmi3d.get_kinematic_segments`. 
    When targets are turned off, their position is replaced by np.nan. 

    Args:
        exp_data (dict): A dictionary containing the experiment data.
        exp_metadata (dict): A dictionary containing the experiment metadata.

    Returns:
        (n_event, n_target, 3) array: position of each target at each event time.
    '''
    
    events = exp_data['events']['code']
    trials = exp_data['bmi3d_trials']
    
    target_idx, location_idx = np.unique(trials['index'], axis=0, return_index=True)
    locations = [np.round(t[[0,2,1]], 4) for t in trials['target'][location_idx]]
    
    # Generate events for each unique target
    target_events = []
    for idx in range(len(locations)):
        target_on_codes = [
            exp_metadata['event_sync_dict']['TARGET_ON'] + target_idx[idx]
        ]
        target_off_codes = [
            exp_metadata['event_sync_dict']['TARGET_OFF'] + target_idx[idx], 
            exp_metadata['event_sync_dict']['TRIAL_END']
        ]

        target_location = locations[idx]
    
        # Create a nan mask encoding when the target is turned on
        target_on = np.zeros((len(events),))
        on = np.nan
        for idx, e in enumerate(events):
            if e in target_on_codes:
                on = 1
            elif e in target_off_codes:
                on = np.nan
            target_on[idx] = on
        
        # Set the non-nan values to the target location
        event_target = target_location[None,:] * target_on[:,None]    
        target_events.append(event_target)
        
    return np.array(target_events).transpose(1,0,2)

def get_ref_dis_frequencies(data, metadata):
    '''
    For continuous tracking tasks, get the set of frequencies (in Hz) used to 
    generate the reference and disturbance trajectories that were preesented 
    on each trial of the experiment.

    Note:
        This function should be used with caution on task entries that have mismatched sync and bmi3d events!
        Prior to 11-16-2022, bmi3d did not allow the number of experimental frequencies to be set by the experimenter, 
            and this parameter defaulted to 8.
        Prior to 2-23-2023, bmi3d did not save the generator index in the task data, and this had to be calculated 
            by the number of times bmi3d entered the 'wait' state.

    Args:
        data (dict): A dictionary containing the experiment data.
        metadata (dict): A dictionary containing the experiment metadata.

    Returns:
        tuple: Tuple containing:
            | **freq_r (list of arrays):** (ntrial) list of (nfreq,) frequencies used to generate reference trajectory
            | **freq_d (list of arrays):** (ntrial) list of (nfreq,) frequencies used to generate disturbance trajectory

    Examples:
        .. code-block:: python

            subject = 'test'
            te_id = '8461'
            date = '2023-02-25'

            data, metadata = load_preproc_exp_data(data_dir, subject, te_id, date)
            freq_r, freq_d = get_ref_dis_frequencies(data, metadata)

            plt.figure()
            plt.plot(freq_r, 'darkorange')
            plt.plot(freq_d, 'tab:red', linestyle='--')
            plt.xlabel('Trial #'); plt.ylabel('Frequency (Hz)')
            
        .. image:: _images/get_ref_dis_freqs_test.png
        
        .. code-block:: python

            subject = 'churro'
            te_id = '375'
            date = '2023-10-02'

            data, metadata = load_preproc_exp_data(data_dir, subject, te_id, date)
            freq_r, freq_d = get_ref_dis_frequencies(data, metadata)

            plt.figure()
            plt.plot(freq_r, 'darkorange')
            plt.plot(freq_d, 'tab:red', linestyle='--')
            plt.xlabel('Trial #'); plt.ylabel('Frequency (Hz)')

        .. image:: _images/get_ref_dis_freqs_churro.png

    '''

    # grab params relevant for generator
    params = json.loads(metadata['sequence_params'])
    if 'num_primes' not in params.keys():
        params['num_primes'] = 8
    primes = np.asarray(list(sympy.primerange(0, sympy.prime(params['num_primes'])+1)))
    even_idx = np.arange(len(primes))[0::2]
    odd_idx = np.arange(len(primes))[1::2]
    base_period = 20

    # recreate random trial order of reference & disturbance frequencies
    np.random.seed(params['seed'])
    o = np.random.rand(params['ntrials'],primes.size) # phase offset - need to generate this like in bmi3d to reproduce correct random order
    order = np.random.choice([0,1])
    if order == 0:
        trial_r_idx = np.array([even_idx, odd_idx]*params['ntrials'], dtype='object')
        trial_d_idx = np.array([odd_idx, even_idx]*params['ntrials'], dtype='object')
    elif order == 1:
        trial_r_idx = np.array([odd_idx, even_idx]*params['ntrials'], dtype='object')
        trial_d_idx = np.array([even_idx, odd_idx]*params['ntrials'], dtype='object')
        
    # get trial segments
    events = data['bmi3d_events']['code']
    cycles = data['bmi3d_events']['time'] # bmi3d cycle number

    start_codes = [metadata['event_sync_dict']['TARGET_ON']]
    if 'PAUSE_START' in metadata['event_sync_dict']:
        end_codes = [metadata['event_sync_dict']['TRIAL_END'], metadata['event_sync_dict']['PAUSE_START']]
    else:
        end_codes = [metadata['event_sync_dict']['TRIAL_END'], metadata['event_sync_dict']['PAUSE']]
    
    _, segment_cycles = base.get_trial_segments(events, cycles, start_codes, end_codes)

    # get trajectory generator index used for each trial
    if 'gen_idx' in data['task'].dtype.names:
        # get generator index from task data (saved on every bmi3d cycle) 
        generator_segments = np.array([data['task']['gen_idx'][cycle[0]:cycle[-1]] for cycle in segment_cycles], dtype='object')
        assert np.all([gen[0]==gen[-1] for gen in generator_segments]), 'Generator index is not consistent throughout trial segment!'

        generator_idx = [int(gen[0]) for gen in generator_segments]
        assert (np.diff(generator_idx) >= 0).all(), 'Generator index should stay the same or increase over trials, never decrease!'
    else:
        # get generator index from number of previous wait states (wait state parses next trial)
        states = data['bmi3d_state']['msg']
        state_cycles = data['bmi3d_state']['time'] # bmi3d cycle number
        generator_idx = [sum(states[state_cycles <= cycle[0]] == b'wait')-1 for cycle in segment_cycles]
        assert (np.diff(generator_idx) >= 0).all(), 'Generator index should stay the same or increase over trials, never decrease!'

    # use generator index to get reference & disturbance frequencies for each trial
    freq_r = [primes[np.array(idx, dtype=int)]/base_period for idx in trial_r_idx[generator_idx]]
    freq_d = [primes[np.array(idx, dtype=int)]/base_period for idx in trial_d_idx[generator_idx]]
    return freq_r, freq_d