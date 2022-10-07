# bmi3d.py
# Code for parsing and preparing data from BMI3D

import warnings
import numpy as np
import numpy.lib.recfunctions as rfn
import os

from .. import precondition
from .. import data as aodata
from .. import utils
from ..postproc import get_source_files
from ..utils import detect_edges
from .base import get_measured_clock_timestamps, find_measured_event_times, validate_measurements, interp_timestamps2timeseries

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
    if sync_version <= 0:
        data, metadata = _parse_bmi3d_v0(data_dir, files)
        metadata['bmi3d_parser'] = 0
        metadata['sync_protocol_version'] = sync_version

    elif sync_version < 12:
        data, metadata = _parse_bmi3d_v1(data_dir, files)
        metadata['bmi3d_parser'] = 1
    else:
        print("Warning: this bmi3d sync version is untested!")
        data, metadata = _parse_bmi3d_v1(data_dir, files)
        metadata['bmi3d_parser'] = 1

    # Standardize the parsed variable names and perform some error checking
    metadata['bmi3d_source'] = os.path.join(data_dir, files['hdf'])
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
    bmi3d_hdf_filename = files['hdf']
    bmi3d_hdf_full_filename = os.path.join(data_dir, bmi3d_hdf_filename)
    metadata = {}

    # Load bmi3d data
    bmi3d_task, bmi3d_task_metadata = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'task')
    bmi3d_root_metadata = aodata.load_bmi3d_root_metadata(data_dir, bmi3d_hdf_filename)

    # Copy metadata
    metadata.update(bmi3d_task_metadata)
    metadata.update(bmi3d_root_metadata)
    metadata.update({
        'source_dir': data_dir,
        'source_files': files,
    }) 

    # Put data into dictionary
    bmi3d_data = dict(
        bmi3d_task=bmi3d_task,
    )

    # Some data/metadata isn't always present
    if aodata.is_table_in_hdf('sync_events', bmi3d_hdf_full_filename):
        bmi3d_events, bmi3d_event_metadata = aodata.load_bmi3d_hdf_table(data_dir, bmi3d_hdf_filename, 'sync_events')
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
    if aodata.is_table_in_hdf('sync_clock', bmi3d_hdf_full_filename): 
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

    if 'ecube' in files:
        ecube_filename = files['ecube']
    
        # Load ecube digital data to find the strobe and events from bmi3d
        digital_data, metadata = aodata.load_ecube_digital(data_dir, ecube_filename)
        digital_samplerate = metadata['samplerate']

        # Load ecube analog data
        ecube_analog, metadata = aodata.load_ecube_analog(data_dir, ecube_filename)
        analog_samplerate = metadata['samplerate']

        # Mask and detect BMI3D computer events from ecube
        event_bit_mask = utils.convert_channels_to_mask(metadata_dict['event_sync_dch']) # 0xff0000
        ecube_sync_data = utils.mask_and_shift(digital_data, event_bit_mask)
        ecube_sync_timestamps, ecube_sync_events = utils.detect_edges(ecube_sync_data, digital_samplerate, rising=True, falling=False)
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
        clock_sync_data = utils.mask_and_shift(digital_data, clock_sync_bit_mask)
        clock_sync_timestamps, _ = utils.detect_edges(clock_sync_data, digital_samplerate, rising=True, falling=False)
        sync_clock = np.empty((len(clock_sync_timestamps),), dtype=[('timestamp', 'f8')])
        sync_clock['timestamp'] = clock_sync_timestamps

        # Mask and detect screen sensor events (A5 and D5)
        clock_measure_bit_mask = utils.convert_channels_to_mask(metadata_dict['screen_measure_dch']) # 1 << 5
        clock_measure_data_online = utils.mask_and_shift(digital_data, clock_measure_bit_mask)
        clock_measure_timestamps_online, clock_measure_values_online = utils.detect_edges(clock_measure_data_online, digital_samplerate, rising=True, falling=True)
        measure_clock_online = np.empty((len(clock_measure_timestamps_online),), dtype=[('timestamp', 'f8'), ('value', 'f8')])
        measure_clock_online['timestamp'] = clock_measure_timestamps_online
        measure_clock_online['value'] = clock_measure_values_online
        clock_measure_analog = ecube_analog[:, metadata_dict['screen_measure_ach']] # 5
        clock_measure_digitized = utils.convert_analog_to_digital(clock_measure_analog, thresh=0.5)
        clock_measure_timestamps_offline, clock_measure_values_offline = utils.detect_edges(clock_measure_digitized, analog_samplerate, rising=True, falling=True)
        measure_clock_offline = np.empty((len(clock_measure_timestamps_offline),), dtype=[('timestamp', 'f8'), ('value', 'f8')])
        measure_clock_offline['timestamp'] = clock_measure_timestamps_offline
        measure_clock_offline['value'] = clock_measure_values_offline

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
            max_voltage = 3.34 # using teensy 3.6
            cursor_analog_cm = ((cursor_analog * metadata['voltsperbit']) - max_voltage/2) / metadata_dict['cursor_out_gain']
            filt_out, freq = precondition.butterworth_filter_data(cursor_analog_cm, analog_samplerate, [metadata_dict['fps']], filter_type='lowpass', order=3)
            cursor_analog_cm_filt = filt_out[0]
            data_dict.update({
                'cursor_analog_volts': cursor_analog,
                'cursor_analog_cm': cursor_analog_cm,
                'cursor_analog_cm_filt': cursor_analog_cm_filt,
            })
            metadata_dict['cursor_analog_samplerate'] = analog_samplerate

        metadata_dict.update({
            'digital_samplerate': digital_samplerate,
            'analog_samplerate': analog_samplerate,
            'analog_voltsperbit': metadata['voltsperbit']
        })

        # Laser sensor (A2)
        if 'qwalor_sensor_ach' in metadata_dict:
            laser_sensor_data = ecube_analog[:, metadata_dict['qwalor_sensor_ach']]
            data_dict['laser_sensor'] = laser_sensor_data

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
    internal_clock = data['bmi3d_clock']
    task = data['bmi3d_task']

    # Estimate display latency
    if 'sync_clock' in data and 'measure_clock_offline' in data and len(data['sync_clock']) > 0:

        # Estimate the latency based on the "sync" state at the beginning of the experiment
        sync_impulse = data['sync_clock']['timestamp'][1:3]
        measure_impulse = get_measured_clock_timestamps(sync_impulse, data['measure_clock_offline']['timestamp'],
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
    corrected_clock = internal_clock.copy()
    corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_bmi3d', corrected_clock['timestamp'], dtypes='f8')
    approx_clock = corrected_clock['timestamp']
    valid_clock_cycles = len(corrected_clock)

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
        approx_clock = corrected_clock['timestamp'] + sync_clock['timestamp'][0] - corrected_clock['timestamp'][0]

        # Find sync clock pulses that match up to the expected internal clock timestamps within 1 radius
        timestamp_sync = get_measured_clock_timestamps(
            approx_clock, sync_clock['timestamp'], 0, sync_search_radius) # assume no latency between bmi3d and ecube via nidaq
        nanmask = np.isnan(timestamp_sync)
        # print(f"this many are NaN: {np.count_nonzero(nanmask)} out of {len(timestamp_sync)}")
        timestamp_sync[nanmask] = approx_clock[nanmask] # if nothing, then use the approximated value
        corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_sync', timestamp_sync, dtypes='f8')
    else:
        warnings.warn("Warning: no sync clock connected! This will usually result in problems.")

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

        # If there are few missing measurements, include this in the data
        metadata['latency_measured'] = np.nanmean(timestamp_measure_online - approx_clock)
        metadata['n_missing_markers'] = np.count_nonzero(np.isnan(timestamp_measure_online[:valid_clock_cycles]))
        n_consecutive_missing_cycles = utils.max_repeated_nans(timestamp_measure_online[:valid_clock_cycles])
        if n_consecutive_missing_cycles < max_consecutive_missing_cycles:
            metadata['has_measured_timestamps'] = True
            corrected_clock['timestamp_measure_online'] = timestamp_measure_online
        else:
            warnings.warn(f"Digital screen sensor missing too many markers ({n_consecutive_missing_cycles}/{max_consecutive_missing_cycles}). Ignoring")

    # 3. Screen photodiode measurements, raw voltage digitized offline
    if 'measure_clock_offline' in data and len(data['measure_clock_offline']) > 0:
        timestamp_measure_offline = get_measured_clock_timestamps(
            approx_clock, data['measure_clock_offline']['timestamp'], 
                measure_latency_estimate, measure_search_radius)
        corrected_clock = rfn.append_fields(corrected_clock, 'timestamp_measure_offline', timestamp_measure_offline, dtypes='f8')
        
        # If there are few missing measurements, include this as the default `timestamp`
        metadata['latency_measured'] = np.nanmean(timestamp_measure_offline - approx_clock)
        metadata['n_missing_markers'] = np.count_nonzero(np.isnan(timestamp_measure_offline[:valid_clock_cycles]))
        n_consecutive_missing_cycles = utils.max_repeated_nans(timestamp_measure_offline[:valid_clock_cycles])
        if n_consecutive_missing_cycles < max_consecutive_missing_cycles:
            corrected_clock['timestamp_measure_offline'] = timestamp_measure_offline
            metadata['has_measured_timestamps'] = True
        else:
            warnings.warn(f"Analog screen sensor missing too many markers ({n_consecutive_missing_cycles}/{max_consecutive_missing_cycles}). Ignoring")

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
        corrected_events = data['bmi3d_events']

    # But use the sync events if they exist and are valid
    if 'sync_events' in data and len(data['sync_events']) > 0:
        if not np.array_equal(data['sync_events']['code'], corrected_events['code']):
            warnings.warn("sync events don't match bmi3d events. This will probably cause problems.")
        corrected_events = data['sync_events']
    else:
        warnings.warn("No sync events present, using bmi3d events instead")

    # Also put some information about the reward system
    if 'reward_system' in data and 'reward_system' in metadata['features']:
        metadata['has_reward_system'] = True
    else:
        metadata['has_reward_system'] = False

    # And interpolated cursor kinematics
    if 'timestamp_sync' in corrected_clock.dtype.names and isinstance(task, np.ndarray) and 'cursor' in task.dtype.names:
        cursor_data_cycles = task['cursor'][:,[0,2]] # cursor (x, z) position on each bmi3d cycle
        clock = corrected_clock['timestamp_sync']
        assert cursor_data_cycles.shape[0] == len(clock), f"Cursor data and clock should have the same number of cycles ({cursor_data_cycles.shape[0]} vs {len(clock)})"
        cursor_samplerate = metadata['analog_samplerate']
        time = np.arange(int((clock[-1] + 10)*metadata['analog_samplerate']))/cursor_samplerate
        cursor_data_time, _ = interp_timestamps2timeseries(clock, cursor_data_cycles, sampling_points=time, interp_kind='linear')
        data['cursor_interp'] = cursor_data_time
        metadata['cursor_interp_samplerate'] = cursor_samplerate

    data.update({
        'task': task,
        'clock': corrected_clock,
        'events': corrected_events,
    })
    return data, metadata

def find_laser_stim_times(laser_event_times, laser_event_widths, laser_event_powers, sensor_data, samplerate, sensor_voltsperbit, 
                          thr_volts=0.005, ds_fs=5000, search_radius=0.015, thr_width=0.001, thr_power=0.05):
    '''
    Given expected laser timing and measured laser sensor data, find the measured timing and power that most likely
    corresponds to actual laser events. 

    See below, example aligned LFP during a saline test where the laser shines directly on an electrode.

    .. image:: _images/laser_aligned_lfp.png

    And the sensor voltage (10mV scale) aligned to the computed laser events.

    .. image:: _images/laser_aligned_sensor.png

    Args:
        laser_event_times (nevent): timestamps of when laser was supposed to fire
        laser_event_widths (nevent): supposed width of each laser event
        laser_event_powers (nevent): supposed power of each laser event
        sensor_data (nt): timeseries data from the laser sensor from the ecube analog port
        samplerate (float): sampling rate of the laser sensor data
        sensor_voltsperbit (float): volts per bit of the laser sensor data
        thr_volts (float, optional): threshold above which laser sensor data is counted. Defaults to 0.005.
        ds_fs (int, optional): downsampling rate, helps to smooth noise from the sensor. Defaults to 5000.
        search_radius (float, optional): time around the expected events to search for measured sensor readings. Defaults to 0.015.
        thr_width (float, optional): deviation in seconds from the expected widths above which the expected value will be used. Defaults to 0.001.
        thr_power (float, optional): threshold from the expected powers above which the expected value will be used. Defaults to 0.05.

    Returns:
        tuple: tuple containing:
            | **corrected_times (nevent):** corrected laser timings
            | **corrected_widths (nevent):** corrected laser widths
            | **corrected_powers (nevent):** corrected laser powers
            | **times_not_found (nevent):** boolean array of times where no sensor measurement was found
            | **widths_above_thr (nevent):** boolean array of widths above the given threshold from the expected width
            | **powers_above_thr (nevent):** boolean array of powers above the given threshold from the expected power
    '''
    
    # Calculate timing using the laser sensor
    ds_data = precondition.downsample(sensor_data, samplerate, ds_fs)
    ds_data = ds_data - np.mean(ds_data)
    threshold = thr_volts/sensor_voltsperbit
    digital_data = ds_data > threshold
    times, values = detect_edges(digital_data, ds_fs)
    if len(times) == 0:
        raise ValueError("No laser events detected. Try lowering the threshold")
    rising = times[values == 1]
    falling = times[values == 0]
    laser_sensor_times = rising
    laser_sensor_widths = falling - rising
    laser_on_times = np.mean([rising, falling], axis=0)
    laser_on_samples = (laser_on_times * ds_fs).astype(int)
    laser_sensor_powers = (ds_data[laser_on_samples])*sensor_voltsperbit
    
    # Normalize sensor power to the highest power, then multiply by the highest trial power
    laser_sensor_powers /= np.max(laser_sensor_powers)
    laser_sensor_powers *= np.max(laser_event_powers)

    # Check that the sensor measurements make sense, otherwise return the sync event versions
    corrected_times, corrected_idx = find_measured_event_times(laser_event_times, laser_sensor_times, search_radius, return_idx=True)
    missing_times = np.isnan(corrected_times)
    if np.any(missing_times):
        warnings.warn(f"{np.count_nonzero(missing_times)} unmeasured laser timestamps")
        corrected_times[missing_times] = laser_event_times[missing_times]
    
    # Adjust the sensor measurements to remove any that were missing
    times_not_found = np.isnan(corrected_idx)
    valid_idx = corrected_idx[~times_not_found].astype(int)
    adjusted_widths = corrected_idx.copy()
    adjusted_widths[~np.isnan(corrected_idx)] = laser_sensor_widths[valid_idx]
    adjusted_powers = corrected_idx.copy()
    adjusted_powers[~np.isnan(corrected_idx)] = laser_sensor_powers[valid_idx]

    # Correct the widths and powers with the given thresholds
    corrected_widths, widths_above_thr = validate_measurements(laser_event_widths, adjusted_widths, thr_width)
    corrected_powers, powers_above_thr = validate_measurements(laser_event_powers, adjusted_powers, thr_power)
    return corrected_times, corrected_widths, corrected_powers, times_not_found, widths_above_thr, powers_above_thr

def get_laser_trial_times(preproc_dir, subject, te_id, date, **kwargs):
    '''
    Get the laser trial times, trial widths, and trial powers from the given experiment. Returned
    values are computed from the laser sensor in combination with the expected laser events from
    BMI3D's hdf records.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        kwargs (dict): to be passed to `:func:~aopy.preproc.bmi3d.find_laser_stim_times`

    Returns:
        tuple: tuple containing:
            | **corrected_times (nevent):** corrected laser timings
            | **corrected_widths (nevent):** corrected laser widths
            | **corrected_powers (nevent):** corrected laser powers
            | **times_not_found (nevent):** boolean array of times where no sensor measurement was found
            | **widths_above_thr (nevent):** boolean array of widths above the given threshold from the expected width
            | **powers_above_thr (nevent):** boolean array of powers above the given threshold from the expected power
    '''
    
    exp_data, exp_metadata = aodata.load_preproc_exp_data(preproc_dir, subject, te_id, date)
    
    # Get the laser event timing from BMI3D's sync events
    events = exp_data['sync_events']['event'] 
    event_times = exp_data['sync_events']['timestamp']
    times = event_times[events == b'TRIAL_START']

    # Get width and power from the 'trials' data
    if 'bmi3d_trials' in exp_data:
        trials = exp_data['bmi3d_trials']
        powers = trials['power'][:len(times)]
        edges = trials['edges'][:len(times)]
        widths = [t[1] - t[0] for t in edges]
        thr_width=0.001
        thr_power=0.05
    else:
        # Can't use the exp data as ground truth, so just load the analog sensor data
        widths = np.zeros((len(times),))
        powers = np.zeros((len(times),))
        thr_width = 999
        thr_power = 1


    # Load the sensor data if it's not already in the bmi3d data
    if 'laser_sensor' not in exp_data:
        files, data_dir = get_source_files(preproc_dir, subject, te_id, date)
        hdf_filepath = os.path.join(data_dir, files['hdf'])
        if not os.path.exists(hdf_filepath):
            raise FileNotFoundError(f"Could not find raw files for te {te_id} ({hdf_filepath})")
        exp_data, exp_metadata = parse_bmi3d(data_dir, files)

    # Correct the event timings using the sensor data
    sensor_data = exp_data['laser_sensor']
    sensor_voltsperbit = exp_metadata['analog_voltsperbit']
    samplerate = exp_metadata['analog_samplerate']       
    return find_laser_stim_times(times, widths, powers, sensor_data, samplerate, 
        sensor_voltsperbit, thr_width=thr_width, thr_power=thr_power, **kwargs)
                                                                                
