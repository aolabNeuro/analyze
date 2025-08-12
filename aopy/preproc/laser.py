# laser.py
#
# preprocessing laser data

import warnings
import sys
if sys.version_info >= (3,9):
    from importlib.resources import files, as_file
else:
    from importlib_resources import files, as_file

import numpy as np
import matplotlib.pyplot as plt

from .. import data as aodata
from .. import analysis
from .. import visualization
from .. import precondition
from .. import utils
from . import base

def find_stim_times(laser_event_times, laser_event_widths, laser_event_gains, sensor_data, samplerate, 
                    sensor_voltsperbit, peak_power_mW, thr_volts=0.005, ds_fs=5000, search_radius=0.015, 
                    thr_width=0.001, calibration_file='qwalor_447nm_ch2.yaml', thr_power=1., debug=False):
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
        laser_event_gains (nevent): supposed gain of each laser event
        sensor_data (nt): timeseries data from the laser sensor from the ecube analog port
        samplerate (float): sampling rate of the laser sensor data
        sensor_voltsperbit (float): volts per bit of the laser sensor data
        peak_power_mW (float): peak power of the laser in mW
        thr_volts (float, optional): threshold in volts above which laser sensor data is counted. Defaults to 0.005.
        ds_fs (int, optional): downsampling rate, helps to smooth noise from the sensor. Defaults to 5000.
        search_radius (float, optional): time in seconds around the expected events to search for measured sensor readings. Defaults to 0.015.
        thr_width (float, optional): deviation in seconds from the expected widths above which the expected value will be used. Defaults to 0.001.
        thr_power (float, optional): threshold from the expected powers above which the expected value will be used. Defaults to 0.05.
        debug (bool, optional): print out debug messages and a plot of the laser sensor aligned to the computed times

    Returns:
        tuple: tuple containing:
            | **corrected_times (nevent):** corrected laser timings (seconds)
            | **corrected_widths (nevent):** corrected laser widths (seconds)
            | **corrected_powers (nevent):** corrected laser powers (mW)
            | **times_not_found (nevent):** boolean array of times without onset and offset sensor measurements
            | **widths_above_thr (nevent):** boolean array of widths above the given threshold from the expected width
            | **powers_above_thr (nevent):** boolean array of powers above the given threshold from the expected power
    '''
    
    # Calculate timing using the laser sensor
    ds_data = precondition.downsample(sensor_data, samplerate, ds_fs)
    ds_data = ds_data - np.mean(ds_data)
    threshold = thr_volts/sensor_voltsperbit
    digital_data = ds_data > threshold
    times, values = utils.detect_edges(digital_data, ds_fs)
    if len(times) == 0:
        raise ValueError("No laser events detected. Try lowering the threshold")
    rising = times[values == 1]
    falling = times[values == 0]
    laser_sensor_times = rising
    laser_sensor_off_times = falling
    
    # Check that the sensor measurements make sense, otherwise return the sync event versions
    corrected_times, corrected_idx = base.find_measured_event_times(laser_event_times, laser_sensor_times, search_radius, return_idx=True)
    missing_times = np.isnan(corrected_times)
    if np.any(missing_times):
        warnings.warn(f"{np.count_nonzero(missing_times)} unmeasured laser timestamps")
        corrected_times[missing_times] = laser_event_times[missing_times]
    corrected_off_times, corrected_off_idx = base.find_measured_event_times(laser_event_times+laser_event_widths, laser_sensor_off_times, search_radius, return_idx=True)
    missing_off_times = np.isnan(corrected_off_times)
    if np.any(missing_off_times):
        warnings.warn(f"{np.count_nonzero(missing_times)} unmeasured laser offsets")
        corrected_off_times[missing_off_times] = laser_event_times[missing_off_times] + laser_event_widths[missing_off_times]
    times_not_found = np.logical_or(np.isnan(corrected_idx), np.isnan(corrected_off_idx))

    # Now calculate the widths and powers based on corrected times
    laser_sensor_widths = corrected_off_times - corrected_times
    laser_on_times = np.mean([corrected_off_times, corrected_times], axis=0)
    laser_on_samples = (laser_on_times * ds_fs).astype(int)
    laser_sensor_volts = (ds_data[laser_on_samples])*sensor_voltsperbit
    laser_sensor_powers = calibrate_sensor(laser_sensor_volts, peak_power_mW)  

    # Correct the widths and powers with the given thresholds
    laser_event_powers = calibrate_gain(laser_event_gains, peak_power_mW, calibration_file=calibration_file)
    corrected_widths, widths_above_thr = base.validate_measurements(laser_event_widths, laser_sensor_widths, thr_width)
    corrected_powers, powers_above_thr = base.validate_measurements(laser_event_powers, laser_sensor_powers, thr_power)

    if debug:
        print(f"BMI3D recorded {len(laser_event_times)} stims")
        print(f"Laser sensor crossed threshold {len(laser_sensor_times)} times")
        if len(laser_event_times) == len(laser_sensor_times):
            print(f"Average difference: {np.mean(laser_event_times - laser_sensor_times):.4f} s")
        else:
            print("Cannot compute average difference. Check the laser threshold is correct")
            print(f"Using threshold: {thr_volts}")
            print(f"Maximum voltage deviation of sensor: {np.max(ds_data)*sensor_voltsperbit}")

        plt.figure()
        visualization.plot_laser_sensor_alignment(ds_data*sensor_voltsperbit, ds_fs, corrected_times)

    return corrected_times, corrected_widths, corrected_powers, times_not_found, widths_above_thr, powers_above_thr

def calibrate_gain(gain, peak_power_mW, calibration_file='qwalor_447nm_ch2.yaml'):
    '''
    Convert gain into laser power in mW

    Args:
        gain ((ntrial,) float array): sensor readings in volts
        peak_power_mW (float): peak power in mW
        calibration_file (str, optional): name of calibration file. Defaults to 'qwalor_447nm_ch2.yaml'.

    Returns:
        (ntrial,) float array: laser power in mW

    Examples:

        .. code-block:: python

            gain = np.arange(0.4,1.0,0.01)
            powers = preproc.laser.calibrate_laser_gain(gain, 12.)
            plt.figure()
            plt.plot(gain, powers)
            plt.xlabel('gain')
            plt.ylabel('power (mW)')

        .. image:: _images/calibrate_laser_gain.png
    '''
    yaml_data = aodata.load_yaml_config(calibration_file)
    voltages = np.interp(gain, yaml_data['gain'], yaml_data['power'])
    
    min_voltage = np.min(yaml_data['power'])
    max_voltage = np.max(yaml_data['power'])
    rng = max_voltage - min_voltage

    powers = (voltages - min_voltage)/rng * peak_power_mW
    return powers

def calibrate_sensor(sensor_voltage, peak_power_mW, 
                           calibration_file='qwalor_447nm_ch2.yaml'):
    '''
    Convert sensor voltage into laser power in mW

    Args:
        sensor_voltage ((ntrial,) float array): sensor readings in volts
        peak_power_mW (float): peak power in mW
        calibration_file (str, optional): name of calibration file. Defaults to 'qwalor_447nm_ch2.yaml'.

    Returns:
        (ntrial,) float array: laser power in mW
    '''
    yaml_data = aodata.load_yaml_config(calibration_file)
    
    # Use interpolation in case calibration doesn't include 0. or 1. gain setting
    voltages = np.interp([0.,1.], yaml_data['gain'], yaml_data['power'])
    rng = voltages[1] - voltages[0]

    powers = (sensor_voltage - voltages[0])/rng * peak_power_mW
    return powers

    
