import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from ..precondition import downsample
from ..utils import derivative, detect_edges

def filter_eye(eye_pos, samplerate, downsamplerate=1000, low_cut=100, buttord=4):
    '''
    Filter and downsample eye data.

    .. image:: _images/proc_oculomatic_freq.png

    Args:
        eye_pos (nt, nch): eye position data, e.g. from oculomatic
        samplerate (float): original sampling rate of the eye data
        downsamplerate (float, optional): sampling rate at which to return eye data
        lowpass_freq (float, optional): low cutoff frequency to limit analysis of saccades
        taper_len (float, optional): length of tapers to use in multitaper lowpass filter

    Returns:
        (nt, nch) eye pos: eye position after filtering and downsampling
    '''
    # Lowpass filter
    b, a = butter(buttord, low_cut, btype='lowpass', fs=samplerate)
    eye_pos = filtfilt(b, a, eye_pos, axis=0)

    # Downsample
    if samplerate > downsamplerate:
        eye_pos = downsample(eye_pos, samplerate, downsamplerate)

    return eye_pos

def convert_pos_to_speed(eye_pos, samplerate):
    '''
    Differentiate to get speed from eye position.

    Args:
        eye_pos (nt, nch): eye position data, e.g. from oculomatic
        samplerate (float): sampling rate of the eye data
    
    Returns:
        (nt, nch) array: eye speed

    Examples:

        .. image:: _images/convert_pos_to_accel_nofilter.png

        .. image:: _images/convert_pos_to_accel_filter.png
    '''
    time = np.arange(eye_pos.shape[0])/samplerate
    speed = derivative(time, eye_pos, norm=True)

    return speed    

def convert_pos_to_accel(eye_pos, samplerate):
    '''
    Differentiate twice to get acceleration from eye position.

    Args:
        eye_pos (nt, nch): eye position data, e.g. from oculomatic
        samplerate (float): sampling rate of the eye data
    
    Returns:
        (nt, nch) array: eye acceleration
    '''
    speed = convert_pos_to_speed(eye_pos, samplerate)
    time = np.arange(eye_pos.shape[0])/samplerate
    accel = derivative(time, speed, norm=False)

    return accel

def detect_saccades(eye_kinematics, samplerate, thr=None, num_sd=1.5, intersaccade_min=None, 
                    min_saccade_duration=0.015, max_saccade_duration=0.16, 
                    debug=False, debug_window=(0, 2)):
    '''
    Detect saccades from eye kinematics data. Uses thresholding of the kinematics 
    to find putative saccades, then ensures that saccades are spaced by some 
    minimum intersaccade time, and finally only considers events with both positive 
    and negative crossings within the minimum saccade time. Optional plots can be 
    generated showing threshold crossings in position, velocity, and acceleration.

    Minimum and maximum saccade durations can be enforced to remove events that are 
    biologically implausible. Default minimum and maximum saccade durations from 
    Dorr et al., 2010 https://doi.org/10.1167/10.10.28
    
    A minimum intersaccade interval can also be set to prevent the categorization 
    of potential overshoot components as saccadic events. Default intersaccade 
    minimum from Sinn & Engbert, 2016 https://doi.org/10.1016/j.visres.2015.05.012 
    
    Example:

        Detecting saccades on the first 5 seconds of test data from beignet block 5974 (included in /tests/data/beignet)
    
        .. image:: _images/detect_saccades.png
        .. image:: _images/detect_saccades_hist.png
        .. image:: _images/detect_saccades_scatter.png
    
    Args:
        eye_kinematics (nt, nch): eye kinematics data, e.g. velocity or acceleration for one eye
        samplerate (float): sampling rate of the eye data
        num_sd (float, optional): number of standard deviations above zero to threshold acceleration
        intersaccade_min (float, optional): minimum time (in seconds) allowed between saccades (from offset to next onset)
        min_saccade_duration (float, optional): minimum time (in seconds) that a saccade can take (inclusive)
        max_saccade_duration (float, optional): maximum time (in seconds) that a saccade can take (exclusive)
        debug (bool, optional): if True, display a figure showing the threshold crossings
        debug_window (tuple, optional): (start, end) time (in seconds) for the debug figure
        
    Returns:
        tuple: tuple containing:
        | **onset (nsaccade):** onset time (in seconds) of each detected saccade
        | **duration (nsaccade):** duration (in seconds) of each detected saccade
        | **distance (nsaccade):** distance (same units as eye_pos) of each detected saccade

    '''
    assert thr is None or (len(thr) == 2 and thr[0] > thr[1]), ("Threshold must be in the form" 
        " (positive thr, negative thr)")
    assert (intersaccade_min is None) or intersaccade_min < max_saccade_duration, ("Max saccade"
        " duration must be longer than the minimum intersaccade interval")
    assert np.ndim(eye_kinematics) == 1, ("Eye kinematics must be 1-dimensional. Examples include"
                                          "speed or acceleration.")

    # Set an appropritate threshold to detect saccades
    if thr is None:
        baseline_mean = np.mean(eye_kinematics)
        baseline_std = np.std(eye_kinematics)
        thr = np.mean(baseline_mean) + num_sd*baseline_std
        thr = (thr, -thr)
    saccade_onset_time, _ = detect_edges(eye_kinematics > thr[0], samplerate, rising=True, falling=False, check_alternating=False)
    saccade_offset_time, _ = detect_edges(eye_kinematics < thr[1], samplerate, rising=False, falling=True, check_alternating=False)

    if saccade_onset_time.size == 0 or saccade_offset_time.size == 0:
        return [], [], []

    # Only consider saccades that have a threshold-crossing offset
    onset = []
    duration = []
    distance = []
    prev_offset = 0
    for t in saccade_onset_time:
        if intersaccade_min and t < prev_offset + intersaccade_min:
            continue # ignore saccades within intersaccade_min of the last saccade
        nearby_offset = np.where((saccade_offset_time >= t + min_saccade_duration) & 
                                 (saccade_offset_time < t + max_saccade_duration))[0]
        if len(nearby_offset):
            offset_time = saccade_offset_time[nearby_offset[0]]
            onset.append(t)
            duration.append(offset_time - t)
            distance.append(
                np.linalg.norm(eye_kinematics[int(t*samplerate),:] - 
                               eye_kinematics[int(offset_time*samplerate),:])
            ) # distance is the deviation of the saccade in space (in cm)
            prev_offset = offset_time
    onset = np.array(onset)
    duration = np.array(duration)
    distance = np.array(distance)

    if debug:
        from ..visualization import plot_saccades

        time = np.arange(len(eye_kinematics))/samplerate
        debug_window = (debug_window[0], min(debug_window[1], time[-1]))
        debug_idx = (onset > debug_window[0]) & (onset < debug_window[1])
        onset_debug = onset[debug_idx]
        duration_debug = duration[debug_idx]
        debug_idx = (int(debug_window[0]*samplerate), int(debug_window[1]*samplerate))

        plt.figure()
        plot_saccades(eye_kinematics[debug_idx[0]:debug_idx[1]], onset_debug, duration_debug)

    return onset, duration, distance
        
