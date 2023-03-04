import numpy as np
import matplotlib.pyplot as plt

from ..precondition import downsample, mt_lowpass_filter
from ..utils import derivative, detect_edges

def detect_saccades(eye_pos, samplerate, downsamplerate=480, lowpass_freq=30, taper_len=0.05, 
                    num_sd=1.5, intersaccade_min=0.02, max_saccade_duration=0.15, 
                    debug=False, debug_window=(0, 2)):
    '''
    Detect saccades from eye position data. Starts with lowpass filtering of the position, then thresholding
    of the acceleration, ensuring that saccades are spaced by some minimum intersaccade time, and finally 
    only considering events with both positive and negative crossings within the minimum saccade time. Optional
    plots can be generated showing threshold crossings in position, velocity, and acceleration.

    Example:

        Detecting saccades on the first 5 seconds of test data from beignet block 5974 (included in /tests/data/beignet)
    
        .. image:: _images/detect_saccades.png
        .. image:: _images/detect_saccades_hist.png
        .. image:: _images/detect_saccades_scatter.png
    
    Args:
        eye_pos (nt, nch): eye position data, e.g. from oculomatic
        samplerate (float): sampling rate of the eye data
        downsamplerate (float, optional): sampling rate at which to do saccade analysis 
        lowpass_freq (float, optional): low cutoff frequency to limit analysis of saccades
        taper_len (float, optional): length of tapers to use in multitaper lowpass filter
        num_sd (float, optional): number of standard deviations above zero to threshold acceleration
        intersaccade_min (float, optional): minimum time (in seconds) allowed between saccade onsets
        max_saccade_duration (float, optional): maximum time (in seconds) that a saccade can take
        debug (bool, optional): if True, display a figure showing the threshold crossings
        debug_window (tuple, optional): (start, end) time (in seconds) for the debug figure
        
    Returns:
        tuple: tuple containing:
        | **onset (nsaccade):** onset time (in seconds) of each detected saccade
        | **duration (nsaccade):** duration (in seconds) of each detected saccade
        | **distance (nsaccade):** distance (same units as eye_pos) of each detected saccade
    '''
    # Downsample
    if samplerate > downsamplerate:
        if debug:
            print(f"downsampling from {samplerate} to {downsamplerate}")
        eye_pos = downsample(eye_pos, samplerate, downsamplerate)
        samplerate = downsamplerate

    # Lowpass filter
    n_pad = int(1.0*samplerate)
    eye_pos = np.pad(eye_pos, ((n_pad,n_pad),(0,0)), mode='edge')
    eye_pos = mt_lowpass_filter(eye_pos, lowpass_freq, taper_len, samplerate)
    eye_pos = eye_pos[n_pad:-n_pad,:]
    
    # Differentiate twice to get acceleration
    time = np.arange(eye_pos.shape[0])/samplerate
    velocity = derivative(time, eye_pos, norm=True)
    accel = derivative(time, velocity, norm=False)

    # Set an appropritate threshold to detect saccades
    baseline_mean = np.mean(accel)
    baseline_std = np.std(accel)
    thr = np.mean(baseline_mean) + num_sd*baseline_std
    saccade_onset_time, _ = detect_edges(accel > thr, samplerate, rising=True, falling=False)
    saccade_offset_time, _ = detect_edges(accel < -thr, samplerate, rising=False, falling=True)

    if debug:
        debug_idx = (int(debug_window[0]*samplerate), int(debug_window[1]*samplerate))
        fig, ax = plt.subplots(3,1)
        ax[0].plot(time[debug_idx[0]:debug_idx[1]], eye_pos[debug_idx[0]:debug_idx[1]])
        ax[1].plot(time[debug_idx[0]:debug_idx[1]], velocity[debug_idx[0]:debug_idx[1]])
        ax[2].plot(time[debug_idx[0]:debug_idx[1]], accel[debug_idx[0]:debug_idx[1]])
        ax[0].set_ylabel('position (cm)')
        ax[1].set_ylabel('velocity (cm/s)')
        ax[2].set_xlabel('time (s)')
        ax[2].set_ylabel('accel cm/s^2')
        ax[2].plot([debug_window[0],debug_window[1]], [thr, thr], '--')
        ax[2].plot([debug_window[0],debug_window[1]], [-thr, -thr], '--')

    if saccade_onset_time.size == 0 or saccade_offset_time.size == 0:
        return [], [], []

    # Only consider saccades that aren't immediately preceded by another saccade
    intersaccade_interval = np.diff(saccade_onset_time) 
    valid_onset_idx = np.where(intersaccade_interval > intersaccade_min)[0]
    saccade_onset_idx = np.insert(valid_onset_idx + 1, 0, 0)
    onset_time = saccade_onset_time[saccade_onset_idx]

    # Only consider saccades that have a threshold-crossing offset
    onset = []
    duration = []
    distance = []
    for t in onset_time:
        nearby_offset = np.where((saccade_offset_time > t) & 
                                 (saccade_offset_time < t + max_saccade_duration))[0]
        if len(nearby_offset):
            offset_time = saccade_offset_time[nearby_offset[0]]
            onset.append(t)
            duration.append(offset_time - t)
            distance.append(
                np.linalg.norm(eye_pos[int(t*samplerate),:] - 
                               eye_pos[int(offset_time*samplerate)])
            ) # distance is the deviation of the saccade in space (in cm)
    onset = np.array(onset)
    duration = np.array(duration)
    distance = np.array(distance)

    if debug:
        debug_idx = (onset > debug_window[0]) & (onset < debug_window[1])
        onset_debug = onset[debug_idx]
        duration_debug = duration[debug_idx]
        for o, d in zip(onset_debug, duration_debug):
            ax[2].plot([o, o], [-thr, thr], 'g--')
            ax[2].plot([o+d, o+d], [-thr, thr], 'r--')

    return onset, duration, distance
        
