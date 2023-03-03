import numpy as np
import matplotlib.pyplot as plt

from ..precondition import downsample, mt_lowpass_filter
from ..utils import derivative, detect_edges

def detect_saccades(eye_kinematics, samplerate, samplerate_max=480, lowpass_freq=30, taper_len=0.05, 
                    num_sd=1.5, intersaccade_min=0.02, max_saccade_duration=0.15, 
                    debug=False, debug_window=(0, 2)):
    '''
    Detect saccade times. 
    
    Args:
        time
        eye_kinematics
        
        samplerate_max
        num_sd
        intersaccade_min
        
    Returns:
        saccades
    '''
    # Downsample
    if samplerate > samplerate_max:
        if debug:
            print(f"downsampling from {samplerate} to {samplerate_max}")
        eye_kinematics = downsample(eye_kinematics, samplerate, samplerate_max)
        samplerate = samplerate_max

    # Lowpass filter
    n_pad = int(1.0*samplerate)
    print(eye_kinematics.shape)
    eye_kinematics = np.pad(eye_kinematics, ((n_pad,n_pad),(0,0)), mode='edge')
    eye_kinematics = mt_lowpass_filter(eye_kinematics, lowpass_freq, taper_len, samplerate)
    eye_kinematics = eye_kinematics[n_pad:-n_pad,:]
    print(eye_kinematics.shape)
    
    # Differentiate twice to get acceleration
    time = np.arange(eye_kinematics.shape[0])/samplerate
    velocity = derivative(time, eye_kinematics, norm=True)
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
        ax[0].plot(time[debug_idx[0]:debug_idx[1]], eye_kinematics[debug_idx[0]:debug_idx[1]])
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
                np.linalg.norm(eye_kinematics[int(t*samplerate),:] - 
                               eye_kinematics[int(offset_time*samplerate)])
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
        
