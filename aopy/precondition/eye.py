# eye.py
#
# Preconditioning eye movement data, e.g. filtering and saccade detection

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import remodnav

from ..precondition import downsample
from ..utils import derivative, detect_edges
from ..data import get_kinematic_segments

def filter_eye(eye_pos, samplerate, downsamplerate=1000, low_cut=200, buttord=4):
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
        tuple: tuple containing:
            | **eye_pos (nt, nch):** eye position after filtering and downsampling
            | **samplerate (float):** sampling rate of the returned eye data
    '''
    # Lowpass filter
    b, a = butter(buttord, low_cut, btype='lowpass', fs=samplerate)
    eye_pos = filtfilt(b, a, eye_pos, axis=0)

    # Downsample
    if samplerate > downsamplerate:
        eye_pos = downsample(eye_pos, samplerate, downsamplerate)
    else:
        downsamplerate = samplerate

    return eye_pos, downsamplerate

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

def detect_saccades(eye_pos, samplerate, thr=None, num_sd=1.5, intersaccade_min=None, 
                    min_saccade_duration=0.015, max_saccade_duration=0.16, 
                    lowpass_filter_freq=30, debug=False, debug_window=(0, 2)):
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
        thr ((high,low) tuple): positive and negative threshold values for acceleration, if desired. If None (default), 
            thresholds will be automatically chosen based on num_sd above the mean.
        num_sd (float, optional): number of standard deviations above zero to threshold acceleration
        intersaccade_min (float, optional): minimum time (in seconds) allowed between saccades (from offset to next onset)
        min_saccade_duration (float, optional): minimum time (in seconds) that a saccade can take (inclusive)
        max_saccade_duration (float, optional): maximum time (in seconds) that a saccade can take (exclusive)
        lowpass_filter_freq (float, optional): frequency in Hz to low-pass filter. If None, no filtering will take place. Default 30 Hz. 
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
    
    if lowpass_filter_freq is not None:
        eye_pos, _ = filter_eye(eye_pos, samplerate, samplerate, low_cut=lowpass_filter_freq)    
    eye_accel = convert_pos_to_accel(eye_pos, samplerate)

    # Set an appropritate threshold to detect saccades
    if thr is None:
        baseline_mean = np.mean(eye_accel)
        baseline_std = np.std(eye_accel)
        thr = np.mean(baseline_mean) + num_sd*baseline_std
        thr = (thr, -thr)
    saccade_onset_time, _ = detect_edges(eye_accel > thr[0], samplerate, rising=True, falling=False)
    saccade_offset_time, _ = detect_edges(eye_accel < thr[1], samplerate, rising=False, falling=True)

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
                np.linalg.norm(eye_pos[int(t*samplerate),:] - 
                               eye_pos[int(offset_time*samplerate),:])
            ) # distance is the deviation of the saccade in space (in cm)
            prev_offset = offset_time
    onset = np.array(onset)
    duration = np.array(duration)
    distance = np.array(distance)

    if debug:
        time = np.arange(len(eye_pos))/samplerate
        debug_window = (debug_window[0], min(debug_window[1], time[-1]))
        debug_idx = (int(debug_window[0]*samplerate), int(debug_window[1]*samplerate))
        fig, ax = plt.subplots(2,1)
        ax[0].plot(time[debug_idx[0]:debug_idx[1]], eye_pos[debug_idx[0]:debug_idx[1]])
        ax[1].plot(time[debug_idx[0]:debug_idx[1]], eye_accel[debug_idx[0]:debug_idx[1]])
        ax[0].set_ylabel('position (cm)')
        ax[1].set_xlabel('time (s)')
        ax[1].set_ylabel('accel cm/s^2')
        ax[1].plot([debug_window[0],debug_window[1]], [thr[0], thr[0]], '--')
        ax[1].plot([debug_window[0],debug_window[1]], [thr[1], thr[1]], '--')
        
        min_max = [np.min(eye_pos[debug_idx[0]:debug_idx[1]]), np.max(eye_pos[debug_idx[0]:debug_idx[1]])]
        debug_idx = (onset > debug_window[0]) & (onset < debug_window[1])
        onset_debug = onset[debug_idx]
        duration_debug = duration[debug_idx]
        for o, d in zip(onset_debug, duration_debug):
            ax[0].plot([o, o], min_max, 'g--')
            ax[0].plot([o+d, o+d], min_max, 'r--')
            ax[1].plot([o, o], [thr[0], thr[1]], 'g--')
            ax[1].plot([o+d, o+d], [thr[0], thr[1]], 'r--')

    return onset, duration, distance
        
'''REMoDNaV classification'''
def get_default_parameters(pursuit_velthresh=5.0,noise_factor=5.0,velthresh_startvelocity=300.0,min_intersaccade_duration=0.04,min_saccade_duration=0.01,
                           max_initial_saccade_freq=2.0,saccade_context_window_length=1.0,max_pso_duration=0.04,min_fixation_duration=0.04,min_pursuit_duration=0.04,
                           lowpass_cutoff_freq=4.0,min_blink_duration=0.02,dilate_nan=0.01,median_filter_length=0.05,savgol_length=0.019,savgol_polyord=2,max_vel=2000.0):
    """
        Returns default parameters required for behavior analysis using REMoDNaV (Dar et al., 2021) https://doi.org/10.3758/s13428-020-01428-x

        Args:
            pursuit_velthresh (float): Velocity threshold to distinguish periods of pursuit from periods of fixation. Default is 5.0.
            noise_factor (float): Factor to account for noise in the data. Default is 5.0.
            velthresh_startvelocity (float): Start value for velocity threshold algorithm. Default is 300.0.
            min_intersaccade_duration (float): No saccade classification is performed in windows shorter than twice this value plus minimum saccade and PSO duration. Default is 0.04.
            min_saccade_duration (float): Minimum duration of a saccade event candidate in seconds. Default is 0.01.
            max_initial_saccade_freq (float): Maximum frequency for initial saccade detection. Default is 2.0.
            saccade_context_window_length (float): Size of a window centered on any velocity peak for adaptive determination of saccade velocity threshold. Default is 1.0.
            max_pso_duration (float): Maximum duration of post-saccadic oscillations in seconds. Default is 0.04.
            min_fixation_duration (float): Minimum duration of fixation event candidate in seconds. Default is 0.04.
            min_pursuit_duration (float): Minimum duration of pursuit event candidate in seconds. Default is 0.04.
            lowpass_cutoff_freq (float): Cutoff frequency for lowpass filtering. Default is 4.0.
            min_blink_duration (float): Missing data windows shorter than this duration will not be considered for dilate nan. Default is 0.02.
            dilate_nan (float): Duration for which to replace data by missing data markers either side of a signal-loss window. Default is 0.01.
            median_filter_length (float): Smoothing median filter size in seconds. Default is 0.05.
            savgol_length (float): Size of the Savitzky-Golay filter for noise reduction in seconds. Default is 0.019.
            savgol_polyord (int): Polynomial order of the Savitzky-Golay filter for noise reduction. Default is 2.
            max_vel (float): Maximum velocity threshold, will replace values above designated maximum. Default is 2000.0.

        Returns:
            tuple: A tuple containing two dictionaries:
                - clf_params: Parameters needed for event classification.
                - preproc_params: Parameters needed for preprocessing.
        """
    clf_params = dict(
        pursuit_velthresh=pursuit_velthresh,
        noise_factor=noise_factor,
        velthresh_startvelocity=velthresh_startvelocity,
        min_intersaccade_duration=min_intersaccade_duration,
        min_saccade_duration=min_saccade_duration,
        max_initial_saccade_freq=max_initial_saccade_freq,
        saccade_context_window_length=saccade_context_window_length,
        max_pso_duration=max_pso_duration,
        min_fixation_duration=min_fixation_duration,
        min_pursuit_duration=min_pursuit_duration,
        lowpass_cutoff_freq=lowpass_cutoff_freq,
        )
    preproc_params = dict(
        min_blink_duration=min_blink_duration,
        dilate_nan=dilate_nan,
        median_filter_length=median_filter_length,
        savgol_length=savgol_length,
        savgol_polyord=savgol_polyord,
        max_vel=max_vel,
        )
    return clf_params, preproc_params

def classify_eye_events(eye_trajectory, clf_params, preproc_params, screen_half_height, viewing_dist, samplerate):
    """
        Classifies eye behavior based on a single trial eye trajectory using the REMoDNaV algorithm.
        
        Args:
            eye_trajectories (nt, n_features): array of eye trajectories,
                                    Where 'nt' is the number of timepoints,
                                    'n_features' are the x,y coordinates for one eye. 
                                    Note: n_features shape must = 2.
            clf_params (dict): Dictionary of classifier parameters to be passed to the REMoDNaV EyegazeClassifier. 
            preproc_params (dict): Dictionary of preprocessing parameters to be passed to the classifier's preproc method.
            screen_half_height (float): Half the height of the screen in cm.
            viewing_dist (float): The viewing distance in cm.
            samplerate (int, optional): Sampling rate of the eye tracker in Hz.
    
        Returns:
            List: list of dictionaries containing classified eye movement events
                    where one list per trial containing multiple dictionaries,
                    and each dictionary contains the following keys:
                        - 'id': event identifier.
                        - 'label': Type of eye movement (e.g., saccade, fixation, pursuit).
                        - 'start_time': Start time of the event in seconds.
                        - 'end_time': End time of the event in seconds.
                        - 'start_x': Start x-coordinate of the event.
                        - 'start_y': Start y-coordinate of the event.
                        - 'end_x': End x-coordinate of the event.
                        - 'end_y': End y-coordinate of the event.
                        - 'amplitude': Amplitude of the eye movement in degrees.
                        - 'peak_velocity': Peak velocity of the eye movement in degrees/second.
                        - 'med_velocity': Median velocity of the eye movement in degrees/second.
                        - 'avg_velocity': Average velocity of the eye movement in degrees/second.

        Example:

            Detecting saccades on the first 5 seconds of test data from beignet block 5974 (included in /tests/data/beignet)
    
            .. image:: _images/remodnav_saccades_hist.png
            .. image:: _images/remodnav_saccades_scatter.png
    """
        
    eye_trajectory=np.array(eye_trajectory)

    if eye_trajectory.ndim != 2 or eye_trajectory.shape[1] != 2:
        raise ValueError("Warning! Input array must be a 2D numpy array with shape (nt, 2).")

    eye_data = eye_trajectory.T

    data = np.core.records.fromarrays(
        eye_data,
        names='x,y',
        formats='f8,f8'
    )
    screen_half_height_deg=np.degrees(np.arctan2(screen_half_height,viewing_dist))
    px2deg=screen_half_height_deg/screen_half_height
    clf = remodnav.EyegazeClassifier(px2deg, samplerate, **clf_params)
    pp = clf.preproc(data, **preproc_params)
    events = clf(pp, classify_isp=True, sort_events=True)
    
    return events

def detect_eye_events(eye_trajectory, event_label, clf_params, preproc_params, screen_half_height, viewing_dist, samplerate):
    """
    Extracts the start and end times and x,y positions of specified events from a single eye trajectory. 
    You must select a specific eye event to extract. Choose from following options using the event_label parameter.
    Options for event labels:
        - 'SACC': Saccadic eye movements.
        - 'PURS': Smooth pursuit eye movements.
        - 'FIXA': Fixation events.
        - 'HPSO': High velocity post saccadic oscillations.
        - 'LPSO': Low velocity post saccadic oscillations.
        - 'IHPS': High velocity post inter-saccadic oscillations.
        - 'ILPS': Low velocity post inter-saccadic oscillations.

    Args:
        eye_trajectories (nt, n_features): array of eye trajectories,
                                    Where 'nt' is the number of timepoints,
                                    'n_features' are the x,y coordinates for each eye.
        event_label (str): The label of the event to extract. (e.g., 'SACC', 'PURS', 'FIXA')
        clf_params (dict): Dictionary of classifier parameters to be passed to the REMoDNaV EyegazeClassifier.
        preproc_params (dict): Dictionary of preprocessing parameters to be passed to the classifier's preproc method.
        screen_half_height (float): Half the height of the screen in cm.
        viewing_dist (float): The viewing distance in cm.
        samplerate (int, optional): Sampling rate of the eye tracker in Hz.

    Returns:
        tuple: A tuple containing
            - times (ndarray): An array of start and end times in seconds for each event.
            - start_positions (ndarray): An array of start positions (x, y) for each event.
            - end_positions (ndarray): An array of end positions (x, y) for each event.
    """
    events=classify_eye_events(eye_trajectory, clf_params, preproc_params, screen_half_height, viewing_dist, samplerate)

    times=[]
    start_positions = []
    end_positions = []

    for i in events:
        if i['label'] == event_label:
            times.append([i['start_time'],i['end_time']])
            start_positions.append([i['start_x'], i['start_y']])
            end_positions.append([i['end_x'], i['end_y']])

    return np.array(times), np.array(start_positions), np.array(end_positions)


def get_eye_event_trials(preproc_dir, subject, te_id, date, start_events, end_events, event_label, clf_params, preproc_params, screen_half_height, viewing_dist, samplerate=1000):
    
    """
    Extracts event times and positions from eye movement data for a given session.
    You must select a specific eye event to extract. Choose from following options using the event_label parameter.
    Options for event labels:
        - 'SACC': Saccadic eye movements.
        - 'PURS': Smooth pursuit eye movements.
        - 'FIXA': Fixations.
        - 'HPSO': High velocity post saccadic oscillations.
        - 'LPSO': Low velocity post saccadic oscillations.
        - 'IHPS': High velocity post inter-saccadic oscillations.
        - 'ILPS': Low velocity post inter-saccadic oscillations.
    
    Args:
        preproc_dir (str): Base directory where file lives
        subject (str): Subject name.
        te_id (int): Block number of task entry object.
        date (str): Date of the recording.
        start_events (list): List of numeric codes representing the start of a trial.
        end_events (list): List of numeric codes representing the end of a trial.
        event_label (str): The label of the event to extract. (e.g., 'SACC', 'PURS', 'FIXA')
        clf_params (dict): Dictionary of classifier parameters to be passed to the REMoDNaV EyegazeClassifier.
        preproc_params (dict): Dictionary of preprocessing parameters to be passed to the classifier's preproc method.
        screen_half_height (float): Half the height of the screen in degrees of visual angle.
        viewing_dist (float): The viewing distance in cm.
        samplerate (int, optional): Sampling rate of the eye tracker in Hz. Default is 1000.

    Returns:
        tuple: A tuple containing
            - start_end_times (ntrials): 
                List of (nevents, 2) arrays containing (start, end) times of detected events for each trial in seconds.
            - start_pos (ntrials):
                List of (nevents, 2) arrays containing start (x,y) positions of detected events for each trial.
            - end_pos (ntrials):
                List of (nevents,2) arrays containing end (x,y) positions of detected events for each trial.
    """

    eye_trajectories, eye_codes = get_kinematic_segments(
        preproc_dir, subject, te_id, date, start_events, end_events, datatype='eye', samplerate=samplerate
    )
    
    start_end_times=[]
    start_pos=[]
    end_pos=[]

    for idx, eye_trajectory in enumerate(eye_trajectories):
        if np.ndim(eye_trajectory) == 2 and np.shape(eye_trajectory)[1] > 2:
            eye_trajectory=eye_trajectory[:,:2]

        s_e, xs_ys, xe_ye = detect_eye_events(eye_trajectory, event_label, clf_params, preproc_params, screen_half_height, viewing_dist, samplerate)
    
        if s_e.size == 0:
            print(f"No matching events found in trial {idx}, appending empty arrays.")
            start_end_times.append([])
            start_pos.append([])
            end_pos.append([])
        else:
            start_end_times.append(s_e)
            start_pos.append(xs_ys)
            end_pos.append(xe_ye)

    return start_end_times,start_pos,end_pos