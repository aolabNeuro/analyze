# postproc.py
# code for post-processing neural data
import numpy as np
import warnings

def translate_spatial_data(spatial_data, new_origin):
    '''
    Shifts 2D or 3D spatial data to a new location.

    Args:
        spatial_data (nt, ndim): Spatial data in 2D or 3D
        new_origin (ndim): Location of point that will become the origin in cartesian coordinates

    Returns:
        new_spatial_data (nt, ndim): new reach trajectory translated to the new origin
    '''
    new_spatial_data = np.subtract(spatial_data, new_origin)

    return new_spatial_data


def rotate_spatial_data(spatial_data, new_axis, current_axis):
    '''
    Rotates data about the origin into a new coordinate system based on the relationship between
    'new_axis' and 'current_axis'. If 'current_axis' and 'new_axis' point in 
    the same direction, the code will return 'spatial_data' with a warning that the vectors point in
    the same direction.
    
    This function was written to rotate spatial data but can be applied to other data of similar form.

    Args:
        spatial_data (nt, ndim): Array of spatial data in 2D or 3D
        new_axis (ndim): vector pointing along the desired orientation of the data
        current_axis (ndim): vector pointing along the current orientation of the dat

    Returns:
        output_spatial_data (nt, ndim): new reach trajectory rotated to the new axis
    '''

    # Check if input data is a single point and enfore that it is a row vector
    if len(spatial_data.shape) == 1:
      spatial_data.shape = (1,len(spatial_data))    

    # Initialize output array
    output_spatial_data = np.empty((spatial_data.shape[0], 3))

    # Check for a 2D or 3D trajectory and convert to 3D points
    if spatial_data.shape[1] == 2:
      spatial_data3d = np.concatenate((spatial_data, np.zeros((spatial_data.shape[0],1))), axis = 1)
      new_axis3d = np.concatenate((new_axis, np.array([0])))
      current_axis3d = np.concatenate((current_axis, np.array([0])))
    elif spatial_data.shape[1] == 3:
      spatial_data3d = spatial_data
      new_axis3d = new_axis
      current_axis3d = current_axis

    # Calcualte angle between 'new_axis3d' and target trajectory via dot product
    angle = np.arccos(np.dot(new_axis3d, current_axis3d)/(np.linalg.norm(new_axis3d)*np.linalg.norm(current_axis3d)))

    # If angle is 0, return the original data and warn
    if np.isclose(angle, 0, atol = 1e-8):
      warnings.warn("Starting and desired vector are the same. No rotation applied")
      output_spatial_data = spatial_data3d
      return output_spatial_data

    # If the angle is exactly 180 degrees, slightly nudge the starting vector by 1e-7
    elif np.isclose(angle, np.pi, atol = 1e-8):
      current_axis3d = current_axis3d.astype('float64')
      current_axis3d[0] += 1e-7

    # Calculate unit vector axis to rotate about via cross product
    rotation_axis = np.cross(current_axis3d, new_axis3d)
    unit_rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)

    # Calculate quaternion
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternions_as_rotations
    qr = np.cos(angle/2)
    qi = np.sin(angle/2)*unit_rotation_axis[0]
    qj = np.sin(angle/2)*unit_rotation_axis[1]
    qk = np.sin(angle/2)*unit_rotation_axis[2]

    # Convert quaternion to rotation matrix
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    rotation_matrix = np.array([[1-2*(qj**2 + qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr)],
                                [2*(qi*qj + qk*qr), 1-2*(qi**2 + qk**2), 2*(qj*qk - qi*qr)],
                                [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1-2*(qi**2 + qj**2)]])

    # Apply rotation matrix to each point in the trajectory
    for point_idx in range(spatial_data3d.shape[0]):
      output_spatial_data[point_idx,:] = rotation_matrix @ spatial_data3d[point_idx,:]

    # Return trajectories the the same dimensions as the input
    if spatial_data.shape[1] == 2:
      return output_spatial_data[:,:2]
    elif spatial_data.shape[1] == 3:
      return output_spatial_data

def calc_reward_intervals(timestamps, values):
    '''
    Given timestamps and values corresponding to reward times and reward state, calculate the
    intervals (start, end) during which the reward was active

    Args:
        timestamps (nt): when the reward transitioned state
        values (nt): what the state was at each corresponding timestamp

    Returns:
        (nt/2): during which the reward was active
    '''
    reward_ts_on = timestamps[values == 1]
    reward_ts_off = timestamps[values == 0]
    if len(reward_ts_on) == len(reward_ts_off):
        return list(zip(reward_ts_on, reward_ts_off))
    else:
        raise ValueError("Invalid reward timestamps or values")


def get_trial_targets(trials, targets):
    '''
    Organizes targets from each trial into a trial array of targets. Essentially reshapes the array,
    but sometimes? there can be more or fewer targets in certain trials than in others

    Args:
        trials (ntargets): trial number for each target presented
        targets (ntargets, 3): target locations
    
    Returns:
        (ntrials list of (ntargets, 3)): list of targets in each trial
    ''' 
    n_trials = np.max(trials) + 1
    trial_targets = [[] for _ in range(n_trials)]
    for idx in range(len(trials)):
        trial = trials[idx]
        trial_targets[trial].append(targets[idx])
    return trial_targets

def sample_events(events, times, samplerate):
    '''
    Converts a list of events and timestamps to a matrix of events where
    each column is a different event and each row is a sample in time.
    For example, if we have events 'reward' and 'penalty', and we want them
    as separate rasters::

        >>> events = ["reward", "reward", "penalty", "reward"]
        >>> times = [0.3, 0.5, 0.7, 1.0]
        >>> samplerate = 10
        >>> frame_events, event_names = sample_events(events, times, samplerate)
        >>> print(frame_events)
        [[False, False],
         [False, False],
         [False, False],
         [False, True ],
         [False, False],
         [False, True ],
         [False, False],
         [ True, False],
         [False, False],
         [False, False],
         [False, True ]]
        >>> print(event_names)
        ["penalty", "reward"]

    Args:
        events (list): list of event names or numbers
        times (list): list of timestamps for each event
        samplerate (float): rate at which you want to sample the events

    Returns:
        tuple: tuple containing:
            frame_events (nt, n_events): logical index of 'events' at the 
                given sampling rate
            event_names (n_events): list of event column names (sorted 
                alphabetically)

    '''
    n_samples = round(times[-1]*samplerate) + 1
    unique_events = np.unique(events)
    frame_events = np.zeros((n_samples, len(unique_events)), dtype='bool')
    for idx_event in range(len(events)):
        unique_idx = unique_events == events[idx_event]
        event_time = times[idx_event]
        event_frame = round(event_time * samplerate)
        frame_events[event_frame,unique_idx] = True
        
    return frame_events, unique_events

def get_calibrated_eye_data(eye_data, coefficients):
    """
    Apply the least square fitting coefficients to segments of eye data
    
    Args:
        eye_data (nt, nch): data to calibrate. Typically 4 channels (left eye x, left eye y, right eye x, right eye y)
        coefficients (nch, 2): coefficients to use for calibration for each channel of data
        
    returns:
        (nt, nch) ndarray: calibrated data
    """    
    #caliberated_eye_data_segments = np.empty((num_time_points, num_dims))
    return eye_data * coefficients[:,0] + coefficients[:,1]