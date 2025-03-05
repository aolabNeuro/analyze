# base.py
#
# Code for post-processing neural data, including separating neural features such as 
# LFP bands or spikes detection / binning

import math
import warnings

import numpy as np
from scipy.signal import convolve

from .. import precondition

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

    Updates: July 2023 : updated function to work when new_axis is not a unit vector
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
    elif (spatial_data.shape[1] == 2) and np.isclose(angle, np.pi, atol = 1e-8):
        current_axis3d = current_axis3d.astype('float64')
        current_axis3d[:2] += 1e-7
    elif np.isclose(angle, np.pi, atol = 1e-8):
        raise ValueError("180 degree rotation cannot be resolved in 3D")

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
    output_spatial_data = rotation_matrix @ spatial_data3d.T
    output_spatial_data = output_spatial_data.T

    # Check that we did the correct rotation
    output_axis = rotation_matrix @ current_axis3d
    output_axis = output_axis/np.linalg.norm(output_axis)
    assert np.allclose(output_axis, new_axis3d/np.linalg.norm(new_axis3d), atol=1e-4), "Something went wrong!"

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

def get_minimum_trials_per_target(target_idx, cond_mask=None):
    '''
    Get the minimum number of trials per target after restricting trials
    
    Args:
        target_index (ntr): target index
        cond_mask (ntr): boolean array to remove trials
        
    Returns:
        (int): minimum number of trials per target
        
    '''
    
    if cond_mask is None:
        min_trial = min([sum(target_idx == itarget) for itarget in np.unique(target_idx)])
    else:
        min_trial = min([sum(target_idx[cond_mask] == itarget) for itarget in np.unique(target_idx)])
    
    return min_trial

def get_conditioned_trials_per_target(target_idx, trials_per_cond, cond_mask=None, replacement=False, seed=None):
    '''
    Get trial index to choose the same number of trials per target in removing trials by a certain condition.
    The trial index is evenly aligned like [1,2,3,1,2,3,1,2,3,...].
    trials_per_cond can be taken from 'get_minimum_trials_per_target' function.
    
    Args:
        target_index (ntr): target index
        trials_per_cond (int): minimum trial across conditions to get the same number of trials per condition
        cond_mask (ntr): boolean array to remove trials
        replacement (bool): whether to allow replacement in choosing trials. This can be used for bootstrapping.
        seed (int): random seed
        
    Returns:
        (ntr): trial index to extract the same number of conditioned trials for each target
        
    '''
    if seed is not None:
        np.random.seed(seed)
        
    # Get trial index to get the same number of trials per target
    for idx, itarget in enumerate(np.unique(target_idx)):
        if cond_mask is None:
            trial_mask_targ = np.where(target_idx == itarget)[0]
        else:
            trial_mask_targ = np.where(cond_mask * (target_idx == itarget))[0] 
                   
        if trial_mask_targ.size:
            if idx == 0:
                trial_mask = np.random.choice(trial_mask_targ, trials_per_cond, replace=replacement)
            else:
                trial_mask = np.vstack([trial_mask, np.random.choice(trial_mask_targ, trials_per_cond, replace=replacement)]) # conditions x trials shape

    # reshape using 'F' so that trial index would be aligned in the pseudorandom order
    trial_mask = trial_mask.reshape(-1, order='F')
    
    return trial_mask

def get_relative_point_location(ref_point_pos, new_point_pos):
    '''
    This function calculates the relative location (angle and position) of a point compared to a reference point.
    Assumes angle 0 starts from the direciton of vector [1, 0].
    This function is specific to the 2D case at the moment but can handle a single point entry or multiple.

    Args:
        ref_point_pos (nxpts, nypts): Point position of reference point in spatial coordinates
        new_point_pos (nxpts, nypts): Point position of new point in spatial coordinates

    Returns:
        A tuple containing:
            | **relative_new_point_angle (float):** Absolute angle between cursor and target position [rad]
            | **relative_new_point_pos (float):** Absolute position of the target relative to the cursor
    '''
    # Handle single point case
    relative_new_point_pos = new_point_pos - ref_point_pos

    if len(ref_point_pos.shape) == 1 or relative_new_point_pos.shape[1] == 1:
        relative_new_point_angle = np.arctan2(relative_new_point_pos[1], relative_new_point_pos[0])
        if relative_new_point_angle < 0:
            relative_new_point_angle = 2*np.pi + relative_new_point_angle
    
    # Handle multi-point case
    else:
        relative_new_point_angle = np.arctan2(relative_new_point_pos[:,1], relative_new_point_pos[:,0])
        relative_new_point_angle_mask = relative_new_point_angle < 0
        relative_new_point_angle[relative_new_point_angle_mask] = 2*np.pi + relative_new_point_angle[relative_new_point_angle_mask]

    return relative_new_point_angle, relative_new_point_pos

def get_inst_target_dir(trial_aligned_pos, targetpospertrial):
    '''
    This function calculates the instantaneous direction from the cursor to the target at each time point across each trial.
    This function is specific to the 2D case at the moment with X coordinates being the horizontal dimension and Y coordinates being the vertical dimension. 

    Args:
         trial_aligned_pos (ntime, ntrials, 2): Position of the cursor in X and Y coordinates.trial_aligned_pos[:,:,0] corresponds to the X cursor positions and trial_aligned_pos[:,:,1] to the Y cursor positions.
         targetpospertrial (ntrials, 2): X and Y pos of target.

    Returns:
        (ntime, ntrials): Array including instantaneous direction to the target from the cursor [rad]

    '''
    ntime = trial_aligned_pos.shape[0]
    ntrials = trial_aligned_pos.shape[1]
    inst_target_dir = np.zeros((ntime, ntrials))*np.nan

    for itrial in range(ntrials):
        cursor_location = trial_aligned_pos[:, itrial,:]
        target_location = targetpospertrial[itrial, :]
        rel_target_angle, _ = get_relative_point_location(cursor_location, target_location)
        inst_target_dir[:, itrial] = rel_target_angle

    return inst_target_dir

def mean_fr_inst_dir(data, trial_aligned_pos, targetpos, data_binwidth, ntarget_directions, data_samplerate, cursor_samplerate):
    '''
    This function takes trial aligned neural data, cursor position, and target position then calculates
    the mean firing rate per target location. Each target location is the instantaneous target direction from the 
    current cursor position, and therefore has multiple values during a single trial. The target locaitons are determined by calling aopy.postproc.get_inst_target_dir. 
    The target directions are assumed to be evenly spaced around the origin and the 0'th target starts directly horizontal from the origin.
    This function is specific to the 2D case at the moment with X coordinates being the horizontal dimension and Y coordinates being the vertical dimension. 

    Args:
        data (ntime, nunit, ntrial): Trial aligned data
        trial_aligned_pos (ntime, ntrials, 2): Position of the cursor in X and Y coordinates.trial_aligned_pos[:,:,0] corresponds to the X cursor positions and trial_aligned_pos[:,:,1] to the Y cursor positions.
        targetpos (ntrial, 2): Target position for each trial. First column is x target position, second column is y target position.
        data_binwidth (float): Bin size for neural data and cursor position. Can not be smaller than allowed by cursor position sampling rate.
        ntarget_directions (float): Number of directions to bin instantaneous direction into.
        data_samplerate (int): Sampling rate for data
        cursor_samplerate (int): Sampling rate for cursor position

    Returns:
        (nunit, ntarget_directions): Average firing rate per unit per direction bin. [spikes/s]
        
    '''
    # Check that binwidths are not lower than allowed by sampling rate
    max_cursor_binwidth = 1/cursor_samplerate #[s/sample]
    if data_binwidth < max_cursor_binwidth:
        data_binwidth = max_cursor_binwidth
    
    ndatatime, nunit, ntrial = data.shape
    nbins = math.ceil(ndatatime/(data_samplerate*data_binwidth)) # the number of bins

    # Bin neural data and cursor pos for each trial to make them the same samplingrate
    binned_data = np.zeros((nbins, nunit, ntrial))*np.nan
    for itrial in range(ntrial):
        binned_data[:,:,itrial] = precondition.bin_spikes(data[:,:,itrial], data_samplerate, data_binwidth)

    # (use precondition.bin_spikes to get average value in each bin)
    binned_cursorxpos = precondition.bin_spikes(trial_aligned_pos[:,:,0], cursor_samplerate, data_binwidth)/cursor_samplerate
    binned_cursorypos = precondition.bin_spikes(trial_aligned_pos[:,:,1], cursor_samplerate, data_binwidth)/cursor_samplerate
    binned_cursorpos = np.concatenate((np.expand_dims(binned_cursorxpos,axis=2),np.expand_dims(binned_cursorypos,axis=2)), axis=2)

    # Get instantaneous target location for each cursor pos (ntime, ntrial)
    inst_target_dir = get_inst_target_dir(binned_cursorpos, targetpos)

    # Match the instantaneous direction to the correct target direction bin and ensure the bin
    # range starts entered on target 0 (directly horizontal from the origin)
    targetloc_binwidth = (2*np.pi)/ntarget_directions # [rad] Angular bin size of each target direction
    target_binid = 1 + (inst_target_dir-(targetloc_binwidth/2))//targetloc_binwidth
    target_binid[target_binid==ntarget_directions] = 0 #combine first and last bins

    # Average data and place into correct points in array
    mean_dir_fr = np.zeros((nunit, ntarget_directions))
    for iunit in range(nunit):
        for idir in range(ntarget_directions):
            temp_data = binned_data[:,iunit,:]
            mean_dir_fr[iunit, idir] = np.nanmean(temp_data[target_binid==idir])

    return mean_dir_fr

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
            | **frame_events (nt, n_events):** logical index of 'events' at the given sampling rate
            | **event_names (n_events):** list of event column names (sorted alphabetically)

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

def smooth_timeseries_gaus(timeseries_data, sd, samplerate, nstd=3, conv_mode='same', return_kernel=False):
    """
    Smooths a time series by convolving it with a user-defined Gaussian kernel. 
    The length of the kernel is computed using integers in the following range:

    .. math::

        x = [\\frac{-nstd*sd*samplerate}{1000}, \\frac{nstd*sd*samplerate}{1000}]


    The Gaussian kernel is then defined by the following equation:

    .. math::

        g(x) = \\frac{1}{\sigma\sqrt{2\pi}}e^{\\frac{-x^{2}}{2\sigma^{2}}}

    
    Args:
        timeseries_data (ndarray): The input time series data to be smoothed (i.e. binned firing rates). This can be an array with any number of dimensions, 
                                    but the smoothing is applied along the first axis (time).
        sd (float): The width of the standard deviation of the Gaussian filter in time [ms]. 
        samplerate (int): The sample rate of the time series data [Hz].
        nstd (float or int, optional): The number of standard deviations to be used in the filter calculation. 
                                       Default is 3.
        conv_mode (str, optional): The convolution mode, which defines the size of the output. If the default input of 'same' is used, the edges will be computed with zero padding.
                                   Accepts 'full', 'valid', or 'same'. Default is 'same'.
                                   See `scipy.signal.convolve` for full documentation on the modes.
        return_kernel (bool, optional): If True, this function returns the kernel used in convolution. Otherwise, only the smoothed data is returned. 

    Returns:
        ndarray: The smoothed time series data. The shape of the output matches the input data, except in 
                 the case of 'valid' or 'full' convolution modes where the output may be smaller depending 
                 on the filter size.

    Example:
        timeseries_data = np.random.randn(1000)  # Example time series data
        smoothed_data = smooth_timeseries_gaus(timeseries_data, 1000, 50)

        .. image:: _images/gaus_smoothing_example.png
    """
    sample_std = sd * samplerate / 1000  # Convert from ms to samples
    x = np.arange(-sample_std * nstd, (nstd * sample_std) + 1)
    gaus_filter = (1 / (sample_std * np.sqrt(2 * np.pi))) * np.exp(-(x ** 2) / (2 * sample_std ** 2))
    if return_kernel:
        return np.apply_along_axis(convolve, 0, timeseries_data, gaus_filter, mode=conv_mode, method='direct'), gaus_filter
    else:
        return np.apply_along_axis(convolve, 0, timeseries_data, gaus_filter, mode=conv_mode, method='direct')
