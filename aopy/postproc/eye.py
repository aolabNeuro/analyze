import numpy as np

def get_saccade_pos(eye_pos, onset_times, duration, samplerate):
    '''
    Returns the coordinates of the start and end of each given saccade
    
    Args:
        eye_pos (nt,nch): eye position data
        onset_times (nsaccade): saccade onset times (in seconds) generated in a trial segment.
        duration (nsaccade): saccade duration (in seconds) of saccades generated in a trial segment
        samplerate (float): sampling rate
    
    Returns:
        tuple: tuple containing:
            | **onset_pos (nsaccade,nch):** eye positions when saccades start in a given trial
            | **offset_pos (nsaccade,nch):** eye positions when saccades end in a given trial 
    '''    

    # Convert onset times and durations to sample indices
    onset_indices = (onset_times * samplerate).astype(int)
    offset_indices = ((onset_times + duration) * samplerate).astype(int)
    
    # Get eye positions for onset and offset indices
    onset_pos = eye_pos[onset_indices, :]
    offset_pos = eye_pos[offset_indices, :]
    
    return onset_pos, offset_pos
        
def get_saccade_target_index(onset_pos, offset_pos, target_pos, target_radius):
    '''
    Determines a target index a subject looked at during a saccade. When the subject 
    looked at irrelevant areas, the target index is -1.
    
    Args:
        onset_pos (nsaccade, 2): eye positions when saccades start in a given trial
        offset_pos (nsaccade, 2): eye positions when saccades end in a given trial
        target_pos (ntarget, 2): target positions, e.g. center and peripheral targets
        target_radius (float): radius around the target center to search
    
    Returns:
        tuple: tuple containing:
            | **onset_target (nsaccade):** target index at saccade start
            | **offset_target (nsaccade):** target index at saccade end    
    '''
    
    # Calculate distances between saccade positions and target positions
    onset_dists = np.linalg.norm(onset_pos[:, np.newaxis, :] - target_pos, axis=2)
    offset_dists = np.linalg.norm(offset_pos[:, np.newaxis, :] - target_pos, axis=2)
    
    # Determine if the saccades are directed towards any target
    onset_label = onset_dists <= target_radius
    offset_label = offset_dists <= target_radius
    
    # Get target numbers for onset and offset
    onset_target = np.argmax(onset_label, axis=1)
    offset_target = np.argmax(offset_label, axis=1)
    
    # Set target number to -1 for saccades not directed at any target
    onset_target[~np.any(onset_label, axis=1)] = -1
    offset_target[~np.any(offset_label, axis=1)] = -1
    
    return onset_target, offset_target

def get_saccade_event(onset_times, duration, event_times, event_codes):
    '''
    Returns event codes corresponding to times in which the start and end of 
    each given saccade falls. 
    
    Args:
        onset_times (nsaccade): saccade onset times (in seconds)
        duration (nsaccade): saccade duration (in seconds) of saccades in the onset_times list
        event_times (nevent): list of event times (in seconds)
        event_codes (nevent): event codes corresponding to the given event_times
    
    Returns:
        tuple: tuple containing:
            | **onset_event (nsaccade):** event in which each saccade starts
            | **offset_event (nsaccade):** event in which each saccade ends
    '''
    
    # Calculate saccade offset times
    offset_times = onset_times + duration
    
    # Each index is the event preceding the saccade
    onset_indices = np.searchsorted(event_times, onset_times, side='right') - 1
    offset_indices = np.searchsorted(event_times, offset_times, side='right') - 1
    
    # Get event codes using the indices
    onset_event = event_codes[onset_indices]
    offset_event = event_codes[offset_indices]
    
    return onset_event, offset_event

def get_relevant_saccade_idx(onset_target, offset_target, saccade_distance, target_idx):
    '''
    Finds a relevant saccade index which is for looking at the peripheral target from the center target in a given trial
    onset_target and offset_target can be obtained by get_saccade_target_index.
    If there are multiple relevant saccades, choose the saccade whose distance is the largest among other saccades
    If there is no relevant saccades, the saccade index becomes -1
    
    Args:
        onset_target (nsaccade): target index at saccade start in a given trial
        offset_target (nsaccade): target index at saccade end in a given trial  
        saccade_distance (nsaccade): eye movement distance in a given trial
        target_idx (int): target index in a given trial
        
    Returns:
        (int): relevant saccade index with the largest distance among saccades. index becomes -1 if there is no relevant saccades.
    '''
    
    # Get saccade target index when saccade starts and ends
    saccade_target_index = np.vstack([onset_target,offset_target])
    
    # Get saccade target index that relevant saccade (saccade for the peripheral target) should have
    relevant_saccade_target_idx = np.array([0, int(target_idx)])

    dist = []
    saccade_idx = []
    relevant_saccade_idx = -1
    
    # Check if each saccade is relevant or not
    for isaccade in range(onset_target.shape[0]):

        # find a saccade for the peripheral target
        if np.all(saccade_target_index[:,isaccade] == relevant_saccade_target_idx):
            saccade_idx.append(isaccade) # get saccade index about potential relevant saccades
            dist.append(saccade_distance[isaccade]) # get eye movement distance about potential relevant saccades
    
    # Get a relevant saccade index with the largest distance
    if len(saccade_idx):
        relevant_saccade_idx = saccade_idx[np.argmax(dist)]  
        
    return relevant_saccade_idx