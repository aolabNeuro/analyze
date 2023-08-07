import numpy as np

def get_saccade_pos(eye_pos, onset_times, duration, samplerate):
    '''
    Computes saccade positions when saccades start and end in a given trial
    
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

    onset_pos = np.zeros((onset_times.shape[0], eye_pos.shape[1]))
    offset_pos = np.zeros((onset_times.shape[0], eye_pos.shape[1]))
    
    for isaccade, (dur, onset) in enumerate(zip(duration, onset_times)): 
        offset = dur + onset
        onset_pos[isaccade,:] = eye_pos[int(onset*samplerate),:]
        offset_pos[isaccade,:] = eye_pos[int(offset*samplerate),:]
        
    return onset_pos, offset_pos
        
def get_saccade_label_center_out(onset_pos, offset_pos, target_pos, target_radius):
    '''
    Determines a target number a subject looked at before and after a saccade. When the subject looked at irrelevant areas, the target number gets -1.
    This function is for the standard center out task.
    
    Args:
        onset_pos (nsaccade, 2): eye positions when saccades start in a given trial
        onset_pos (nsaccade, 2): eye positions when saccades end in a given trial
        target_pos (ntarget, 2): target positions including the center target
        target_radius (float): target radius or larger radius to determine which targets a saccade is for
    
    Returns:
        tuple: tuple containing:
        | **onset_target (nsaccade):** target number a subject looked at when a saccade starts
        | **offset_target (nsaccade):** target number a subject looked at when a saccade ends    
    '''
    
    onset_target = []
    offset_target = []

    nsaccade = onset_pos.shape[0]
    for isaccade in range(nsaccade):        
        # Check if this saccade is for looking at a target or not
        onset_label = (onset_pos[isaccade,0] - target_pos[:,0])**2 + (onset_pos[isaccade,1] - target_pos[:,1])**2 <= target_radius**2
        offset_label = (offset_pos[isaccade,0] - target_pos[:,0])**2 + (offset_pos[isaccade,1] - target_pos[:,1])**2 <= target_radius**2

        # Get a target number of the saccade
        if np.any(onset_label):
            onset_label = np.where(onset_label)[0][0]
        else:
            onset_label = -1

        if np.any(offset_label):
            offset_label = np.where(offset_label)[0][0]
        else:
            offset_label = -1
        
        onset_target.append(onset_label)
        offset_target.append(offset_label)

    onset_target = np.array(onset_target)
    offset_target = np.array(offset_target)

    return onset_target, offset_target

def get_saccade_event(onset_times, duration, event_times, event_code):
    '''
    Returns event codes passed just before a saccade starts and ends
    
    Args:
        onset_times (nsaccade): saccade onset times (in seconds) generated in a trial segment.
        duration (nsaccade): saccade duration (in seconds) of saccades generated in a trial segment
        event_times (nevent): event times in a trial segment
        event_code (nevent): event codes in a trial segment
    
    Returns:
        tuple: tuple containing:
        | **onset_event (nsaccade):** event code passed when a saccade starts
        | **offset_event (nsaccade):** event code passed when a saccade ends
    '''
    
    onset_event = []
    offset_event = []
    
    for dur, onset in zip(duration, onset_times): 
        offset = dur + onset
        
        # Extract the last event code before and after a saccade
        onset_event_code = event_code[np.where(event_times < onset)[0][-1]]
        offset_event_code = event_code[np.where(event_times < offset)[0][-1]]
        
        onset_event.append(onset_event_code)
        offset_event.append(offset_event_code)
        
    onset_event = np.array(onset_event)
    offset_event = np.array(offset_event)
    
    return onset_event, offset_event