from .. import precondition
from .. import data as aodata
import numpy as np

def parse_oculomatic(data_dir, files, samplerate=480, debug=True):
    """
    Loads eye data from ecube and hdf data. 
    
    Data includes:
        data (nt, nch): eye data in volts
        
    Metadata includes:
        channels (nch): analog channels on which eye data was recorded
        labels (nch): string labels for each eye channel
        samplerate (float): sampling rate of the eye data
        units (str): measurement unit of eye data

    Args:
        data_dir (str): folder containing the data you want to load
        files (dict): a dictionary that has 'ecube' as the key
        samplerate (float, optional): sampling rate to output in Hz. Default 480. 
        debug (bool, optional): prints debug information

    Returns:
        tuple: tuple contatining:
            | **eye_data (nt, neyech):** voltage per eye channel (normally [left eye x, left eye y, right eye x, right eye y])
            | **eye_metadata (dict):** metadata associated with the eye data, including the above labels
    """
    
    eye_metadata = dict()
    
    if 'hdf' in files:
        bmi3d_events, bmi3d_event_metadata = aodata.load_bmi3d_hdf_table(data_dir, files['hdf'], 'sync_events')

        # get eye channels 
        if 'left_eye_ach' in bmi3d_event_metadata and 'right_eye_ach' in bmi3d_event_metadata:
            eye_channels = bmi3d_event_metadata['left_eye_ach'] + bmi3d_event_metadata['right_eye_ach']
            if debug: print(f'use bmi3d supplied eye channel definition {eye_channels}')
        else:
            eye_channels = [9, 8, 10, 11]
            if debug: print(f'eye channel definitions do not exist, use eye channels {eye_channels} ')
    else:
        # from https://github.com/aolabNeuro/analyze/issues/225
        eye_channels = [10, 11, 8, 9]
        if debug: print(f'No metadata from BMI3D, assuming eye channels {eye_channels} ')
        
    eye_metadata['n_channels'] = len(eye_channels)
    eye_metadata['channels'] = eye_channels
    eye_metadata['labels']  = ['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y']
    
    # Get eye data and downsample
    analog_data, analog_metadata = aodata.load_ecube_analog(data_dir, files['ecube'], channels=eye_channels)
    raw_samplerate = analog_metadata['samplerate']
    downsample_data = precondition.downsample(analog_data, raw_samplerate, samplerate)
    eye_metadata['samplerate'] = samplerate
    eye_metadata['n_samples'] = downsample_data.shape[0]
    
    #scale eye data from bits to volts
    if 'voltsperbit' in analog_metadata:
        analog_voltsperbit = analog_metadata['voltsperbit']
    else:
        analog_voltsperbit = 3.0517578125e-4
    eye_metadata['units'] = 'volts'
        
    eye_data = {
        'data': downsample_data * analog_voltsperbit
    }
    return eye_data, eye_metadata

def detect_noise(eye_data, samplerate, step_thr=3, t_closed_min=0.1):
    '''
    Detect noise in oculomatic eye data.
    
    Args:
        eye_data (): unfiltered raw or calibrated eye position data
        samplerate
        t_closed_min
        
    Returns:
        eye_closed_mask
    '''
    
    time = np.arange(eye_data.shape[0])/samplerate
    eye_closed_mask = np.zeros(eye_data.shape, dtype='bool')

    # Detect when the eyes are closed -- oculomatic specific
    for eye_idx in range(eye_data.shape[1]):
        
        # Find where value changes from one sample to next
        diff = abs(np.diff(eye_data[:,eye_idx], axis=0))
        step_size = np.min(diff[diff > 0])
        change = diff > step_thr*step_size
        change = np.insert(change, 0, True)
        change_idx = np.where(change)[0]
        
        # Count the length of the gaps between those changes
        repetitions = np.diff(change_idx)
        repetitions = np.insert(repetitions, 0, True)

        # Mask anything that is longer than a predetermined length
        suspicious = repetitions > int(samplerate*t_closed_min)
        suspicious_idx = change_idx[suspicious]
        durations = repetitions[suspicious]
        for idx in range(len(suspicious_idx)):
            idx_end = suspicious_idx[idx]
            idx_start = idx_end - durations[idx]
            eye_closed_mask[idx_start:idx_end,eye_idx] = 1
            
    return eye_closed_mask
