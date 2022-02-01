from .. import data as aodata

def parse_oculomatic(data_dir, files, debug=True):
    """
    Loads eye data from ecube and hdf data

    Args:
        data_dir (str): folder containing the data you want to load
        files (dict): a dictionary that has 'ecube' as the key
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
        
    eye_metadata['channels'] = eye_channels
    eye_metadata['labels']  = ['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y']
    
    # get eye data
    analog_data, analog_metadata = aodata.load_ecube_analog(data_dir, files['ecube'], channels=eye_channels)
    eye_metadata['samplerate'] = analog_metadata['samplerate']
    
    #scale eye data from bits to volts
    if 'voltsperbit' in analog_metadata:
        analog_voltsperbit = analog_metadata['voltsperbit']
    else:
        analog_voltsperbit = 3.0517578125e-4
        eye_metadata['voltsperbit'] = analog_voltsperbit
        
    eye_data = {
        'data': analog_data * analog_voltsperbit
    }
    return eye_data, eye_metadata