# oculomatic.py
# 
# preprocessing eye data from oculomatic

import os
import numpy as np

from ..data.bmi3d import load_ecube_data_chunked
from .. import precondition
from .. import data as aodata
from .. import utils

def parse_oculomatic(data_dir, files, samplerate=1000, max_memory_gb=1.0, debug=True, **filter_kwargs):
    """
    Loads eye data from ecube and hdf data. 

    .. image:: _images/proc_oculomatic_downsample.png
    .. image:: _images/proc_oculomatic_mask.png
    
    Data includes:
        data (nt, nch): eye data in volts
        mask (nt, nch): boolean mask of when the eyes are closed
        
    Metadata includes:
        channels (nch): analog channels on which eye data was recorded
        labels (nch): string labels for each eye channel
        samplerate (float): sampling rate of the eye data
        units (str): measurement unit of eye data

    Args:
        data_dir (str): folder containing the data you want to load
        files (dict): a dictionary that has 'ecube' as the key
        samplerate (float, optional): sampling rate to output in Hz. Default 1000. 
        max_memory_gb (float, optional): max memory used to load binary data at one time
        debug (bool, optional): prints debug information
        filter_kwargs (kwargs, optional): optional keyword arguments to send to `filter_eye()`
        
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
        
    eye_metadata['source'] = 'oculomatic voltage output'
    eye_metadata['n_channels'] = len(eye_channels)
    eye_metadata['channels'] = eye_channels
    eye_metadata['labels']  = ['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y']
    
    # Create an empty array for the downsampled data
    data_path = os.path.join(data_dir, files['ecube'])
    analog_metadata = aodata.load_ecube_metadata(data_path, 'AnalogPanel')
    dtype = 'float'
    chunksize = int(max_memory_gb * 1e9 / np.dtype(dtype).itemsize / eye_metadata['n_channels'])
    downsample_factor = int(analog_metadata['samplerate']/samplerate)
    n_samples = int(np.ceil(analog_metadata['n_samples']/downsample_factor))
    n_channels = int(eye_metadata['n_channels'])
    downsample_data = np.zeros((n_samples, n_channels), dtype=dtype)
    downsample_mask = np.zeros((n_samples, n_channels), dtype='bool')

    # Filter broadband data into LFP directly into the hdf file
    n_samples = 0
    for analog_chunk in load_ecube_data_chunked(data_path, 'AnalogPanel', channels=eye_channels, chunksize=chunksize):
        mask_chunk = detect_noise(analog_chunk, analog_metadata['samplerate'], min_step=3, step_thr=3)
        # mask_chunk = precondition.downsample(mask_chunk, analog_metadata['samplerate'], samplerate)
        # mask_chunk = mask_chunk > 0.5 # downsample takes a mean over boolean, this makes it back into a boolean
        mask_chunk = mask_chunk[::downsample_factor,:]
        analog_chunk, _ = precondition.filter_eye(analog_chunk, analog_metadata['samplerate'], downsamplerate=samplerate, **filter_kwargs)
        chunk_len = analog_chunk.shape[0]
        downsample_data[n_samples:n_samples+chunk_len,:] = analog_chunk
        downsample_mask[n_samples:n_samples+chunk_len,:] = mask_chunk
        n_samples += chunk_len

    eye_metadata['source'] = data_path
    eye_metadata['raw_samplerate'] = analog_metadata['samplerate']
    eye_metadata['samplerate'] = samplerate
    eye_metadata['n_samples'] = downsample_data.shape[0]
    eye_metadata['taper_len'] = 0.05
    eye_metadata['lowpass_freq'] = 30
    eye_metadata['pad_t'] = 1.0
    eye_metadata.update(filter_kwargs)
    
    #scale eye data from bits to volts
    if 'voltsperbit' in analog_metadata:
        analog_voltsperbit = analog_metadata['voltsperbit']
    else:
        analog_voltsperbit = 3.0517578125e-4
    eye_metadata['units'] = 'volts'
        
    eye_data = {
        'data': downsample_data * analog_voltsperbit,
        'mask': downsample_mask
    }
    return eye_data, eye_metadata

def detect_noise(eye_data, samplerate, min_step=None, step_thr=3, t_closed_min=0.1):
    '''
    Detect noise in oculomatic eye data. Searches for repeated data which indicates
    that the subject's eyes were closed and returns a mask for each eye. Uses the
    minimum step size in the data to estimate when the voltage has "changed" versus
    when it is fluctuating because of measurement noise.
    
    Args:
        eye_data ((nt,nch) array): unfiltered raw or calibrated eye position data
        samplerate (float): sampling rate of the eye data
        min_step (float, optional): minimum step size in the data. If None, it will be 
            calculated automatically. Default None.
        step_thr (float, optional): multiple of the minimum step size to use as a threshold for 
            detecting changing values in the eye data
        t_closed_min (float, optional): number of seconds the data must remain unchanged to be
            included in the eye closed mask
        
    Returns:
        (nt, nch) eye_closed_mask: boolean mask, True for each eye when it was probably closed.
    '''
    eye_closed_mask = np.zeros(eye_data.shape, dtype='bool')

    # Detect when the eyes are closed -- oculomatic specific
    for eye_idx in range(eye_data.shape[1]):
        
        # Find where value changes from one sample to next
        diff = abs(np.diff(eye_data[:,eye_idx]))
        if min_step is None:
            step_size = step_thr*np.min(diff[diff > 0])
        else:
            step_size = step_thr*min_step
        repetitions, repetitions_idx = utils.count_repetitions(eye_data[:,eye_idx], step_size)
        # Mask anything that is longer than a predetermined length
        suspicious = repetitions > int(samplerate*t_closed_min)
        suspicious_idx = repetitions_idx[suspicious]
        durations = repetitions[suspicious]
        for idx in range(len(suspicious_idx)):
            idx_start = suspicious_idx[idx]
            idx_end = idx_start + durations[idx]
            eye_closed_mask[idx_start:idx_end,eye_idx] = 1
            
    return eye_closed_mask
