
import os
import numpy as np

from .base import get_preprocessed_filename, load_preproc_eye_data, save_hdf, find_preproc_ids_from_day
from ..postproc import get_calibrated_eye_data

def apply_eye_calibration(coeff, preproc_dir, subject, te_id, date):
    '''
    Apply eye calibration coefficients to a given preprocessed file.
    
    Args:
        coeff
        preproc_dir
        subject
        te_id
        date
    '''
    eye_data, eye_metadata = load_preproc_eye_data(preproc_dir, subject, te_id, date)
    eye_data['calibrated_data'] = get_calibrated_eye_data(eye_data['raw_data'], coeff)
    eye_data['coefficients'] = coeff
    eye_metadata['external_calibration'] = True
    preproc_file = get_preprocessed_filename(subject, te_id, date, 'eye')
    save_hdf(preproc_dir, preproc_file, eye_data, "/eye_data", append=True)
    save_hdf(preproc_dir, preproc_file, eye_metadata, "/eye_metadata", append=True)

def proc_eye_day(preproc_dir, subject, date, correlation_min=0.9):
    '''
    Finds files from the given subject and date with the best eye calibration and automatically 
    applies it to every recording on that day for that subject. If no good calibration is found,
    raises a ValueError exception. 
    
    Args:
        preproc_dir
        subject
        date
        correlation_min (float, optional): correlation below which is unacceptable
        
    Raises:
        ValueError
    '''
    
    # Find best calibration from the given subject and date 
    te_ids = find_preproc_ids_from_day(preproc_dir, subject, date, 'eye')
    best_id = None
    best_coeff = None
    best_correlation = 0
    for te_id in te_ids:
        eye_data, eye_metadata = load_preproc_eye_data(preproc_dir, subject, te_id, date)
        if 'external_calibration' in eye_metadata and eye_metadata['external_calibration']:
            continue # ignore the file if it has already had another calibration applied to it
        if 'correlation_coeff' not in eye_data:
            continue # ignore if there isn't any calibration data
        correlation = np.mean(abs(eye_data['correlation_coeff']))
        print(correlation)
        if correlation > best_correlation:
            best_id = te_id
            best_coeff = eye_data['coefficients']
            best_correlation = correlation
    
    if best_correlation < correlation_min:
        raise ValueError(f"Could not find calibrated eye data with correlation > {correlation_min}"
                         f" for {subject} on {date} (best {best_correlation})")
        
    # Apply that calibration to all the other files
    print(f"Applying eye calibration from {subject} block {te_id} on {date} (r={best_correlation})"
          f" to {len(te_ids)-1} files...")
    for te_id in te_ids:
        if te_id == best_id:
            continue
        apply_eye_calibration(best_coeff, preproc_dir, subject, te_id, date)
    
