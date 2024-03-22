
import os
import numpy as np

from .base import get_preprocessed_filename, load_preproc_eye_data, save_hdf, find_preproc_ids_from_day
from ..postproc import get_calibrated_eye_data

def proc_eye_day(preproc_dir, subject, date, correlation_min=0.9, dry_run=False):
    '''
    Finds files from the given subject and date with the best eye calibration and automatically 
    applies it to every recording on that day for that subject. If no good calibration is found,
    raises a ValueError exception.
    
    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        date (str): Date of recording
        correlation_min (float, optional): correlation below which is unacceptable
        dry_run (bool, optional): if True, files will not be modified. 
        
    Raises:
        ValueError

    Returns:
        tuple: tuple containing:
            | **best_id (int)**: the task entry id with the highest mean absolute value correlation coefficient
            | **te_ids (list of int)**: the ids to which the coeff were applied
    '''
    
    # Find best calibration from the given subject and date 
    te_ids = find_preproc_ids_from_day(preproc_dir, subject, date, 'eye')
    if len(te_ids) == 0:
        print(f"No preprocessed files found on {date}")
        return None, []
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
        if correlation > best_correlation and correlation < 1.0: # 1.0 correlations aren't valid
            best_id = te_id
            best_coeff = eye_data['coefficients']
            best_correlation = correlation
            best_version = eye_metadata.get('calibration_version', 'unknown')
            best_date = eye_metadata.get('calibration_date', '2000-01-01')
    
    if best_correlation < correlation_min:
        raise ValueError(f"Could not find calibrated eye data with correlation > {correlation_min}"
                         f" for {subject} on {date} (best {best_correlation})")
        
    # Apply that calibration to all the other files
    te_ids = np.delete(te_ids, np.where(te_ids == best_id)[0])
    print(f"Applying eye calibration from {subject} block {best_id} on {date} (r={best_correlation})"
          f" to {len(te_ids)-1} files...")
    if dry_run:
        return best_id, te_ids

    for te_id in te_ids:
        eye_data, eye_metadata = load_preproc_eye_data(preproc_dir, subject, te_id, date)
        eye_data['calibrated_data'] = get_calibrated_eye_data(eye_data['raw_data'], best_coeff)
        eye_data['coefficients'] = best_coeff
        eye_metadata['external_calibration'] = True
        eye_metadata['external_calibration_id'] = best_id
        eye_metadata['calibration_version'] = best_version
        eye_metadata['calibration_date'] = best_date
        preproc_file = get_preprocessed_filename(subject, te_id, date, 'eye')
        preproc_dir_subject = os.path.join(preproc_dir, subject)
        save_hdf(preproc_dir_subject, preproc_file, eye_data, "/eye_data", append=True)
        save_hdf(preproc_dir_subject, preproc_file, eye_metadata, "/eye_metadata", append=True)

    return best_id, te_ids
