from .base import *
from .bmi3d import parse_bmi3d
from .oculomatic import parse_oculomatic
from .optitrack import parse_optitrack
from .. import postproc
from ..data import load_ecube_data_chunked, load_ecube_metadata, proc_ecube_data, save_hdf, load_hdf_group, get_hdf_dictionary
import os
import h5py
from glob import glob

'''
proc_* wrappers
'''
def proc_day(data_dir, result_dir, files, overwrite=False, save_res=True, proc_exp=True, proc_eyetracking=True, proc_broadband=True, proc_lfp=True):
    if proc_exp:
        exp_result_filename = 'experiment_data.h5'
        print('processing experiment data...')
        proc_exp(
            data_dir,
            files,
            result_dir,
            result_filename = exp_result_filename,
            overwrite = overwrite,
            save_res = save_res
        )
    if proc_eyetracking:
        eyetracking_result_filename = 'eyetracking_data.h5'
        print('processing eyetracking data...')
        proc_eyetracking(
            data_dir,
            files,
            result_dir,
            result_filename = eyetracking_result_filename,
            debug = False, #TODO: is this supposed to be True by default?
            overwrite = overwrite,
            save_res = save_res,
        )
    if proc_broadband:
        broadband_result_filename = 'broadband_data.h5'
        print('processing broadband data...')
        proc_broadband(
            data_dir,
            files,
            result_dir,
            result_filename=broadband_result_filename,
            overwrite=overwrite
        )
    if proc_lfp:
        lfp_result_filename = 'lfp_data.h5'
        print('processing local field potential data...')
        proc_lfp(
            data_dir,
            files,
            result_dir,
            result_filename = lfp_result_filename,
            overwrite=overwrite,
            filter_kwargs={'ayy':'lmao'}
        )

def proc_exp(data_dir, files, result_dir, result_filename, overwrite=False, save_res=True):
    '''
    Process experiment data files: 
        Currently supports BMI3D only
        Loads 'hdf' and 'ecube' (if present) data, parses, and prepares experiment data and metadata
    The above data is prepared into structured arrays:
        exp_data:
            task ([('cursor', '<f8', (3,)), ('trial', 'u8', (1,)), ('time', 'u8', (1,)), ...])
            state ([('msg', 'S', (1,)), ('time', 'u8', (1,))])
            clock ([('timestamp', 'f8', (1,)), ('time', 'u8', (1,))])
            events ([('timestamp', 'f8', (1,)), ('time', 'u8', (1,)), ('event', 'S32', (1,)), 
                ('data', 'u2', (1,)), ('code', 'u2', (1,))])
            trials ([('trial', 'u8'), ('index', 'u8'), (condition_name, 'f8', (3,)))])
        exp_metadata:
            source_dir (str)
            source_files (dict)
            bmi3d_start_time (float)
            n_cycles (int)
            n_trials (int)
            <other metadata from bmi3d>
    
    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool): whether to remove existing processed files if they exist

    Returns:
        None
    '''   
    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "exp_data" in contents or "exp_metadata" in contents:
            print("File {} already preprocessed, doing nothing.".format(result_filename))
            return
    
    # Prepare the BMI3D data
    if 'hdf' not in files:
        print("No HDF data found.")
        return

    bmi3d_data, bmi3d_metadata = parse_bmi3d(data_dir, files)
    if save_res:
        save_hdf(result_dir, result_filename, bmi3d_data, "/exp_data", append=True)
        save_hdf(result_dir, result_filename, bmi3d_metadata, "/exp_metadata", append=True)
    return bmi3d_data, bmi3d_metadata

def proc_mocap(data_dir, files, result_dir, result_filename, overwrite=False):
    '''
    Process motion capture files:
        Loads metadata, position data, and rotation data from 'optitrack' files
        If present, reads 'hdf' metadata to find appropriate strobe channel
        If present, loads 'ecube' analog data representing optitrack camera strobe
    The data is prepared along with timestamps into HDF datasets:
        mocap_data:
            optitrack [('position', 'f8', (3,)), ('rotation', 'f8', (4,)), ('timestamp', 'f8', (1,)]
        mocap_metadata:
            source_dir (str)
            source_files (dict)
            samplerate (float)
            <other metadata from motive>
    
    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool): whether to remove existing processed files if they exist

    Returns:
        None
    '''  
    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "mocap_data" in contents or "mocap_metadata" in contents:
            print("File {} already preprocessed, doing nothing.".format(result_filename))
            return

    # Parse Optitrack data
    if 'optitrack' in files:
        optitrack_data, optitrack_metadata = parse_optitrack(data_dir, files)
        save_hdf(result_dir, result_filename, optitrack_data, "/mocap_data", append=True)
        save_hdf(result_dir, result_filename, optitrack_metadata, "/mocap_metadata", append=True)

def proc_eyetracking(data_dir, files, result_dir, result_filename, debug=True, overwrite=False, save_res=True, **kwargs):
    '''
    Loads eyedata from ecube analog signal and calculates calibration profile using least square fitting.
    Requires that experimental data has already been preprocessed in the same result hdf file.
    
    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_dir (str): where to store the processed result 
        result_filename (str): what to call the preprocessed filename
        debug (bool, optional): if true, prints additional debug messages
        overwrite (bool, optional): whether to recalculated and overwrite existing preprocessed eyetracking data
        save_res (bool, optional): whether to save the calculated eyetracking data
        **kwargs (dict, optional): keyword arguments to pass to :func:`aopy.preproccalc_eye_calibration()`

    Returns:
        eye_dict (dict): all the data pertaining to eye tracking, calibration
        eye_metadata (dict): metadata for eye tracking
    '''
    # Check if data already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "eye_data" in contents and "eye_metadata" in contents:
            print("Eye data already preprocessed in {}, returning existing data.".format(result_filename))
            eye_data = load_hdf_group(result_dir, result_filename, 'eye_data')
            eye_metadata = load_hdf_group(result_dir, result_filename, 'eye_metadata')
            return eye_data, eye_metadata
    
    # Load the preprocessed experimental data
    try:
        exp_data = load_hdf_group(result_dir, result_filename, 'exp_data')
        exp_metadata = load_hdf_group(result_dir, result_filename, 'exp_metadata')
    except (FileNotFoundError, ValueError):
        raise ValueError(f"File {result_filename} does not include preprocessed experimental data. Please call proc_exp() first.")
    
    # Parse the raw eye data; this could be extended in the future to support other eyetracking hardware
    eye_data, eye_metadata = parse_oculomatic(data_dir, files, debug=debug)
    
    # Calibrate the eye data
    cursor_data = exp_data['task']['cursor'][:,[0,2]] # cursor (x, z) position on each bmi3d cycle
    clock = exp_data['clock']
    events = exp_data['events']
    eye_data = eye_data['data']
    event_cycles = events['time'] # time points in bmi3d cycles
    event_codes = events['code']
    event_times = clock['timestamp_sync'][events['time']] # time points in the ecube time frame
    coeff, correlation_coeff, cursor_calibration_data, eye_calibration_data = calc_eye_calibration(
        cursor_data, exp_metadata['fps'], eye_data, eye_metadata['samplerate'], 
        event_cycles, event_times, event_codes, debug=debug, return_datapoints=True, **kwargs)
    calibrated_eye_data = postproc.get_calibrated_eye_data(eye_data, coeff)

    # Save everything into the HDF file
    eye_dict = {
        'raw_data': eye_data,
        'calibrated_data': calibrated_eye_data,
        'coefficients': coeff,
        'correlation_coeff': correlation_coeff,
        'cursor_calibration_data': cursor_calibration_data,
        'eye_calibration_data': eye_calibration_data
    }
    if save_res:
        save_hdf(result_dir, result_filename, eye_dict, "/eye_data", append=True)
        save_hdf(result_dir, result_filename, eye_metadata, "/eye_metadata", append=True)
    return eye_dict, eye_metadata


def proc_broadband(data_dir, files, result_dir, result_filename, overwrite=False, max_memory_gb=1.):
    '''
    Process broadband data:
        Loads 'ecube' headstage data and metadata
    Saves broadband data into the HDF datasets:
        broadband_data (nt, nch)
        broadband_metadata (dict)

    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool, optional): whether to remove existing processed files if they exist
        max_memory_gb (float, optional): max memory used to load binary data at one time

    Returns:
        None
    '''

    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "broadband_data" in contents:
            raise FileExistsError("File {} already preprocessed, doing nothing.".format(result_filename))
    elif os.path.exists(filepath):
        os.remove(filepath) # maybe bad, since it deletes everything, not just broadband data

    # Copy the broadband data into an HDF dataset
    if 'ecube' in files:
        
        # Process the binary data
        data_filepath = os.path.join(data_dir, files['ecube'])
        result_filepath = os.path.join(result_dir, result_filename)
        _, metadata = proc_ecube_data(data_filepath, 'Headstages', result_filepath, result_name='broadband_data', max_memory_gb=max_memory_gb)

        # Append the broadband metadata to the file
        save_hdf(result_dir, result_filename, metadata, "/broadband_metadata", append=True)

def proc_lfp(data_dir, files, result_dir, result_filename, overwrite=False, max_memory_gb=1., filter_kwargs={}):
    '''
    Process lfp data:
        Loads 'ecube' headstage data and metadata
    Saves broadband data into the HDF datasets:
        lfp_data (nt, nch)
        lfp_metadata (dict)
    
    Args:
        data_dir (str): where the data files are located
        files (dict): dictionary of filenames indexed by system
        result_filename (str): where to store the processed result
        overwrite (bool, optional): whether to remove existing processed files if they exist
        batchsize (float, optional): time in seconds for each batch to be processed into lfp
        filter_kwargs (dict, optional): keyword arguments to pass to :func:`aopy.precondition.filter_lfp`
    Returns:
        None
    '''  
    # Check if a processed file already exists
    filepath = os.path.join(result_dir, result_filename)
    if not overwrite and os.path.exists(filepath):
        contents = get_hdf_dictionary(result_dir, result_filename)
        if "lfp_data" in contents:
            print("File {} already preprocessed, doing nothing.".format(result_filename))
            return
    elif os.path.exists(filepath):
        os.remove(filepath) # maybe bad, since it deletes everything, not just lfp_data

    # Preprocess neural data into lfp
    if 'ecube' in files:
        data_path = os.path.join(data_dir, files['ecube'])
        metadata = load_ecube_metadata(data_path, 'Headstages')
        samplerate = metadata['samplerate']
        chunksize = int(max_memory_gb * 1e9 / np.dtype(dtype).itemsize / metadata['n_channels'])
        lfp_samplerate = filter_kwargs.pop('lfp_samplerate', 1000)
        downsample_factor = int(samplerate/lfp_samplerate)
        lfp_samples = np.ceil(metadata['n_samples']/downsample_factor)
        n_channels = metadata['n_channels']
        dtype = 'int16'

        # Create an hdf dataset
        result_filepath = os.path.join(result_dir, result_filename)
        hdf = h5py.File(result_filepath, 'a') # should append existing or write new?
        dset = hdf.create_dataset('lfp_data', (lfp_samples, n_channels), dtype=dtype)

        # Filter broadband data into LFP directly into the hdf file
        n_samples = 0
        for broadband_chunk in load_ecube_data_chunked(data_path, 'Headstages', chunksize=chunksize):
            lfp_chunk = precondition.filter_lfp(broadband_chunk, samplerate, **filter_kwargs)
            chunk_len = lfp_chunk.shape[0]
            dset[n_samples:n_samples+chunk_len,:] = lfp_chunk
            n_samples += chunk_len
        hdf.close()

        # Append the lfp metadata to the file
        lfp_metadata = metadata
        lfp_metadata['lfp_samplerate'] = lfp_samplerate
        lfp_metadata['low_cut'] = 500
        lfp_metadata['buttord'] = 4
        lfp_metadata.update(filter_kwargs)
        save_hdf(result_dir, result_filename, lfp_metadata, "/lfp_metadata", append=True)