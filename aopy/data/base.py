# data.py
# Code for directly loading and saving data (and results)

import h5py
import tables
import os
import glob
import re
import warnings
import pickle as pkl
import numpy as np
from pandas import read_excel
import warnings
import yaml
import pandas as pd
from importlib.resources import files, as_file

###############################################################################
# Loading preprocessed data
###############################################################################
def get_filenames_in_dir(base_dir, te):
    '''
    Gets the filenames for available systems in a given task entry. Requires that
    files are organized by system in the base directory, and named with their task
    entry somewhere in their filename or directory name.

    Args:
        base_dir (str): directory where the files will be
        te (int): block number for the task entry

    Returns:
        dict: dictionary of files indexed by system
    '''
    warnings.warn("This function is deprecated. Please use the database instead!", DeprecationWarning)
    contents = glob.glob(os.path.join(base_dir,'*/*'))
    relevant_contents = filter(lambda f: str(te) in f, contents)
    files = {}
    for file in relevant_contents:
        system = os.path.basename(os.path.dirname(file))
        filename = os.path.relpath(file, base_dir)
        files[system] = filename
    return files

def get_preprocessed_filename(subject, te_id, date, data_source):
    '''
    Generates preprocessed filenames as per our naming conventions. 
    Format: preproc_<Date>_<MonkeyName>_<TaskEntry>_<DataSource>.hdf

    Args:
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording
        data_source (str): Processed data type (exp, eye, broadband, lfp, etc.)
    
    Returns:
        str: filename
    '''  
    return f"preproc_{date}_{subject}_{te_id}_{data_source}.hdf"

def find_preproc_ids_from_day(preproc_dir, subject, date, data_source):
    '''
    Returns the task entry ids that have preprocessed files in the given directory matching
    the subject, date, and data source given.
    
    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        date (str): Date of recording
        data_source (str): Processed data type (exp, eye, broadband, lfp, etc.)
        
    Returns
        list of ids: task entry id for each matching file found in the given folder
    '''
    contents = glob.glob(os.path.join(preproc_dir,subject,f"preproc_{date}_{subject}_*_{data_source}.hdf"))
    ids = []
    for file in contents:
        try:
            filename = os.path.basename(file)
            te_id = int(re.match(f"preproc_{date}_{subject}_(\d*)_{data_source}.hdf$", filename).group(1))
        except AttributeError:
            return []
        ids.append(te_id)
    return ids

def load_preproc_exp_data(preproc_dir, subject, te_id, date):
    '''
    Loads experiment data from a preprocessed file.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording

    Returns:
        dict: Dictionary of exp data
        dict: Dictionary of exp metadata
    '''
    filename = get_preprocessed_filename(subject, te_id, date, 'exp')
    preproc_dir = os.path.join(preproc_dir, subject)
    data = load_hdf_group(preproc_dir, filename, 'exp_data')
    metadata = load_hdf_group(preproc_dir, filename, 'exp_metadata')
    return data, metadata

def load_preproc_eye_data(preproc_dir, subject, te_id, date):
    '''
    Loads eye data from a preprocessed file.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording

    Returns:
        dict: Dictionary of eye data
        dict: Dictionary of eye metadata
    '''
    filename = get_preprocessed_filename(subject, te_id, date, 'eye')
    preproc_dir = os.path.join(preproc_dir, subject)
    data = load_hdf_group(preproc_dir, filename, 'eye_data')
    metadata = load_hdf_group(preproc_dir, filename, 'eye_metadata')
    return data, metadata

def load_preproc_broadband_data(preproc_dir, subject, te_id, date):
    '''
    Loads broadband data from a preprocessed file.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording

    Returns:
        dict: broadband data
        dict: Dictionary of broadband metadata
    '''
    filename = get_preprocessed_filename(subject, te_id, date, 'broadband')
    preproc_dir = os.path.join(preproc_dir, subject)
    data = load_hdf_data(preproc_dir, filename, 'broadband_data')
    metadata = load_hdf_group(preproc_dir, filename, 'broadband_metadata')
    return data, metadata

def load_preproc_lfp_data(preproc_dir, subject, te_id, date):
    '''
    Loads LFP data from a preprocessed file.

    Args:
        preproc_dir (str): base directory where the files live
        subject (str): Subject name
        te_id (int): Block number of Task entry object 
        date (str): Date of recording

    Returns:
        dict: lfp data
        dict: Dictionary of lfp metadata
    '''
    filename = get_preprocessed_filename(subject, te_id, date, 'lfp')
    preproc_dir = os.path.join(preproc_dir, subject)
    data = load_hdf_data(preproc_dir, filename, 'lfp_data')
    metadata = load_hdf_group(preproc_dir, filename, 'lfp_metadata')
    return data, metadata

    
###############################################################################
# Loading / saving data
###############################################################################
def save_hdf(data_dir, hdf_filename, data_dict, data_group="/", compression=0, append=False, debug=False):
    '''
    Writes data_dict and params into a hdf file in the data_dir folder 

    Args: 
        data_dir (str): destination file directory
        hdf_filename (str): name of the hdf file to be saved
        data_dict (dict): the data to be saved as a hdf file
        data_group (str, optional): where to store the data in the hdf
        compression(int, optional): gzip compression level. 0 indicate no compression. Compression not added to existing datasets. (default: 0)
        append (bool, optional): append an existing hdf file or create a new hdf file

    Returns: 
        None
    '''

    full_file_name = os.path.join(data_dir, hdf_filename)
    if append:
        hdf = h5py.File(full_file_name, 'a')
    elif not os.path.exists(full_file_name):
        hdf = h5py.File(full_file_name, 'w')
    else:
        raise FileExistsError("Will not overwrite existing file!")
        
    # Find or make the appropriate group
    if not data_group in hdf:
        group = hdf.create_group(data_group)
        if debug: print("Writing new group: {}".format(data_group))
    else:
        group = hdf[data_group]
        if debug: print("Adding data to group: {}".format(data_group))

    # Write each key, unless it exists and append is False
    for key in data_dict.keys():
        if key in group:
            if debug: print("Warning: dataset " + key + " already exists in " + data_group + "!")
            del group[key]
        data = data_dict[key]
        if hasattr(data, 'dtype') and data.dtype.char == 'U':
            data = str(data)
        elif type(data) is dict:
            import json
            key = key + '_json'
            data = json.dumps(data)
        try:
            if compression > 0:
                group.create_dataset(key, data=data, compression='gzip', compression_opts=compression)
            else:
                group.create_dataset(key, data=data)
            if debug: print("Added " + key)
        except:
            if debug: print("Warning: could not add key {} with data {}".format(key, data))
    
    hdf.close()
    if debug: print("Done!")
    return

def get_hdf_dictionary(data_dir, hdf_filename, show_tree=False):
    '''
    Lists the hdf contents in a dictionary. Does not read any data! For example,
    calling get_hdf_dictionary() with show_tree will result in something like this::

        >>> dict = get_hdf_dictionary('/exampledir', 'example.hdf', show_tree=True)
        example.hdf
        └──group1
        |  └──group_data: [shape: (1000,), type: int64]
        └──test_data: [shape: (1000,), type: int64]
        >>> print(dict)
        {
            'group1': {
                'group_data': ((1000,), dtype('int64'))
            }, 
            'test_data': ((1000,), dtype('int64'))
        }

    Args:
        data_dir (str): folder where data is located
        hdf_filename (str): name of hdf file
    
    Returns:
        dict: contents of the file keyed by name as tuples containing:
            | **shape (tuple):** size of the data
            | **dtype (np.dtype):** type of the data
    '''
    full_file_name = os.path.join(data_dir, hdf_filename)
    hdf = h5py.File(full_file_name, 'r')

    def _is_dataset(hdf):
        return isinstance(hdf, h5py.Dataset)

    def _get_hdf_contents(hdf, str_prefix=""):
        
        # If we're at a dataset print it out
        if _is_dataset(hdf):
            name = os.path.split(hdf.name)[1]
            if show_tree: 
                print(f'{str_prefix}{name}: [shape: {hdf.shape}, type: {hdf.dtype}]')
            return (hdf.shape, hdf.dtype)
        
        # Otherwise recurse if we're in a group
        else:
            contents = dict()
            for name, group in hdf.items():
                if show_tree and not _is_dataset(group):
                    print(str_prefix+"└──" + name)
                contents[name] = _get_hdf_contents(group, str_prefix.replace("└──", "|  ")+"└──")
            return contents
    
    if show_tree: 
        print(hdf_filename)
    return _get_hdf_contents(hdf)

def _load_hdf_dataset(dataset, name):
    '''
    Internal function for loading hdf datasets. Decodes json and unicode data automatically.

    Args:
        dataset (hdf object): dataset to load
        name (str): name of the dataset

    Returns:
        tuple: Tuple containing:
            | **name (str):** name of the dataset (might be modified)
            | **data (object):** loaded data
    '''
    data = dataset[()]
    if '_json' in name:
        import json
        name = name.replace('_json', '')
        data = json.loads(data)
    try:
        data = data.decode('utf-8')
    except:
        pass
    return name, data

def load_hdf_data(data_dir, hdf_filename, data_name, data_group="/"):
    '''
    Simple wrapper to get the data from an hdf file as a numpy array

    Args:
        data_dir (str): folder where data is located
        hdf_filename (str): name of hdf file
        data_name (str): table to load
        data_group (str): from which group to load data
    
    Returns:
        ndarray: numpy array of data from hdf
    '''
    full_file_name = os.path.join(data_dir, hdf_filename)
    hdf = h5py.File(full_file_name, 'r')
    full_data_name = os.path.join(data_group, data_name).replace("\\", "/")
    if full_data_name not in hdf:
        raise ValueError('{} not found in file {}'.format(full_data_name, hdf_filename))
    _, data = _load_hdf_dataset(hdf[full_data_name], data_name)
    hdf.close()
    return np.array(data)

def load_hdf_group(data_dir, hdf_filename, group="/"):
    '''
    Loads any datasets from the given hdf group into a dictionary. Also will
    recursively load other groups if any exist under the given group

    Args:
        data_dir (str): folder where data is located
        hdf_filename (str): name of hdf file
        group (str): name of the group to load
    
    Returns:
        dict: all the datasets contained in the given group
    '''
    full_file_name = os.path.join(data_dir, hdf_filename)
    hdf = h5py.File(full_file_name, 'r')
    if group not in hdf:
        raise ValueError('No such group in file {}'.format(hdf_filename))

    # Recursively load groups until datasets are reached
    def _load_hdf_group(hdf):
        keys = hdf.keys()
        data = dict()
        for k in keys:
            if isinstance(hdf[k], h5py.Group):
                data[k] = _load_hdf_group(hdf[k])
            else:
                k_, v = _load_hdf_dataset(hdf[k], k)
                data[k_] = v
        return data

    data = _load_hdf_group(hdf[group])
    hdf.close()
    return data

# Set up a cache mapping filenames to pandas dataframes so we don't have to load the
# dataframe every time someone calls the lookup functions
_cached_dataframes = {}


def is_table_in_hdf(table_name:str, hdf_filename:str):
    """
    Checks if a table exists in an hdf file' first level directory(i.e. non-recursively)

    Args:
        table_name(str): table name to be checked
        hdf_filename(str): full path to the hdf file
    
    Returns: 
        Boolean
    """
    with tables.open_file(hdf_filename, mode = 'r') as f:
        return table_name in f.root


def lookup_excel_value(data_dir, excel_file, from_column, to_column, lookup_value):
    '''
    Finds a matching value for the given key in an excel file. Used for looking up
    electrode and acquisition channels for signal path files, but can also be useful
    as a lookup table for other numeric mappings.

    Args:
        data_dir (str): where the signal path file is located
        signal_path_file (str): signal path definition file
        from_column (str, optional): the name of the electrode column
        to_column (str, optional): the name of the acquisition column
        lookup_value (int): match this value in the from_column

    Returns:
        int: the corresponding value in the lookup table, or 0 if none is found
    '''
    fullfile = os.path.join(data_dir, excel_file)
    if fullfile in _cached_dataframes:
        dataframe = _cached_dataframes[fullfile]
    else:
        dataframe = read_excel(fullfile)
        _cached_dataframes[fullfile] = dataframe
    
    row = dataframe.loc[dataframe[from_column] == lookup_value]
    if len(row) > 0:
        return row[to_column].to_numpy()[0]
    else:
        return 0

def lookup_acq2elec(data_dir, signal_path_file, acq, zero_index=True):
    '''
    Looks up the electrode number for a given acquisition channel using an excel map file (from Dr. Map)

    Args:
        data_dir (str): where the signal path file is located
        signal_path_file (str): signal path definition file
        acq (int): which channel to look up
        zero_index (bool, optional): use 0-indexing for acq and elec (default True)

    Returns:
        int: matching electrode number. If no matching electrode is found, returns -1 (or 0 with zero_index=False)
    '''
    value = lookup_excel_value(data_dir, signal_path_file, 'acq', 'electrode', acq + 1*zero_index) 
    return value - 1*zero_index

def lookup_elec2acq(data_dir, signal_path_file, elec, zero_index=True):
    '''
    Looks up the acquisition channel for a given electrode number using an excel map file (from Dr. Map)

    Args:
        data_dir (str): where the signal path file is located
        signal_path_file (str): signal path definition file
        elec (int): which electrode to look up
        zero_index (bool, optional): use 0-indexing for acq and elec (default True)

    Returns:
        int: matching acquisition channel. If no matching channel is found, returns -1 (or 0 with zero_index=False)
    '''
    value = lookup_excel_value(data_dir, signal_path_file, 'electrode', 'acq', elec + 1*zero_index)
    return value - 1*zero_index

def load_electrode_pos(data_dir, pos_file):
    '''
    Reads an electrode position map file and returns the x and y positions. The file
    should have the columns 'topdown_x' and 'topdown_y'.

    Args:
        data_dir (str): where to find the file
        pos_file (str): the excel file

    Returns:
        tuple: Tuple containing:
            | **x_pos (nch):** x position of each electrode
            | **y_pos (nch):** y position of each electrode
    '''
    fullfile = os.path.join(data_dir, pos_file)
    electrode_pos = read_excel(fullfile)
    x_pos = electrode_pos['topdown_x'].to_numpy()
    y_pos = electrode_pos['topdown_y'].to_numpy()
    return x_pos, y_pos

def map_acq2elec(signalpath_table, acq_ch_subset=None):
    '''
    Create index mapping from acquisition channel to electrode number. 
    Excel files can be loaded as a pandas dataframe using pd.read_excel
    
    Args:
        signalpath_table (pd dataframe): Signal path information in a pandas dataframe. (Mapping between electrode and acquisition ch)
        acq_ch_subset (nacq): Subset of acquisition channels to call. If not called, all acquisition channels and connected electrodes will be return. If a requested acquisition channel isn't returned a warned will be displayed

    Returns:
        tuple: Tuple containing:
            | **acq_chs (nelec):** Acquisition channels that map to electrodes (e.g. 240/256 for viventi ECoG array)
            | **connected_elecs (nelec):** Electrodes used (e.g. 240/244 for viventi ECoG array)   
    '''    
    # Parse acquisition channels used and the connected electrodes
    connected_elecs_mask = np.logical_not(np.isnan(signalpath_table['acq']))
    connected_elecs = signalpath_table['electrode'][connected_elecs_mask].to_numpy()
    acq_chs = signalpath_table['acq'][connected_elecs_mask].to_numpy(dtype=int)

    if acq_ch_subset is not None:
        acq_chs_mask = np.where(np.in1d(acq_chs, acq_ch_subset))[0]
        acq_chs = acq_chs[acq_chs_mask]
        connected_elecs = connected_elecs[acq_chs_mask]
        if len(acq_chs) < len(acq_ch_subset):
            missing_acq_chs = acq_ch_subset[np.in1d(acq_ch_subset,acq_chs, invert=True)]      
            warning_str = "Requested acquisition channels " + str(missing_acq_chs) + " are not connected"
            warnings.warn(warning_str)

    return acq_chs, connected_elecs

def map_elec2acq(signalpath_table, elecs):
    '''
    This function finds the acquisition channels that correspond to the input electrode numbers given the signal path table input. 
    This function works by calling aopy.data.map_acq2elec and subsampling the output.
    If a requested electrode isn't connected to an acquisition channel a warning will be displayed alerting the user
    and the corresponding index in the output array will be a np.nan value.

    Args:
        signalpath_table (pd dataframe): Signal path information in a pandas dataframe. (Mapping between electrode and acquisition ch)
        elecs (nelec): Electrodes to find the acquisition channels for

    Returns:
        acq_chs: Acquisition channels that map to electrodes (e.g. nelec/256 for viventi ECoG array)
    '''
    acq_chs, connected_elecs = map_acq2elec(signalpath_table)
    elec_idx = np.in1d(connected_elecs, elecs) # Find elements in 'connected_elecs' that are also in 'elecs'

    # If the output acq_chs are not the same length as the input electodes, 1+ electrodes weren't connected
    if np.sum(elec_idx) < len(elecs):
        output_acq_chs = np.zeros(len(elecs))
        output_acq_chs[:] = np.nan
        missing_elecs = []

        for ielec, elecid in enumerate(elecs):
            matched_idx = np.where(connected_elecs == elecid)[0]
            if len(matched_idx) == 0:
                missing_elecs.append(elecid)
            else:
                output_acq_chs[ielec] = acq_chs[matched_idx]
        warning_str = 'Electrodes ' + str(missing_elecs) + ' are not connected.'
        print(warning_str)

        return output_acq_chs

    else:
        return acq_chs[elec_idx]


def map_acq2pos(signalpath_table, eleclayout_table, acq_ch_subset=None, xpos_name='topdown_x', ypos_name='topdown_y'):
    '''
    Create index mapping from acquisition channel to electrode position by calling aopy.data.map_acq2elec 
    Excel files can be loaded as a pandas dataframe using pd.read_excel
    
    Args:
        signalpath_table (pd dataframe): Signal path information in a pandas dataframe. (Mapping between electrode and acquisition ch)
        eleclayout_table (pd dataframe): Electrode position information in a pandas dataframe. (Mapping between electrode and position on array)
        acq_ch_subset (nacq): Subset of acquisition channels to call. If not called, all acquisition channels and connected electrodes will be return. If a requested acquisition channel isn't returned a warned will be displayed
        xpos_name (str): Column name for the electrode 'x' position. Defaults to 'topdown_x' used with the viventi ECoG array
        ypos_name (str): Column name for the electrode 'y' position. Defaults to 'topdown_y' used with the viventi ECoG array

    Returns:
        tuple: Tuple Containing:
            | **acq_ch_position (nelec, 2):** X and Y coordinates of the electrode each acquisition channel gets data from.
                                        X position is in the first column and Y position is in the second column
            | **acq_chs (nelec):** Acquisition channels that map to electrodes (e.g. 240/256 for viventi ECoG array)
            | **connected_elecs (nelec):** Electrodes used (e.g. 240/244 for viventi ECoG array)   
    '''
    # Get index mapping from acquisition channel to electrode number
    acq_chs, connected_elecs = map_acq2elec(signalpath_table, acq_ch_subset=acq_ch_subset)
    nelec = len(connected_elecs)
    
    # Map connected electrodes to their position
    acq_ch_position = np.empty((nelec, 2))

    for ielec, elecid in enumerate(connected_elecs):
        acq_ch_position[ielec,0] = eleclayout_table[xpos_name][eleclayout_table['electrode']==elecid]
        acq_ch_position[ielec,1] = eleclayout_table[ypos_name][eleclayout_table['electrode']==elecid]

    return acq_ch_position, acq_chs, connected_elecs

def map_data2elec(datain, signalpath_table, acq_ch_subset=None, zero_indexing=False):
    '''
    Map data from its acquisition channel to the electrodes recorded from. Wrapper for aopy.data.map_acq2elec
    Excel files can be loaded as a pandas dataframe using pd.read_excel

    Args:
        datain (nt, nacqch): Data recoded from an array.
        signalpath_table (pd dataframe): Signal path information in a pandas dataframe. (Mapping between electrode and acquisition ch)
        acq_ch_subset (nacq): Subset of acquisition channels to call. If not called, all acquisition channels and connected electrodes will be return. If a requested acquisition channel isn't returned a warned will be displayed
        zero_indexing (bool): Set true if acquisition channel numbers start with 0. Defaults to False. 

    Returns:
        tuple: Tuple containing:
            | **dataout (nt, nelec):** Data from the connected electrodes
            | **acq_chs (nelec):** Acquisition channels that map to electrodes (e.g. 240/256 for viventi ECoG array)
            | **connected_elecs (nelec):** Electrodes used (e.g. 240/244 for viventi ECoG array) 
    '''
    
    acq_chs, connected_elecs = map_acq2elec(signalpath_table, acq_ch_subset=acq_ch_subset)
    if zero_indexing:
        dataout = datain[:,acq_chs]
    else:
        dataout = datain[:,acq_chs-1]
    
    return dataout, acq_chs, connected_elecs

def map_data2elecandpos(datain, signalpath_table, eleclayout_table, acq_ch_subset=None, xpos_name='topdown_x', ypos_name='topdown_y', zero_indexing=False):
    '''
    Map data from its acquisition channel to the electrodes recorded from and their position. Wrapper for aopy.data.map_acq2pos
    Excel files can be loaded as a pandas dataframe using pd.read_excel

    Args:
        datain (nt, nacqch): Data recoded from an array.
        signalpath_table (pd dataframe): Signal path information in a pandas dataframe. (Mapping between electrode and acquisition ch)
        eleclayout_table (pd dataframe): Electrode position information in a pandas dataframe. (Mapping between electrode and position on array)
        acq_ch_subset (nacq): Subset of acquisition channels to call. If not called, all acquisition channels and connected electrodes will be return. If a requested acquisition channel isn't returned a warned will be displayed
        xpos_name (str): Column name for the electrode 'x' position. Defaults to 'topdown_x' used with the viventi ECoG array
        ypos_name (str): Column name for the electrode 'y' position. Defaults to 'topdown_y' used with the viventi ECoG array
        zero_indexing (bool): Set true if acquisition channel numbers start with 0. Defaults to False. 

    Returns:
        tuple: Tuple containing:
            | **dataout (nt, nelec):** Data from the connected electrodes
            | **acq_ch_position (nelec, 2):** X and Y coordinates of the electrode each acquisition channel gets data from.
                                        X position is in the first column and Y position is in the second column
            | **acq_chs (nelec):** Acquisition channels that map to electrodes (e.g. 240/256 for viventi ECoG array)
            | **connected_elecs (nelec):** Electrodes used (e.g. 240/244 for viventi ECoG array) 
    '''
    
    acq_ch_position, acq_chs, connected_elecs = map_acq2pos(signalpath_table, eleclayout_table, acq_ch_subset=acq_ch_subset, xpos_name='topdown_x', ypos_name='topdown_y')
    if zero_indexing:
        dataout = datain[:,acq_chs]
    else:
        dataout = datain[:,acq_chs-1]
    
    return dataout, acq_ch_position, acq_chs, connected_elecs

def load_chmap(drive_type='ECoG244', acq_ch_subset=None):
    '''
    Load the mapping between acquisition channel and electrode number for the viventi ECoG array.
    
    Args:
        drive_type (str): Drive type of the viventi ECoG array. Currently only supports `ECoG244`'
        acq_ch_subset (nacq): Subset of acquisition channels to call. If not called, all acquisition 
            channels and connected electrodes will be returned.

    Returns:
        tuple: Tuple Containing:
            | **acq_ch_position (nelec, 2):** X and Y coordinates of the electrode each acquisition channel gets data from.
                                        X position is in the first column and Y position is in the second column
            | **acq_chs (nelec):** Acquisition channels that map to electrodes (e.g. 240/256 for viventi ECoG array)
            | **connected_elecs (nelec):** Electrodes used (e.g. 240/244 for viventi ECoG array)   
    '''
    config_files = files('aopy').joinpath('config')
    if drive_type == 'ECoG244':
        signal_path_filepath = as_file(config_files.joinpath('210910_ecog_signal_path.xlsx'))
        elec_to_pos_filepath = as_file(config_files.joinpath('244ch_viventi_ecog_elec_to_pos.xlsx'))
    elif drive_type == 'Opto32':
        signal_path_filepath = as_file(config_files.joinpath('221021_opto_signal_path.xlsx'))
        elec_to_pos_filepath = as_file(config_files.joinpath('32ch_fiber_optic_assy_elec_to_pos.xlsx'))
    else:
        raise ValueError('Drive type not supported')
    
    with signal_path_filepath as f:
        signal_path = pd.read_excel(f)
    with elec_to_pos_filepath as f:
        layout = pd.read_excel(f)
    if acq_ch_subset is not None:
        acq_ch_subset = np.array(acq_ch_subset, dtype='int')
    return map_acq2pos(signal_path, layout, acq_ch_subset=acq_ch_subset)

def parse_str_list(strings, str_include=None, str_avoid=None):
    '''
    This function parses a list of strings to return the strings that include/avoid specific substrings
    It was designed to parse dictionary keys

    Args: 
        strings (list of strings): List of strings 
        str_include (list of strings): List of substrings that must be included in a string to keep it
        str_avoid (list of strings): List of substrings that can not be included in a string to keep it
        
    Returns:
        (list of strings): List of strings fitting the input conditions

    Example::
        >>> str_list = ['sig001i_wf', 'sig001i_wf_ts', 'sig002a_wf', 'sig002a_wf_ts', 
                        'sig002b_wf', 'sig002b_wf_ts', 'sig002i_wf', 'sig002i_wf_ts']
        >>> parsed_strings = parse_str_list(str_list, str_include=['sig002', 'wf'], str_avoid=['b_wf', 'i_wf'])
        >>> print(parsed_strings)
        ['sig002a_wf', 'sig002a_wf_ts']
    '''

    parsed_str = []
    
    for str_idx, str_val in enumerate(strings):
        counter = 0
        nconditions = 0
        if str_include is not None:
            for istr_incl, istr_incl_val in enumerate(str_include):
                nconditions += 1
                if istr_incl_val in strings[str_idx]:
                    counter += 1
        if str_avoid is not None:
            for istr_avd, istr_avd_val in enumerate(str_avoid):
                nconditions += 1
                if istr_avd_val not in strings[str_idx]:
                    counter += 1
        
        if counter == nconditions:
            parsed_str.append(strings[str_idx])
            
    return parsed_str

def load_matlab_cell_strings(data_dir, hdf_filename, object_name):
    '''
    This function extracts strings from an object within .mat file that was saved from 
    matlab in version -7.3 (-v7.3). 

    example::

        >>> testfile = 'matlab_cell_str.mat'
        >>> strings = load_matlab_cell_strings(data_dir, testfile, 'bmiSessions')
        >>> print(strings)
        ['jeev070412j', 'jeev070512g', 'jeev070612d', 'jeev070712e', 'jeev070812d']

    Args:
        data_dir (str): where the matlab file is located
        hdf_filename (str): .mat filename
        object_name (str): Name of object to load. This is typically the variable name saved from matlab
    
    Returns:
        (list of strings): List of strings in the hdf file object

    '''
    full_file_name = os.path.join(data_dir, hdf_filename)
    strings = []
    with h5py.File(full_file_name, 'r') as f:
        objects = f[object_name]
        
        if objects.shape[0] == 1:
            for iobject in objects[0]:
                string_unicode = f[iobject]
                temp_string = ''.join(chr(i) for i in string_unicode[:].flatten())
                strings.append(temp_string)
        else:
            for iobject in objects:  
                string_unicode = f[iobject[0]]
                temp_string = ''.join(chr(i) for i in string_unicode[:].flatten())
                strings.append(temp_string)
    
    return strings


def pkl_write(file_to_write, values_to_dump, write_dir):
    '''
    Write data into a pickle file. Note: H5D5 (HDF) files can not be pickled.  Refer :func:`aopy.data.save_hdf` for saving HDF data
    
    Args:
        file_to_write (str): filename with '.pkl' extension
        values_to_dump (any): values to write in a pickle file
        write_dir (str): Path - where do you want to write this file

    Returns:
        None

    examples: pkl_write('meta.pkl', data, '/data_dir')
    '''
    file = os.path.join(write_dir, file_to_write)
    with open(file, 'wb') as pickle_file:
        pkl.dump(values_to_dump, pickle_file)


def pkl_read(file_to_read, read_dir):
    '''
    Reads data stored in a pickle file.
    
    Args:
        file_to_read (str): filename with '.pkl' extension
        read_dir (str): Path to folder where the file is stored

    Returns:
        data in a format as it is stored

    '''
    file = os.path.join(read_dir, file_to_read)
    with open(file, "rb") as f:
        this_dat = pkl.load(f)
    return this_dat

# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #
# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #

def yaml_write(filename, data):
    '''
    YAML stands for Yet Another Markup Language. It can be used to save Params or configuration files.
    Args:
        filename(str): Filename including the full path
        data (dict) : Params data to be dumped into a yaml file
    Returns: None

    Example:
        >>>params = [{ 'CENTER_TARGET_ON': 16 , 'CURSOR_ENTER_CENTER_TARGET' : 80 , 'REWARD' : 48 , 'DELAY_PENALTY' : 66 }]
        >>>params_file = '/test_data/task_codes.yaml'
        >>>yaml_write(params_file, params)
    '''
    with open(filename, 'w') as file:
        documents = yaml.dump(data, file)


def yaml_read(filename):
    '''
    The FullLoader parameter handles the conversion from YAML scalar values to Python the dictionary format
    Args:
        filename(str): Filename including the full path

    Returns:
        data (dict) : Params data dumped into a yaml file

    Example:
        >>>params_file = '/test_data/task_codes.yaml'
        >>>task_codes = yaml_read(params_file, params)
    '''
    with open(filename) as file:
        task_codes = yaml.load(file, Loader=yaml.FullLoader)

    return task_codes
