# data.py
# Code for directly loading and saving data (and results)

import numpy as np
from .whitematter import ChunkedStream, Dataset
import h5py
import tables
import csv
import pandas as pd
import os
import glob
import warnings
import pickle

import torch
from torch.utils.data import Dataset, SubsetRandomSampler, RandomSampler, DataLoader
import os.path as path # may need to build a switch here for PC/POSIX
import re
import json
import pickle as pkl
from torch.utils.data import dataset, IterableDataset
import bisect

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
    contents = glob.glob(os.path.join(base_dir,'*/*'))
    relevant_contents = filter(lambda f: str(te) in f, contents)
    files = {}
    for file in relevant_contents:
        system = os.path.basename(os.path.dirname(file))
        filename = os.path.relpath(file, base_dir)
        files[system] = filename
    return files

def get_exp_filename(te):
    '''
    Returns the experiment filename for a given task entry

    Args:
        te (int): block number for the task entry
    
    Returns:
        str: filename
    '''
    return "preprocessed_te" + str(te) + ".hdf"

def load_optitrack_metadata(data_dir, filename, metadata_row=0):
    '''
    This function loads optitrack metadata from .csv file that has 1 rigid body
    exported with the following settings:

        | **Markers:** Off
        | **Unlabeled markers:** Off
        | **Quality Statistics:** Off
        | **Rigid Bodies:** On
        | **Rigid Body Markers:** Off
        | **Bones:** Off
        | **Bone Markers:** Off
        | **Header Information:** On
        | **Optitrack format Version(s):** 1.23

    Required packages: csv, pandas

    Args:
        data_dir (string): Directory to load data from
        filename (string): File name to load within the data directory

    Returns:
        dict: Dictionary of metadata for for an optitrack datafile
    '''

    # Constants for indexing into mocap data
    mocap_data_column_idx = 2 # Column index where data begins
    rigid_body_name_idx_csvrow = 3 #csv row that contains the rigid body name
    column_type_idx_csvrow = 5 #csv row that contains the column type 
    column_names_idx_csvrow = 6  #csv row that contains the column names

    # Initialize empty dict to add to
    mocap_metadata = {}

    filepath = os.path.join(data_dir, filename)
    # Create dict with .csv meta data
    with open(filepath) as csvfile:
        mocap_reader = csv.reader(csvfile, delimiter= ',')  # csv.reader object for optitrack data loading

        for idx_csvrow, row in enumerate(mocap_reader):
            # Create a dictionary of metadata for the first line
            if idx_csvrow == metadata_row:
                # For every pair of values in the list add to dict
                for idx_dictval in range(0, len(row), 2):
                    mocap_metadata[row[idx_dictval]]= row[idx_dictval + 1]

            # Get Rigid body name for each column
            elif idx_csvrow == rigid_body_name_idx_csvrow:
                mocap_metadata['Rigid Body Name'] = row[mocap_data_column_idx:][0]

            # Load whether column is rotation/position
            elif idx_csvrow == column_type_idx_csvrow:
                mocap_metadata['Data Column Motion Types'] = row[mocap_data_column_idx:]

            # Get column names for data
            elif idx_csvrow == column_names_idx_csvrow:
                mocap_metadata['Data Column Motion Names'] = row[mocap_data_column_idx:]  

    if 'Export Frame Rate' in mocap_metadata:
        mocap_metadata['samplerate'] = float(mocap_metadata['Export Frame Rate'])
    return mocap_metadata

def load_optitrack_data(data_dir, filename):
    '''
    This function loads a series of x, y, z positional data from the optitrack
    .csv file that has 1 rigid body exported with the following settings:

        | **Markers:** Off
        | **Unlabeled markers:** Off
        | **Quality Statistics:** Off
        | **Rigid Bodies:** On
        | **Rigid Body Markers:** Off
        | **Bones:** Off
        | **Bone Markers:** Off
        | **Header Information:** On
        | **Optitrack format Version(s):** 1.23

    Required packages: pandas, numpy

    Args:
        data_dir (string): Directory to load data from
        filename (string): File name to load within the data directory

    Returns:
        tuple: Tuple containing:
            | **mocap_data_pos (nt, 3):** Positional mocap data
            | **mocap_data_rot (nt, 4):** Rotational mocap data
    '''

    # Load the metadata to check the columns are going to line up
    mocap_metadata = load_optitrack_metadata(data_dir, filename)
    if not mocap_metadata:
        raise Exception('No metadata found for optitrack file')
    assert mocap_metadata['Rotation Type'] == 'Quaternion', 'Rotation type must be Quaternion'
    assert mocap_metadata['Format Version'] == '1.23', 'Only supports version 1.23'

    # Load the data columns
    column_names_idx_csvrow = 5 # Header row index
    mocap_data_rot_column_idx = range(2,6) # Column index for rotation data
    mocap_data_pos_column_idx = range(6,9) # Column indices for position data
    filepath = os.path.join(data_dir, filename)
    # Load .csv file as a pandas data frame, convert to a numpy array, and remove
    # the 'Frame' and 'Time (Seconds)' columns.
    mocap_data_rot = pd.read_csv(filepath, header=column_names_idx_csvrow).to_numpy()[:,mocap_data_rot_column_idx]
    mocap_data_pos = pd.read_csv(filepath, header=column_names_idx_csvrow).to_numpy()[:,mocap_data_pos_column_idx]

    return mocap_data_pos, mocap_data_rot

def load_optitrack_time(data_dir, filename):
    '''
    This function loads timestamps from the optitrack .csv file 

    Required packages: pandas, numpy

    Args:
        data_dir (string): Directory to load data from
        filename (string): File name to load within the data directory

    Returns:
        (nt): Array of timestamps for each captured frame
    '''

    column_names_idx_csvrow = 5 # Header row index
    timestamp_column_idx = 1 # Column index for time data
    filepath = os.path.join(data_dir, filename)
    # Load .csv file as a pandas data frame, convert to a numpy array, and only
    # return the 'Time (Seconds)' column
    timestamps = pd.read_csv(filepath, header=column_names_idx_csvrow).to_numpy()[:,timestamp_column_idx]
    return timestamps


def get_ecube_data_sources(data_dir):
    '''
    Lists the available data sources in a given data directory

    Args: 
        data_dir (str): eCube data directory

    Returns:
        str array: available sources (AnalogPanel, Headstages, etc.)
    '''
    dat = Dataset(data_dir)
    return dat.listsources()
    
def load_ecube_metadata(data_dir, data_source):
    '''
    Sums the number of channels and samples across all files in the data_dir

    Args: 
        data_dir (str): eCube data directory
        source (str): selects the source (AnalogPanel, Headstages, etc.)

    Returns:
        dict: Dictionary of metadata with fields:
            | **samplerate (float):** sampling rate of data for this source
            | **data_source (str):** copied from the function argument
            | **n_channels (int):** number of channels
            | **n_samples (int):** number of samples for one channel
    '''

    # For now just load the metadata provieded by pyECubeSig
    # TODO: Really need the channel names and voltage per bit from the xml file
    n_channels = 0
    n_samples = 0
    dat = Dataset(data_dir)
    recordings = dat.listrecordings()
    for r in recordings: # r: (data_source, n_channels, n_samples)
        if data_source in r[0]:
            n_samples += r[2]  
            n_channels = r[1]
    samplerate = dat.samplerate
    metadata = dict(
        samplerate = samplerate,
        data_source = data_source,
        n_channels = n_channels,
        n_samples = n_samples,
    )
    return metadata

def load_ecube_data(data_dir, data_source, channels=None):
    '''
    Loads data from eCube for a given directory and datasource

    Requires load_ecube_metadata(), process_channels()

    Args:
        data_dir (str): folder containing the data you want to load
        data_source (str): type of data ("Headstages", "AnalogPanel", "DigitalPanel")
        channels (int array or None): list of channel numbers (0-indexed) to load. If None, will load all channels by default

    Returns:
        (nt, nch): all the data for the given source
    '''

    # Read metadata, check inputs
    metadata = load_ecube_metadata(data_dir, data_source)
    if channels is None:
        channels = range(metadata['n_channels'])
    elif len(channels) > metadata['n_channels']:
        raise ValueError("Supplied channel numbers are invalid")

    # Datatype is currently fixed for each data_source
    if data_source == 'DigitalPanel':
        dtype = np.uint64
    else:
        dtype = np.int16

    # Fetch all the data for all the channels
    n_samples = metadata['n_samples']
    timeseries_data = np.zeros((n_samples, len(channels)), dtype=dtype)
    n_read = 0
    for chunk in _process_channels(data_dir, data_source, channels, metadata['n_samples'], dtype=dtype):
        chunk_len = chunk.shape[0]
        timeseries_data[n_read:n_read+chunk_len,:] = chunk
        n_read += chunk_len
    return timeseries_data

def load_ecube_data_chunked(data_dir, data_source, channels=None, chunksize=728):
    '''
    Loads a data file one "chunk" at a time. Useful for replaying files as if they were online data.

    Args:
        data_dir (str): folder containing the data you want to load
        data_source (str): type of data ("Headstages", "AnalogPanel", "DigitalPanel")
        channels (int array or None): list of channel numbers (0-indexed) to load. If None, will load all channels by default
        chunksize (int): how many samples to include in each chunk

    Yields:
        (chunksize, nch): one chunk of data for the given source
    '''
    # Read metadata, check inputs
    metadata = load_ecube_metadata(data_dir, data_source)
    if channels is None:
        channels = range(metadata['n_channels'])
    elif len(channels) > metadata['n_channels']:
        raise ValueError("Supplied channel numbers are invalid")

    # Datatype is currently fixed for each data_source
    if data_source == 'DigitalPanel':
        dtype = np.uint64
    else:
        dtype = np.int16
    
    # Fetch all the channels but just return the generator
    n_samples = metadata['n_samples']
    n_channels = metadata['n_channels']
    kwargs = dict(maxchunksize=chunksize*n_channels*np.dtype(dtype).itemsize)
    return _process_channels(data_dir, data_source, channels, n_samples, **kwargs)

def proc_ecube_data(data_dir, data_source, result_filepath, **dataset_kwargs):
    '''
    Loads and saves eCube data into an HDF file

    Requires load_ecube_metadata()

    Args:
        data_dir (str): folder containing the data you want to load
        data_source (str): type of data ("Headstages", "AnalogPanel", "DigitalPanel")
        result_filepath (str): path to hdf file to be written (or appended)
        dataset_kwargs (kwargs): list of key value pairs to pass to the ecube dataset

    Returns:
        None
    '''

    metadata = load_ecube_metadata(data_dir, data_source)
    n_samples = metadata['n_samples']
    n_channels = metadata['n_channels']
    if data_source == 'DigitalPanel':
        dtype = np.uint64
    else:
        dtype = np.int16

    # Create an hdf dataset
    hdf = h5py.File(result_filepath, 'a') # should append existing or write new?
    dset = hdf.create_dataset(data_source, (n_samples, n_channels), dtype=dtype)

    # Open and read the eCube data into the new hdf dataset
    n_read = 0
    for chunk in _process_channels(data_dir, data_source, range(n_channels), n_samples, **dataset_kwargs):
        chunk_len = chunk.shape[0]
        dset[n_read:n_read+chunk_len,:] = chunk
        n_read += chunk_len

    dat = Dataset(data_dir)
    dset.attrs['samplerate'] = dat.samplerate
    dset.attrs['data_source'] = data_source
    dset.attrs['channels'] = range(n_channels)

    return dset

def _process_channels(data_dir, data_source, channels, n_samples, dtype=None, debug=False, **dataset_kwargs):
    '''
    Reads data from an ecube data source by channel until the number of samples requested 
    has been loaded. If a processing function is supplied, it will be applied to 
    each batch of data. If not, the data will be appended 

    Args:
        data_dir (str): folder containing the data you want to load
        data_source (str): type of data ("Headstages", "AnalogPanel", "DigitalPanel")
        channels (int array): list of channels to process
        n_samples (int): number of samples to read. Must be geq than a single chunk
        dtype (np.dtype): format for data_out if none supplied
        data_out (nt, nch): array of data to be written to. If None, it will be created
        debug (bool): whether the data is read in debug mode
        dataset_kwargs (kwargs): list of key value pairs to pass to the ecube dataset
        
    Yields:
        (nt, nch): Chunks of the requested samples for requested channels
    '''

    dat = Dataset(data_dir, **dataset_kwargs)
    dat.selectsource(data_source)
    chunk = dat.emitchunk(startat=0, debug=debug)
    datastream = ChunkedStream(chunkemitter=chunk)

    idx_samples = 0 # keeps track of the number of samples already read/written
    while idx_samples < n_samples:
        try:
            data_chunk = next(datastream)
            data_len = np.shape(data_chunk)[1]
            if len(channels) == 1:
                yield data_chunk[channels,:].T 
            else:
                yield np.squeeze(data_chunk[channels,:]).T # this might be where you filter data
            idx_samples += data_len
        except StopIteration:
            break

def load_ecube_digital(path, data_dir):
    '''
    Just a wrapper around load_ecube_data() and load_ecube_metadata()

    Args:
        path (str): base directory where ecube data is stored
        data_dir (str): folder you want to load

    Returns:
        tuple: Tuple containing:
            | **data (nt):** digital data, arranged as 64-bit numbers representing the 64 channels
            | **metadata (dict):** metadata (see load_ecube_metadata() for details)
    '''
    data = load_ecube_data(os.path.join(path, data_dir), 'DigitalPanel')
    metadata = load_ecube_metadata(os.path.join(path, data_dir), 'AnalogPanel')
    return data, metadata

def load_ecube_analog(path, data_dir, channels=None):
    '''
    Just a wrapper around load_ecube_data() and load_ecube_metadata()

    Args:
        path (str): base directory where ecube data is stored
        data_dir (str): folder you want to load
        channels (int array, optional): which channels to load

    Returns:
        tuple: Tuple containing:
            | **data (nt, nch):** analog data for the requested channels
            | **metadata (dict):** metadata (see load_ecube_metadata() for details)
    '''
    data = load_ecube_data(os.path.join(path, data_dir), 'AnalogPanel', channels)
    metadata = load_ecube_metadata(os.path.join(path, data_dir), 'AnalogPanel')
    return data, metadata

def load_ecube_headstages(path, data_dir, channels=None):
    '''
    Just a wrapper around load_ecube_data() and load_ecube_metadata()

    Args:
        path (str): base directory where ecube data is stored
        data_dir (str): folder you want to load
        channels (int array, optional): which channels to load

    Returns:
        tuple: Tuple containing:
            | **data (nt, nch):** analog data for the requested channels
            | **metadata (dict):** metadata (see load_ecube_metadata() for details)
    '''
    data = load_ecube_data(os.path.join(path, data_dir), 'Headstages', channels)
    metadata = load_ecube_metadata(os.path.join(path, data_dir), 'Headstages')
    return data, metadata

def save_hdf(data_dir, hdf_filename, data_dict, data_group="/", compression=0, append=False, debug=False):
    '''
    Writes data_dict and params into a hdf file in the data_dir folder 

    Args: 
        data_dir (str): destination file directory
        hdf_filename (str): name of the hdf file to be saved
        data_dict (dict, optional): the data to be saved as a hdf file
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

def load_bmi3d_hdf_table(data_dir, filename, table_name):
    '''
    Loads data and metadata from a table in an hdf file generated by BMI3D

    Args:
        data_dir (str): path to the data
        filename (str): name of the file to load from
        table_name (str): name of the table you want to load

    Returns:
        tuple: Tuple containing:
            | **data (ndarray):** data from bmi3d
            | **metadata (dict):** attributes associated with the table
    '''
    filepath = os.path.join(data_dir, filename)
    with tables.open_file(filepath, 'r') as f:
        if table_name not in f.root:
            raise ValueError(f"{table_name} not found in {filename}")
        table = getattr(f.root, table_name)
        param_keys = table.attrs._f_list("user")
        metadata = {k : getattr(table.attrs, k) for k in param_keys}
        return table.read(), metadata

def load_bmi3d_root_metadata(data_dir, filename):
    '''
    Root metadata not accessible using pytables, instead use h5py

    Args:
        data_dir (str): path to the data
        filename (str): name of the file to load from

    Returns:
        dict: key-value attributes
    '''
    with h5py.File(os.path.join(data_dir, filename), 'r') as f:
        return dict(f['/'].attrs.items())

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
        dataframe = pd.read_excel(fullfile)
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
    electrode_pos = pd.read_excel(fullfile)
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
    Write data into a pickle file.
    
    Args:
        file_to_write (str): filename with '.pkl' extension
        values_to_dump (any): values to write in a pickle file
        write_dir (str): Path - where do you want to write this file

    Returns:
        None

    examples: pkl_write(meta.pkl, data, '/data_dir')
    '''
    file = os.path.join(write_dir, file_to_write)
    with open(file, 'wb') as pickle_file:
        pickle.dump(values_to_dump, pickle_file)


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
        this_dat = pickle.load(f)
    return this_dat

# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #
# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #

class DataFile():
    r''' DataFile() class - interface class for multichannel signal data stored in binary files. Allows for segment reading without full simultaneous RAM storage
    inputs:
        - data_file_path: string
        - experiment_file_path=None: 
        - mask_file_path=None:
    
    methods:
        - read(): returns data segments defined by time start and stop points. Default behavior reads entire time span while masking channels as specified in data mask file.
    '''

    def __init__(self, data_file_path, exp_file_path=None, mask_file_path=None):

        # parse file directory and components
        data_dir = path.dirname(data_file_path)
        data_basename = path.basename(data_file_path)
        rec_id, device_id, rec_type, data_ext = data_basename.split('.')

        # experiment data file: construct and load
        if not exp_file_path:
            exp_file_name = rec_id + 'experiment.json'
            exp_file_path = path.join(data_dir,exp_file_name)

        # mask file: construct and load
        if not mask_file_path:
            mask_file_name = rec_id + '.' + device_id + '.' + rec_type + '.mask.pkl'
            mask_file_path = path.join(data_dir,mask_file_name)
         
        # set recording parameters
        self.set_data_parameters(data_file_path,exp_file_path,mask_file_path)

    # this is returned when the print() command is called.
    def __repr__(self):
        path_repr_str = f'Data file object: {self.data_file_path}'
        sample_repr_str = f'\tsamples: {self.n_sample} ({self.n_sample/self.srate:0.2f}s, {self.data_mask.mean()*100:0.2f}% masked)'
        ch_repr_str = f'\tchannels: {self.n_ch} ({self.ch_idx.mean()*100:0.2f}% masked)'
        return path_repr_str + '\n' + sample_repr_str + '\n' + ch_repr_str + '\n'
                

    # read data segment. Default call (no arguments) returns the entire recording.
    def read( self, t_start=0, t_len=-1, ch_idx=None, use_mask=True, mask_value=0., mask_pad_t=5 ):

        # get offset sample/byte values
        n_offset_samples = int(round(t_start * self.srate))
        n_offset_items = n_offset_samples * self.n_ch
        n_offset_bytes = n_offset_items * self.data_type().nbytes
        if t_len == -1:
            n_read_items = t_len
            n_read_samples = int(self.n_sample)
        else:
            n_read_samples = int(t_len * self.srate)
            n_read_items = n_read_samples * self.n_ch
        
        # read data
        with open(self.data_file_path,'rb') as f:
            data = np.fromfile(f,self.data_type,count=n_read_items,offset=n_offset_bytes)
        data = np.reshape(data,(self.n_ch,n_read_samples),order=self.reshape_order)

        # remove channels
        if not ch_idx:
            ch_idx = ~self.ch_idx
        data = data[ch_idx,:] # mask values are True for bad spots

        # mask data
        sample_idx = np.arange(n_offset_samples,n_offset_samples+n_read_samples)
        data[:,self.data_mask[sample_idx]] = mask_value

        # consider: time array? May not want to incorporate until global time is added
        return data

    @staticmethod
    def get_microdrive_parameters(exp_dict,microdrive_name):
        microdrive_name_list = [md['name'] for md in exp_dict['hardware']['microdrive']]
        microdrive_idx = [md_idx for md_idx, md in enumerate(microdrive_name_list) if microdrive_name == md][0]
        microdrive_dict = exp_dict['hardware']['microdrive'][microdrive_idx]
        electrode_label_list = [e['label'] for e in exp_dict['hardware']['microdrive'][0]['electrodes']]
        n_ch = len(electrode_label_list)
        return electrode_label_list, n_ch

    @staticmethod
    def get_read_parameters(exp_dict,rec_type):
        clfp_pattern = 'clfp*'
        if rec_type == 'raw':
            srate = exp_dict['hardware']['acquisition']['samplingrate']
            data_type = np.ushort
            reshape_order = 'F'
        elif rec_type == 'lfp':
            srate = 1000
            data_type = np.float32
            reshape_order = 'F'
        elif re.match(clfp_pattern,rec_type):
            data_type = np.float32
            if rec_type == 'clfp':
                # there are a few different naming conventions, this is the default
                srate = 1000
                reshape_order = 'F'
            else:
                clfp_ds_pattern = 'clfp_ds(\d+)'
                ds_match = re.search(clfp_ds_pattern,rec_type)
                srate = int(ds_match.group(1))
                reshape_order = 'C'
        assert isinstance(srate,int), 'parsed srate value not an integer'
        return srate, data_type, reshape_order

    @staticmethod
    def get_mask_file_path(data_path,rec_type,data_file_kern):
        clfp_pattern = 'clfp*'
        if rec_type == 'raw':
            ecog_mask_file = None
        elif rec_type == 'lfp':
            ecog_mask_file = None
        elif re.match(clfp_pattern,rec_type):
            if rec_type == 'clfp':
                ecog_mask_file = path.join(data_path,data_file_kern + ".mask.pkl")
            else:
                clfp_ds_pattern = 'clfp_ds(\d+)'
                ds_match = re.search(clfp_ds_pattern,rec_type)
                clfp_ds_file_kern = ".".join(data_file_kern.split(".")[:-1] + [ds_match.group()])
                ecog_mask_file = path.join(data_path,clfp_ds_file_kern+".mask.pkl")
        return ecog_mask_file
                

    # compute data parameter values and add as object attributes
    def set_data_parameters( self, data_file_path, exp_file_path, mask_file_path):
        # parse file
        data_file = path.basename(data_file_path)
        data_file_kern = path.splitext(data_file)[0]
        rec_id, microdrive_name, rec_type = data_file_kern.split('.')
        data_path = path.dirname(data_file_path)
        
        # read experiment file
        exp_file = path.join(data_path,rec_id + ".experiment.json")
        with open(exp_file,'r') as f:
            exp_dict = json.load(f)
        
        # get microdrive parameters
        electrode_label_list, n_ch = self.get_microdrive_parameters(exp_dict,microdrive_name)
        
        # get srate
        srate, data_type, reshape_order = self.get_read_parameters(exp_dict, rec_type)
        
        # read mask
        ecog_mask_file = self.get_mask_file_path(data_path,rec_type,data_file_kern)
        with open(ecog_mask_file,"rb") as mask_f:
            mask = pkl.load(mask_f)
        # data_mask = grow_bool_array(mask["hf"] | mask["sat"], growth_size=int(srate*0.5))
        data_mask = mask["hf"] | mask["sat"]
        if 'ch' in mask.keys():
            ch_idx = mask['ch']
        else:
            ch_idx = np.arange(n_ch)

        # clean channel labels - formatting can change from recording to recording. Get Channel ID from full string.
        ch_label_pattern = r'E\d+'
        ch_label_cleaned = [re.findall(ch_label_pattern,ch_l)[0] for ch_l in electrode_label_list]
        
        # set parameters
        self.data_file_path = data_file_path
        self.exp_file_path = exp_file_path
        self.mask_file_path = mask_file_path
        self.rec_id = rec_id
        self.microdrive_name = microdrive_name
        self.rec_type = rec_type
        self.srate = srate
        self.data_type = data_type
        self.reshape_order = reshape_order
        self.data_mask = data_mask
        self.n_ch = n_ch
        self.ch_idx = ch_idx
        self.ch_labels = ch_label_cleaned

        # set sample length information
        self.n_sample = len(self.data_mask)
        self.t_total = self.n_sample/self.srate # (s)

class DatafileDataset(Dataset):

    r"""pytorch Dataset accessing Datafile interface.

    Dataset object allowing (src, trg) sampling directly from structured binary data files.
    Built to interface with aoLab datasets. Specifically constructed for the ECoG/LFP wireless platform data.

    Arguments:
        datafile (DataFile): DataFile object
        src_t (float):\ttime length (s) of source sample
        trg_t (float):\ttime length (s) of target sample
        step_t (float):\ttime length (s) between src/trg pair sample starting points
        transform (function):\tdata transformation method for adjusting sample output pairs.

    """

    def __init__( self, datafile, src_t, trg_t, step_t, in_mem=False, transform=None, device='cpu' ):
        assert (isinstance(datafile, DataFile) or path.exists(datafile)), 'first argument must be DataFile object or valid path string'
        if isinstance(datafile, str):
            datafile = DataFile(datafile)
        sample_t = src_t + trg_t
        src_len = round(src_t*datafile.srate)
        trg_len = round(trg_t*datafile.srate)
        step_len = round(step_t*datafile.srate)
        sample_len = round(sample_t*datafile.srate)
        sample_start_idx = np.arange(0,datafile.n_sample-sample_len,step_len) # all candidate starting indices
        sample_start_idx_in_mask = [np.any(datafile.data_mask[s_s_idx:s_s_idx+sample_len]) for s_s_idx in sample_start_idx] # sample window is masked
        sample_start_idx = sample_start_idx[np.logical_not(sample_start_idx_in_mask)] # remove masked starting indices
        sample_start_t = sample_start_idx/datafile.srate

        # read whole data file if in_mem
        if in_mem:
            print(f'reading data from {datafile.data_file_path}...')
            self.data = datafile.read()
        else:
            self.data = None

        self.datafile = datafile
        self.src_len = src_len
        self.trg_len = trg_len
        self.step_len = step_len
        self.sample_len = sample_len
        self.src_t = src_t
        self.trg_t = trg_t
        self.step_t = step_t
        self.sample_t = sample_t
        self.in_mem = in_mem
        self.sample_start_idx = sample_start_idx
        self.sample_start_t = sample_start_t
        self.transform = transform
        self.device = device

    def read_sample( self, idx, ch_idx):
        if self.in_mem:
            sample_idx = np.arange(self.sample_len) + self.sample_start_idx[idx]
            sample = self.data[:,sample_idx]
            if ch_idx:
                sample = sample[ch_idx,:]
        else:
            sample = self.datafile.read(t_start=self.sample_start_t[idx],t_len=self.sample_t,ch_idx=ch_idx)
        return sample

    def __len__( self ):
        return len(self.sample_start_idx)

    def __getitem__( self, idx, ch_idx=None ):
        sample = self.read_sample(idx,ch_idx)
        src = torch.tensor(sample[:,:self.src_len]).T
        trg = torch.tensor(sample[:,self.src_len:]).T
        if self.transform:
            src,trg = self.transform((src,trg))
        return src.to(self.device), trg.to(self.device)


class DatafileConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datafile datasets.

    This class is useful to assemble different existing datafile datasets and draw from the channel indices that they share.

    Arguments:
        datasets (sequence): List of datafile datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, transform=None):
        super(DatafileConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        srate_set = list(set([ds.datafile.srate for ds in self.datasets]))
        assert len(srate_set) == 1, 'all datasets must have the same sampling rate'
        # get intersection of channel labels present in each dataset in self.datasets
        file_labels = []
        for d in self.datasets:
            file_mask_idx = np.arange(d.datafile.n_ch)[~d.datafile.ch_idx] # idx of unmasked channels in this file
            file_labels.append(np.array(d.datafile.ch_labels)[file_mask_idx])
        self.ch_label = list(set(file_labels[0]).intersection(*file_labels))
        # get index list of intersection channel locations in each datafile
        ch_sample_idx_list = []
        for d in self.datasets:
            dataset_ch_sample_idx_list = []
            for ch_i_l in self.ch_label:
                dataset_ch_sample_idx_list.append(list(np.array(d.datafile.ch_labels)[~np.array(d.datafile.ch_idx)]).index(ch_i_l))
            ch_sample_idx_list.append(dataset_ch_sample_idx_list)
        self.ch_idx = ch_sample_idx_list
        self.n_ch = len(self.ch_label)
        self.srate = srate_set[0]
        self.transform = transform

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        src, trg = self.datasets[dataset_idx].__getitem__(sample_idx)
        src = src[:,self.ch_idx[dataset_idx]]
        trg = trg[:,self.ch_idx[dataset_idx]]
        if self.transform:
            src, trg = self.transform((src,trg))
        return src, trg

    def get_data_loaders( self, partition=(0.8,0.2,0.0), batch_size=1, num_workers=0, rand_part=False, rand_seed=42 ):
        r'''
            Return dataloader objects for accessing training, validation and testing 
            partitions of the DatafileConcatDataset. Dataloaders can sample sequentially or randomly.

            arguments:
                - partition (default (0.8,0.2,0.0)): tuple of partition fractional sizes (train_frac, valid_frac, test_frac).
                    Values will be normalized to sum to 1.
                - batch_size (default 1): int value defining the size of each batch draw
                - rand_part (default False): bool determining sequential or random partitioning
                - rand_seed (default 42): int setting the rng. Keeps partitions consistent
        '''
        # get partition sizes
        frac_sum = np.sum(partition)
        train_frac = partition[0]/frac_sum
        valid_frac = partition[1]/frac_sum
        test_frac = partition[2]/frac_sum
        n_train_samp = round(train_frac * len(self))
        n_valid_samp = round(valid_frac * len(self))
        n_test_samp = round(test_frac * len(self))
        # create partition index arrays
        if rand_part:
            if not isinstance(rand_seed, int):
                try: rand_seed_new = int(rand_seed)
                except:
                    raise TypeError(f'Could not cast rand_seed value {rand_seed} to int.')
                raise Warning(f'ValueWarning: rand_seed must be of type int. Casting from {type(rand_seed)} {rand_seed} to int {rand_seed_new}. This might cause issues.')
            sample_idx = np.random.RandomState(seed=rand_seed).permutation(len(self))
        else:
            sample_idx = np.arange(len(self))
        train_sample_idx = sample_idx[:n_train_samp]
        valid_sample_idx = sample_idx[n_train_samp:(n_train_samp+n_valid_samp)]
        test_sample_idx = sample_idx[(n_train_samp+n_valid_samp):]
        # create samplers
        train_sampler = SubsetRandomSampler(train_sample_idx)
        valid_sampler = SubsetRandomSampler(valid_sample_idx)
        test_sampler = SubsetRandomSampler(test_sample_idx)
        # create dataloaders
        train_loader = DataLoader(self,batch_size=batch_size,sampler=train_sampler,num_workers=num_workers)
        valid_loader = DataLoader(self,batch_size=batch_size,sampler=valid_sampler,num_workers=num_workers)
        test_loader = DataLoader(self,batch_size=batch_size,sampler=test_sampler,num_workers=num_workers)

        return train_loader, valid_loader, test_loader

    @property
    def cummulative_sizes( self ):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

def data_transform_normalize( src, trg, scale_factor=1. ):
    r'''Data transform. Normalizes src, trg pairs through z-scoring.
    '''

    sample = np.concatenate((src,trg),axis=-1)
    center = np.mean(sample,axis=-1)
    std = np.std(sample,axis=-1)
    src = scale_factor * ((src.T - center)/std).T # is there a better way to align dimensions? einsum?
    trg = scale_factor * ((trg.T - center)/std).T
    return (src, trg)


def parse_file_info(file_path):
    file_name = os.path.basename(file_path)
    data_file_noext = os.path.splitext(file_name)[0]
    data_file_parts = data_file_noext.split('.')
    if len(data_file_parts) == 3:
        rec_id, microdrive_name, rec_type = data_file_parts
    else:
        rec_id, microdrive_name, _, rec_type = data_file_parts
    data_dir = os.path.dirname(file_path)
    exp_file_name = os.path.join(data_dir,rec_id + ".experiment.json")
    mask_file_name = os.path.join(data_dir,data_file_noext + ".mask.pkl")
    return exp_file_name, mask_file_name, microdrive_name, rec_type

def load_experiment_data(exp_file_name):
    assert os.path.exists(exp_file_name), f'inferred experiment file not found at {exp_file_name}'
    with open(exp_file_name,'r') as f:
        experiment = json.load(f)
    electrode_df = DataFrame(experiment['hardware']['microdrive'][0]['electrodes'])
    electrode_df = DataFrame.join(electrode_df,DataFrame(list(electrode_df.position)))
    del electrode_df['position']
    return experiment, electrode_df

def load_mask_data(mask_file_name):
    assert os.path.exists(mask_file_name), f'inferred mask file not found at {mask_file_name}'
    with open(mask_file_name,'r') as f:
        mask = pkl.load(f)

def read_lfp(file_path,t_range=(0,-1)):

    # get local experiment, mask files
    exp_file_name, mask_file_name, microdrive_name, rec_type = parse_file_info(file_path)

    # load experiment data
    experiment, electrode_df = load_experiment_data(exp_file_name)

    # load mask data
    mask = load_mask_data(mask_file_name)

    # get parameters: srate
    dsmatch = re.search('clfp_ds(\d+)',rec_type)
    if rec_type == 'raw':
        srate = experiment['hardware']['acquisition']['samplingrate']
        data_type = np.ushort
        reshape_order = 'F'
    elif rec_type == 'lfp':
        srate = 1000
        data_type = np.float32
        reshape_order = 'F'
    elif rec_type == 'clfp':
        srate = 1000
        data_type = np.float32
        reshape_order = 'F'
    elif dsmatch:
        # downsampled data - get srate from name
        srate = int(dsmatch.group(1))
        data_type = np.float32
        reshape_order = 'C' # files created with np.tofile which forces C ordering.

    # get microdrive parameters
    microdrive_name_list = [md['name'] for md in experiment['hardware']['microdrive']]
    microdrive_idx = [md_idx for md_idx, md in enumerate(microdrive_name_list) if microdrive_name == md][0]
    microdrive_dict = experiment['hardware']['microdrive'][microdrive_idx]
    num_ch = len(microdrive_dict['electrodes'])

    # get file size information
    data_type_size = data_type().nbytes
    file_size = os.path.getsize(file_path)
    n_offset_samples = np.round(t_range[0]*srate)
    n_offset_bytes = n_offset_samples*data_type_size
    n_all = int(np.floor(file_size/num_ch/data_type_size))
    n_stop = n_all if t_range[1] == -1 else np.min((np.round(t_range[1]*srate),n_all))
    n_read = n_stop-n_offset_samples

    # read signal data
    data = read_from_file(
        file_path,
        data_type,
        num_ch,
        n_read,
        n_offset_bytes,
        reshape_order=reshape_order
    )

    # create xarray from data and channel information
    da = xr.DataArray(
        data.T,
        dime = ('sample','ch'),
        coords = {
            'ch': electrode_df.label,
            'x_pos': ('ch', electrode_df.x),
            'y_pos': ('ch', electrode_df.y),
            'row': ('ch', electrode_df.row),
            'col': ('ch', electrode_df.col),
        },
        attrs = {'srate': srate}
    )

    return da, mask

# wrapper to read and handle clfp ECOG data
def load_ecog_clfp_data(data_file_name,t_range=(0,-1),exp_file_name=None,mask_file_name=None,compute_mask=True):

    # get file path, set ancillary data file names
    exp_file_name, mask_file_name, microdrive_name, rec_type = parse_file_info(data_file_name)

    # check for experiment file, load if valid, exit if not.
    if os.path.exists(exp_file_name):
        with open(exp_file_name,'r') as f:
            experiment = json.load(f)
    else:
        raise NameError(f'Experiment file {exp_file_name} either invalid or not found. Aborting Process.')

    # get srate
    dsmatch = re.search('clfp_ds(\d+)',rec_type)
    if rec_type == 'raw':
        srate = experiment['hardware']['acquisition']['samplingrate']
        data_type = np.ushort
        reshape_order = 'F'
    elif rec_type == 'lfp':
        srate = 1000
        data_type = np.float32
        reshape_order = 'F'
    elif rec_type == 'clfp':
        srate = 1000
        data_type = np.float32
        reshape_order = 'F'
    elif dsmatch:
        # downsampled data - get srate from name
        srate = int(dsmatch.group(1))
        data_type = np.float32
        compute_mask = False
        reshape_order = 'C' # files created with np.tofile which forces C ordering. Sorry!
    else:
        raise NameError(f'File type {rec_type}.dat not recognized. Aborting read process.')

    # get microdrive parameters
    microdrive_name_list = [md['name'] for md in experiment['hardware']['microdrive']]
    microdrive_idx = [md_idx for md_idx, md in enumerate(microdrive_name_list) if microdrive_name == md][0]
    microdrive_dict = experiment['hardware']['microdrive'][microdrive_idx]
    num_ch = len(microdrive_dict['electrodes'])

    exp = {"srate":srate,"num_ch":num_ch}

    data_type_size = data_type().nbytes
    file_size = os.path.getsize(data_file_name)
    n_offset_samples = np.round(t_range[0]*srate)
    n_offset = n_offset_samples*data_type_size
    n_all = int(np.floor(file_size/num_ch/data_type_size))
    if t_range[1] == -1:
        n_stop = n_all
    else:
        n_stop = np.min((np.round(t_range[1]*srate),n_all))
    n_read = n_stop-n_offset_samples

    # load data
    print("Loading data file:")
    # n_offset value is the number of bytes to skip
    # n_read value is the number of items to read (by data type)
    data = read_from_file(data_file_name,data_type,num_ch,n_read,n_offset,
                          reshape_order=reshape_order)
    if rec_type == 'raw': # correct uint16 encoding errors
        data = np.array(data,dtype=np.float32)
        for ch_idx in range(num_ch):
            is_neg = data[ch_idx,:] > 2**15
            data[ch_idx,is_neg] = data[ch_idx,is_neg] - (2**16 - 1)

    # check for mask file, load if valid, compute if not
    if os.path.exists(mask_file_name):
        with open(mask_file_name,"rb") as mask_f:
            mask = pkl.load(mask_f)
    elif compute_mask:
        print("No mask data file found for {0}".format(data_file))
        print("Computing data masks:")
        hf_mask,_ = datafilter.high_freq_data_detection(data,srate)
        _,sat_mask_all = datafilter.saturated_data_detection(data,srate)
        sat_mask = np.any(sat_mask_all,axis=0)
        mask = {"hf":hf_mask,"sat":sat_mask}
        # save mask data to current directory
        print("Saving mask data for {0} to {1}".format(data_file,mask_file_name))
        with open(mask_file_name,"wb") as mask_f:
            pkl.dump(mask,mask_f)
    else:
        mask = []

    return data, exp, mask

# read T seconds of data from the start of the recording:
def read_from_start(data_file_path,data_type,n_ch,n_read):
    data_file = open(data_file_path,"rb")
    data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch)
    data = np.reshape(data,(n_ch,n_read),order='F')
    data_file.close()

    return data

# read some time from a given offset
def read_from_file(data_file_path,data_type,n_ch,n_read,n_offset,reshape_order='F'):
    data_file = open(data_file_path,"rb")
    if np.version.version >= "1.17": # "offset" field not added until later installations
        data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch,
                           offset=n_offset*n_ch)
    else:
        warnings.warn("'offset' feature not available in numpy <= 1.13 - reading from the top",FutureWarning)
        data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch)
    data = np.reshape(data,(n_ch,n_read),order=reshape_order)
    data_file.close()

    return data

# read variables from the "experiment.mat" files
def get_exp_var(exp_data,*args):
    out = exp_data.copy()
    for k, var_name in enumerate(args):
        if k > 1:
            out = out[None][0][None][0][var_name]

        else:
            out = out[var_name]

    return out
