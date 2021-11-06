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


