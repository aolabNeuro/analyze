# data.py
# code for accessing neural data collected by the aoLab
import numpy as np
from .whitematter import ChunkedStream, Dataset
import h5py
import tables
import csv
import pandas as pd
import os
import glob

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
        filename = os.path.basename(file)
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

        Markers: Off
        Unlabeled markers: Off
        Quality Statistics: Off
        Rigid Bodies: On
        Rigid Body Markers: Off
        Bones: Off
        Bone Markers: Off
        Header Information: On
        Optitrack format Version(s): 1.23

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

        Markers: Off
        Unlabeled markers: Off
        Quality Statistics: Off
        Rigid Bodies: On
        Rigid Body Markers: Off
        Bones: Off
        Bone Markers: Off
        Header Information: On
        Optitrack format Version(s): 1.23

    Required packages: pandas, numpy

    Args:
        data_dir (string): Directory to load data from
        filename (string): File name to load within the data directory

    Returns:
        tuple: tuple containing:
        
            mocap_data_pos (nt, 3): Positional mocap data
            mocap_data_rot (nt, 4): Rotational mocap data
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

def load_ecube_metadata(data_dir, data_source):
    '''
    Sums the number of channels and samples across all files in the data_dir

    Args: 
        data_dir (str): eCube data directory
        source (str): selects the source (AnalogPanel, Headstages, etc.)

    Returns:
        dict: Dictionary of metadata with fields:

            samplerate (float): sampling rate of data for this source
            data_source (str): copied from the function argument
            n_channels (int): number of channels
            n_samples (int): number of samples for one channel
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
        data_source (str): type of data ("Headstage", "AnalogPanel", "DigitalPanel")
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
        dtype = np.uint16

    # Fetch all the data for all the channels
    timeseries_data = process_channels(data_dir, data_source, channels, metadata['n_samples'], dtype=dtype)
    return timeseries_data

def proc_ecube_data(data_dir, data_source, result_filepath, **dataset_kwargs):
    '''
    Loads and saves eCube data into an HDF file

    Requires load_ecube_metadata()

    Args:
        data_dir (str): folder containing the data you want to load
        data_source (str): type of data ("Headstage", "AnalogPanel", "DigitalPanel")
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
        dtype = np.uint16

    # Create an hdf dataset
    hdf = h5py.File(result_filepath, 'a') # should append existing or write new?
    dset = hdf.create_dataset(data_source, (n_samples, n_channels), dtype=dtype)

    # Open and read the eCube data into the new hdf dataset
    process_channels(data_dir, data_source, range(n_channels), n_samples, data_out=dset, **dataset_kwargs)
    dat = Dataset(data_dir)
    dset.attrs['samplerate'] = dat.samplerate
    dset.attrs['data_source'] = data_source
    dset.attrs['channels'] = range(n_channels)

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

def process_channels(data_dir, data_source, channels, n_samples, dtype=None, data_out=None, **dataset_kwargs):
    '''
    Reads data from an ecube data source by channel until the number of samples requested 
    has been loaded. If a processing function is supplied, it will be applied to 
    each batch of data. If not, the data will be appended 

    Args:
        data_dir (str): folder containing the data you want to load
        data_source (str): type of data ("Headstage", "AnalogPanel", "DigitalPanel")
        channels (int array): list of channels to process
        n_samples (int): number of samples to read. Must be geq than a single chunk
        dtype (np.dtype): format for data_out if none supplied
        data_out (nt, nch): array of data to be written to. If None, it will be created
        dataset_kwargs (kwargs): list of key value pairs to pass to the ecube dataset
        
    Returns:
        (nt, nch): Requested samples for requested channels
    '''
    if data_out == None:
        data_out = np.zeros((n_samples, len(channels)), dtype=dtype)

    dat = Dataset(data_dir, **dataset_kwargs)
    dat.selectsource(data_source)
    chunk = dat.emitchunk(startat=0, debug=True)
    datastream = ChunkedStream(chunkemitter=chunk)

    idx_samples = 0 # keeps track of the number of samples already read/written
    while idx_samples < n_samples:
        try:
            data_chunk = next(datastream)
            data_len = np.shape(data_chunk)[1]
            data_out[idx_samples:idx_samples+data_len,:] = np.reshape(data_chunk[channels,:], (data_len, len(channels))) # this might be where you filter data
            idx_samples += data_len
        except StopIteration:
            break
    return data_out

def load_ecube_digital(path, data_dir):
    '''
    Just a wrapper around load_ecube_data() and load_ecube_metadata()

    Args:
        path (str): base directory where ecube data is stored
        data_dir (str): folder you want to load

    Returns:
        tuple: tuple containing:
        
            data (nt): digital data, arranged as 64-bit numbers representing the 64 channels
            metadata (dict): metadata (see load_ecube_metadata() for details)
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
        tuple: tuple containing:
        
            data (nt, nch): analog data for the requested channels
            metadata (dict): metadata (see load_ecube_metadata() for details)
    '''
    data = load_ecube_data(os.path.join(path, data_dir), 'AnalogPanel', channels)
    metadata = load_ecube_metadata(os.path.join(path, data_dir), 'AnalogPanel')
    return data, metadata

def save_hdf(data_dir, hdf_filename, data_dict, data_group="/", append=False, debug=False):
    '''
    Writes data_dict and params into a hdf file in the data_dir folder 

    Args: 
        data_dir (str): destination file directory
        hdf_filename (str): name of the hdf file to be saved
        data_dict (dict, optional): the data to be saved as a hdf file
        data_group (str, optional): where to store the data in the hdf
        append (bool, optional): append an existing hdf file or create a new hdf file

    Returns: None
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
            print("Warning: dataset " + key + " already exists in " + data_group + "!")
            del group[key]
        data = data_dict[key]
        if hasattr(data, 'dtype') and data.dtype.char == 'U':
            data = str(data)
        elif type(data) is dict:
            import json
            key = key + '_json'
            data = json.dumps(data)
        try:
            group.create_dataset(key, data=data)
            if debug: print("Added " + key)
        except:
            if debug: print("Warning: could not add key {} with data {}".format(key, data))
    
    hdf.close()
    if debug: print("Done!")
    return

def get_hdf_contents(data_dir, hdf_filename, show_tree=False):
    '''
    Lists the hdf contents in a dictionary

    Args:
        data_dir (str): folder where data is located
        hdf_filename (str): name of hdf file
    
    Returns:
        dict: contents of the file
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
        tuple: tuple containing:

            name (str): name of the dataset (might be modified)
            data (object): loaded data
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
    full_data_name = os.path.join(data_group, data_name)
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

def load_bmi3d_hdf_table(data_dir, filename, table_name=None):
    '''
    Loads data and metadata from a table in an hdf file generated by BMI3D

    Args:
        data_dir (str): path to the data
        filename (str): name of the file to load from
        table_name (str, optional): name of the table you want to load

    Returns:
        tuple: tuple containing:
        
            data (ndarray): data from bmi3d
            metadata (dict): attributes associated with the table
    '''
    filepath = os.path.join(data_dir, filename)
    with tables.open_file(filepath, 'r') as f:
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

_cached_signal_path = {}

def lookup_acq2elec(data_dir, signal_path_file, acq):
    '''
    Looks up the electrode number for a given acquisition channel using an excel map file (from Dr. Map)

    Inputs:
        data_dir (str): where the signal path file is located
        signal_path_file (str): signal path definition file
        acq (int): which channel to look up (0-indexed)

    Output:
        ch2elec [nchannels]: lookup table mapping channels to electrodes or -1
        pos (nelecs, nelecs): tuple of x and y positions for each electrode
    '''
    fullfile = os.path.join(data_dir, signal_path_file)
    if fullfile in _cached_signal_path:
        signal_path = _cached_signal_path[fullfile]
    else:
        signal_path = pd.read_excel(fullfile)
        _cached_signal_path[fullfile] = signal_path
    
    ch = acq + 1 # signal paths are 1-indexed
    row = signal_path.loc[signal_path['acq'] == ch]
    if len(row) > 0:
        return row['electrode'].to_numpy() - 1 # translate to 0-index
    else:
        return -1

def load_electrode_pos(data_dir, pos_file):
    '''
    Reads an electrode position map file and returns the x and y positions. The file
    should have the columns 'topdown_x' and 'topdown_y'.

    Args:
        data_dir (str): where to find the file
        pos_file (str): the excel file

    Returns:
        tuple: tuple containing:

            x_pos (nch): x position of each electrode
            y_pos (nch): y position of each electrode
    '''
    fullfile = os.path.join(data_dir, pos_file)
    electrode_pos = pd.read_excel(fullfile)
    x_pos = electrode_pos['topdown_x'].to_numpy()
    y_pos = electrode_pos['topdown_y'].to_numpy()
    return x_pos, y_pos