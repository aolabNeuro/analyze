import os
import warnings
import pickle as pkl
import re
import json

import numpy as np
from pandas import DataFrame
import xarray as xr

from ..preproc.quality import high_freq_data_detection, saturated_data_detection

def parse_file_info(file_path):
    """parse_file_info

    Parses file strings for goose_wireless ECoG and LFP signal data into data parameters.

    Args:
        file_path (str): path to the file's location

    Returns:
        exp_file_name (str): JSON experiment data file path
        mask_file_name (str): binary data mask file path
        microdrive_name (str): string name of the microdrive type used to collect data in file_path
        rec_type (str): recording modality reflected in this file ('ECOG', 'LFP', etc.)
    """

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
    """load_experiment_data

    Reads experiment metadata from an experiment JSON file. Returns the complete data structure as a dictionary and returns electrode data as a pandas DataFrame.

    Args:
        exp_file_name (str): JSON experiment data file path

    Returns:
        experiment (dict): dict data object containing experiment metadata. See lab documentation for more information.
        electrode_df (DataFrame): pandas DataFrame containing microdrive electrode information. Individual channels are indexed along columns, column names are electrode IDs.
    """

    assert os.path.exists(exp_file_name), f'inferred experiment file not found at {exp_file_name}'
    with open(exp_file_name,'r') as f:
        experiment = json.load(f)
    electrode_df = DataFrame(experiment['hardware']['microdrive'][0]['electrodes'])
    electrode_df = DataFrame.join(electrode_df,DataFrame(list(electrode_df.position)))
    del electrode_df['position']
    return experiment, electrode_df

def load_mask_data(mask_file_name):
    """load_mask_data

    Loads binary mask data from recording mask files. Binary True values indicate "bad" or noisy data not used in analyses.

    Args:
        mask_file_name (str): file path to binary mask file

    Returns:
        mask (numpy.array): numpy array of binary values. Length is equal to the number of time points in the respective data array.
    """

    assert os.path.exists(mask_file_name), f'inferred mask file not found at {mask_file_name}'
    with open(mask_file_name,'rb') as f:
        return pkl.load(f)

# def read_lfp(file_path,t_range=(0,-1)):
#     """read_lfp

#     reads data from a structured binary *lfp file in the goose wireless dataset.

#     Args:
#         file_path (str): file path to data file
#         t_range (listlike, optional): Start and stop times to read data. (0, -1) reads the entire file. Defaults to (0,-1).

#     Returns:
#         da (numpy.array): numpy array of multichannel recorded neural activity saved in file_path
#         mask (numpy.array): numpy array of binary mask values
#     """

#     # get local experiment, mask files
#     exp_file_name, mask_file_name, microdrive_name, rec_type = parse_file_info(file_path)

#     # load experiment data
#     experiment, electrode_df = load_experiment_data(exp_file_name)

#     # load mask data
#     mask = load_mask_data(mask_file_name)

#     # get parameters: srate
#     dsmatch = re.search(r'clfp_ds(\d+)',rec_type)
#     if rec_type == 'raw':
#         srate = experiment['hardware']['acquisition']['samplingrate']
#         data_type = np.ushort
#         reshape_order = 'F'
#     elif rec_type == 'lfp':
#         srate = 1000
#         data_type = np.float32
#         reshape_order = 'F'
#     elif rec_type == 'clfp':
#         srate = 1000
#         data_type = np.float32
#         reshape_order = 'F'
#     elif dsmatch:
#         # downsampled data - get srate from name
#         srate = int(dsmatch.group(1))
#         data_type = np.float32
#         reshape_order = 'C' # files created with np.tofile which forces C ordering.

#     # get microdrive parameters
#     microdrive_name_list = [md['name'] for md in experiment['hardware']['microdrive']]
#     microdrive_idx = [md_idx for md_idx, md in enumerate(microdrive_name_list) if microdrive_name == md][0]
#     microdrive_dict = experiment['hardware']['microdrive'][microdrive_idx]
#     num_ch = len(microdrive_dict['electrodes'])

#     # get file size information
#     data_type_size = data_type().nbytes
#     file_size = os.path.getsize(file_path)
#     n_offset_samples = np.round(t_range[0]*srate)
#     n_offset_bytes = n_offset_samples*data_type_size
#     n_all = int(np.floor(file_size/num_ch/data_type_size))
#     n_stop = n_all if t_range[1] == -1 else np.min((np.round(t_range[1]*srate),n_all))
#     n_read = n_stop-n_offset_samples

#     # read signal data
#     data = read_from_file(
#         file_path,
#         data_type,
#         num_ch,
#         n_read,
#         n_offset_bytes,
#         reshape_order=reshape_order
#     )

#     # create xarray from data and channel information
#     da = xr.DataArray(
#         data.T,
#         dime = ('sample','ch'),
#         coords = {
#             'ch': electrode_df.label,
#             'x_pos': ('ch', electrode_df.x),
#             'y_pos': ('ch', electrode_df.y),
#             'row': ('ch', electrode_df.row),
#             'col': ('ch', electrode_df.col),
#         },
#         attrs = {'srate': srate}
#     )

#     return da, mask

# wrapper to read and handle clfp ECOG data
def load_ecog_clfp_data(data_file_name,t_range=(0,-1),exp_file_name=None,mask_file_name=None,compute_mask=True):
    """load_ecog_clfp_data

    Load ECoG data file from a goose wireless dataset file.

    Args:
        data_file_name (str): file path to data file
        t_range (listlike, optional): Start and stop times to read data. (0, -1) reads the entire file. Defaults to (0,-1).
        exp_file_name (str, optional): File path to experiment data JSON file.
        mask_file_name (str, optional): File path to data quality mask file. Defaults to None.
        compute_mask (bool, optional): Compute a data quality mask array if no mask file is given or found. Defaults to True.

    Raises:
        NameError: If experiment file cannot be found, NameError is raised.
        NameError: If mask file cannot be found, NameError is raised.

    Returns:
        data (nt x nch): numpy array of multichannel ECoG data
        mask (numpy.array): binary mask indicating bad data samples
        exp (dict): dictionary of experiment data
    """

    # get file path, set ancillary data file names
    exp_file_name, mask_file_name, microdrive_name, rec_type = parse_file_info(data_file_name)

    # check for experiment file, load if valid, exit if not.
    if os.path.exists(exp_file_name):
        with open(exp_file_name,'r') as f:
            experiment = json.load(f)
    else:
        raise NameError(f'Experiment file {exp_file_name} either invalid or not found. Aborting Process.')

    # get srate
    dsmatch = re.search(r'clfp_ds(\d+)',rec_type)
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
        print(f"No mask data file found for {data_file_name}")
        print("Computing data masks:")
        hf_mask,_ = high_freq_data_detection(data,srate)
        _,sat_mask_all = saturated_data_detection(data,srate)
        sat_mask = np.any(sat_mask_all,axis=0)
        mask = {"hf":hf_mask,"sat":sat_mask}
        # save mask data to current directory
        print(f"Saving mask data for {data_file_name} to {mask_file_name}")
        with open(mask_file_name,"wb") as mask_f:
            pkl.dump(mask,mask_f)
    else:
        mask = []

    return data, exp, mask

# read T seconds of data from the start of the recording:
def read_from_start(data_file_path,data_type,n_ch,n_read):
    """read_from_start

    Read data from goose wireless data file. Reads a fixed number of samples from the start of the recording.

    Args:
        data_file_path (str): file path to data file
        data_type (numeric type): numpy numeric type reflecting the variable encoding in data_file_path
        n_ch (int): number of channels saved in data_file_path
        n_read (int): number of time points to read from data_file_path

    Returns:
        data (np.array): numpy array of neural recording data saved in data_file_path
    """
    data_file = open(data_file_path,"rb")
    data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch)
    data = np.reshape(data,(n_ch,n_read),order='F')
    data = data.T
    data_file.close()

    return data

# read some time from a given offset
def read_from_file(data_file_path,data_type,n_ch,n_read,n_offset,reshape_order='F'):
    """read_from_file

    Reads recorded neural activity from a goose_wireless file.

    Args:
        data_file_path (str): file path to data file
        data_type (numeric type): numpy numeric type reflecting the variable encoding in data_file_path
        n_ch (int): Number of channels in data_file_path
        n_read (int): Number of data samples read from data_file_path
        n_offset (int): Offset point defining where data reading starts
        reshape_order (str, optional): Data reshaping order. Defaults to 'F'

    Returns:
        data (np.array): numpy array of neural activity stored in data_file_path
    """
    data_file = open(data_file_path,"rb")
    if np.version.version >= "1.17": # "offset" field not added until later installations
        data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch,
                           offset=n_offset*n_ch)
    else:
        warnings.warn("'offset' feature not available in numpy <= 1.13 - reading from the top",FutureWarning)
        data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch)
    data = np.reshape(data,(n_ch,n_read),order=reshape_order)
    data_file.close()
    data = data.T

    return data

# read variables from the "experiment.mat" files
def get_exp_var(exp_data,*args):
    """get_exp_var

    Generate a list of variable names from a .MAT formatted experiment data

    Args:
        exp_data (dict): MAT file data dict

    Returns:
        var_names (list): list of variable names in exp_data
    """
    out = exp_data.copy()
    for k, var_name in enumerate(args):
        if k > 1:
            out = out[None][0][None][0][var_name]

        else:
            out = out[var_name]

    return out

# data filtration code

