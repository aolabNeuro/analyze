import os
import warnings
import pickle as pkl
import re
import sys
import json
import numpy as np
import numpy.linalg as npla
import scipy.signal as sps
from pandas import DataFrame
import xarray as xr

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
    with open(mask_file_name,'r') as f:
        return pkl.load(f)

def read_lfp(file_path,t_range=(0,-1)):
    """read_lfp

    reads data from a structured binary *lfp file in the goose wireless dataset.

    Args:
        file_path (str): file path to data file
        t_range (listlike, optional): Start and stop times to read data. (0, -1) reads the entire file. Defaults to (0,-1).

    Returns:
        da (numpy.array): numpy array of multichannel recorded neural activity saved in file_path
        mask (numpy.array): numpy array of binary mask values
    """

    # get local experiment, mask files
    exp_file_name, mask_file_name, microdrive_name, rec_type = parse_file_info(file_path)

    # load experiment data
    experiment, electrode_df = load_experiment_data(exp_file_name)

    # load mask data
    mask = load_mask_data(mask_file_name)

    # get parameters: srate
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
        data (numpy.array): numpy array of multichannel ECoG data
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

# python implementation of badChannelDetection.m - see which channels are too noisy
def bad_channel_detection( data, srate, lf_c=100, sg_win_t=8, sg_over_t=4, sg_bw = 0.5 ):
    print("Running bad channel assessment:")
    (num_ch,num_samp) = np.shape(data)

    # compute low-freq PSD estimate
    [fxx,txx,Sxx] = mt_sgram(data,srate,sg_win_t,sg_over_t,sg_bw)
    low_freq_mask = fxx < lf_c
    Sxx_low = Sxx[:,low_freq_mask,:]
    Sxx_low_psd = np.mean(Sxx_low,axis=2)

    psd_var = np.var(Sxx_low_psd,axis=1)
    norm_psd_var = psd_var/npla.norm(psd_var)
    low_var_θ = np.mean(norm_psd_var)/3
    bad_ch_mask = norm_psd_var <= low_var_θ

    return bad_ch_mask


# python implementation of highFreqTimeDetection.m - looks for spectral signatures of junk data
def high_freq_data_detection( data, srate, bad_channels=None, lf_c=100):
    print("Running high frequency noise detection: lfc @ {0}".format(lf_c))
    [num_ch,num_samp] = np.shape(data)
    bad_data_mask_all_ch = np.zeros((num_ch,num_samp))
    data_t = np.arange(num_samp)/srate
    if not bad_channels:
        bad_channels = np.zeros(num_ch)

    # mt sgram parameters
    sg_win_t = 8 # (s)
    sg_over_t = sg_win_t // 2 # (s)
    sg_bw = 0.5 # (Hz)

    # estimate hf influence, channel-wise
    for ch_i in np.arange(num_ch)[np.logical_not(bad_channels)]:
        print_progress_bar(ch_i,num_ch)
        fxx,txx,Sxx = mt_sgram(data[ch_i,:],srate,sg_win_t,sg_over_t,sg_bw) # Sxx: [num_ch]x[num_freq]x[num_t]
        num_freq, = np.shape(fxx)
        num_t, = np.shape(txx)
        Sxx_mean = np.mean(Sxx,axis=1) # average across all windows, i.e. numch x num_f periodogram

        # get low-freq, high-freq data
        low_f_mask = fxx < lf_c # Hz
        high_f_mask = np.logical_not(low_f_mask)
        low_f_mean = np.mean(Sxx_mean[low_f_mask],axis=0)
        low_f_std = np.std(Sxx_mean[low_f_mask],axis=0)
        high_f_mean = np.mean(Sxx_mean[high_f_mask],axis=0)
        high_f_std = np.std(Sxx_mean[high_f_mask],axis=0)

        # set thresholds for high, low freq. data
        low_θ = low_f_mean - 3*low_f_std
        high_θ = high_f_mean + 3*high_f_std

        for t_i, t_center in enumerate(txx):
            low_f_mean_ = np.mean(Sxx[low_f_mask,t_i])
            high_f_mean_ = np.mean(Sxx[high_f_mask,t_i])
            if low_f_mean_ < low_θ or high_f_mean_ > high_θ:
                # get indeces for the given sgram window and set them to "bad:True"
                t_bad_mask = np.logical_and(data_t > t_center - sg_win_t/2, data_t < t_center + sg_win_t/2)
                bad_data_mask_all_ch[ch_i,t_bad_mask] = True

#     bad_ch_θ = 0
#     bad_data_mask = np.sum(bad_data_mask_all_ch,axis=0) > bad_ch_θ
    bad_data_mask = np.any(bad_data_mask_all_ch,axis=0)

    return bad_data_mask, bad_data_mask_all_ch


# py version of noiseByHistogram.m - get upper and lower signal value bounds from a histogram
def histogram_defined_noise_levels( data, nbin=20 ):
    # remove data in outer bins of the histogram calculation
    hist, bin_edge = np.histogram(data,bins=nbin)
    low_edge, high_edge = bin_edge[1], bin_edge[-2]
    no_edge_mask = np.all([(data > low_edge), (data < high_edge)],axis = 0)
    data_no_edge = data[no_edge_mask]
    # compute gaussian 99% CI estimate from trimmed data
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_CI_lower, data_CI_higher = data_mean - 3*data_std, data_mean + 3*data_std
    # return min/max values from whole dataset or the edge values, whichever is lower
    noise_lower = low_edge if low_edge < data_CI_lower else min(data)
    noise_upper = high_edge if high_edge > data_CI_higher else max(data)

    return (noise_lower, noise_upper)


# multitaper spectrogram estimator (handles missing data, i.e. NaN values
def mt_sgram(x,srate,win_t,over_t,bw,interp=False,mask=None,detrend=False):
    # x - input data
    # srate - sampling rate of x
    # win_t - length of window (s)
    # over_t - size of window overlap (s)
    # bw - frequency resolution, i.e. bandwidth

    n_t = np.shape(x)[-1]
    t = srate*np.arange(n_t)

    # find, interpolate nan-values (replace in the output with nan)
#     nan_idx = np.any(np.isnan(x),axis=0)
    if interp:
        x = interp_multichannel(x)

    # compute parameters
    nw = bw*win_t/2 # time-half bandwidth product
    n_taper = int(max((np.floor(nw*2-1),1)))
    win_n = int(srate*win_t)
    over_n = int(srate*over_t)
    dpss_w = sps.windows.dpss(win_n,nw,Kmax=n_taper)

    # estimate mt spectrogram
    Sxx_m = []
    for k in range(n_taper):
        fxx,txx,Sxx_ = sps.spectrogram(x,srate,window=dpss_w[k,:],noverlap=over_n,detrend=detrend)
        Sxx_m.append(Sxx_)
    Sxx = np.mean(Sxx_m,axis=0)

    # align sgram time bins with bad times, overwrite values with NaN
    if np.any(mask):
        n_bin = np.shape(txx)[0]
        txx_edge = np.append(txx - win_t/2,txx[-1]+win_t/2)
        bad_txx = np.zeros(n_bin)
        for k in range(n_bin):
            t_in_bin = np.logical_and(t>txx_edge[k],t<txx_edge[k+1])
            bad_txx[k] = np.any(np.logical_and(t_in_bin,mask))
        bad_txx = bad_txx > 0
        Sxx[...,bad_txx] = np.nan

    return fxx, txx, Sxx


# py version of saturatedTimeDetection.m - get indeces of saturated data segments
def saturated_data_detection( data, srate, bad_channels=None, adapt_tol=1e-8 ,
                              win_n=20 ):
    print("Running saturated data segment detection:")
    num_ch, num_samp = np.shape(data)
    if not bad_channels:
        bad_channels = np.zeros(num_ch)
    bad_all_ch_mask = np.zeros((num_ch,num_samp))
    data_rect = np.abs(data)
    mask = [bool(not x) for x in bad_channels]
    for ch_i in np.arange(num_ch)[mask]:
        print_progress_bar(ch_i,num_ch)
        ch_data = data_rect[ch_i,:]
        θ1 = 50 # initialize threshold value
        θ0 = 0
        h, valc = np.histogram(ch_data,int(np.max(ch_data)))
        val = (valc[1:] + valc[:-1])/2 # computes the midpoints of each bin, valc are the edges
        val = np.floor(val)
        prob_val = h/np.shape(h)[0]

        # estimate midpoint between bimodal distribution for a theshold value
        while np.abs(θ1 - θ0) > adapt_tol:
            θ0 = θ1
            sub_θ_val_mask = val <= θ1
            sup_θ_val_mask = val > θ1
            sub_θ_val_mean = np.sum(np.multiply(val[sub_θ_val_mask],prob_val[sub_θ_val_mask]))/np.sum(prob_val[sub_θ_val_mask])
            sup_θ_val_mean = np.sum(np.multiply(val[np.logical_not(sup_θ_val_mask)],prob_val[np.logical_not(sup_θ_val_mask)]))/np.sum(prob_val[sup_θ_val_mask])
            θ1 = (sub_θ_val_mean + sup_θ_val_mean)/2

        # filter signal, boxcar window
        b_filt = np.ones(win_n)/win_n
        a_filt = 1
        ch_data_filt = sps.lfilter(b_filt,a_filt,ch_data)
        ch_data_filt_sup_θ_mask = ch_data_filt > θ1

        # get histogram-derived noise limits
        n_low, n_high = histogram_defined_noise_levels(ch_data)
        ch_data_low_mask = ch_data < n_low
        ch_data_high_mask = ch_data > n_high
        ch_data_filt_low_mask = np.logical_and(ch_data_filt_sup_θ_mask,ch_data_low_mask)
        ch_data_filt_high_mask = np.logical_and(ch_data_filt_sup_θ_mask,ch_data_high_mask)
        bad_all_ch_mask[ch_i,:] = np.logical_or(ch_data_filt_low_mask,ch_data_filt_high_mask)

        # clear out straggler values
        # I will hold off on implementing this until
#         out_of_range_samp_mask = np.logical_or(ch_data < n_low, ch_data > n_high)

#         for samp_i in np.arange(samp_i)[np.logical_and(out_of_range_samp_mask,np.logical_not(bad_all_ch_mask[i,:]))]:
#             if np.abs(ch_data[samp_i]) >= θ1 and
#             if samp_i < num_samp - srate*45:

#             else:

    num_bad = np.sum(bad_all_ch_mask,axis=0)
    sat_data_mask = num_bad > num_ch/2

    return sat_data_mask, bad_all_ch_mask

# 1-d interpolation of missing values (NaN) in multichannel data (unwraps, interpolates over NaN, fills in.)
def interp_multichannel(x):
    nan_idx = np.isnan(x)
    ok_idx = ~nan_idx
    xp = ok_idx.ravel().nonzero()[0]
    fp = x[ok_idx]
    idx = nan_idx.ravel().nonzero()[0]
    x[nan_idx] = np.interp(idx,xp,fp)

    return x

# simple progressbar, not tied to the iterator
def print_progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
