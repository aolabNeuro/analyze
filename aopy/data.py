import torch
from torch.utils.data import Dataset, SubsetRandomSampler, RandomSampler, DataLoader
import numpy as np
import scipy as sp
import os.path as path # may need to build a switch here for PC/POSIX
import glob
import re
import json
import pickle as pkl
from torch.utils.data import dataset, IterableDataset
import bisect


class DataFile():

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
            data = np.fromfile(f,self.data_type,count=n_read_items,offset=n_offset_bytes).reshape(n_read_samples,self.n_ch).T

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
    def get_srate_and_datatype(exp_dict,rec_type):
        clfp_pattern = 'clfp*'
        if rec_type == 'raw':
            srate = exp_dict['hardware']['acquisition']['samplingrate']
            data_type = np.ushort
        elif rec_type == 'lfp':
            srate = 1000
            data_type = np.float32
        elif re.match(clfp_pattern,rec_type):
            data_type = np.float32
            if rec_type == 'clfp':
                # there are a few different naming conventions, this is the default
                srate = 1000
            else:
                clfp_ds_pattern = 'clfp_ds(\d+)'
                ds_match = re.search(clfp_ds_pattern,rec_type)
                srate = int(ds_match.group(1))
        assert isinstance(srate,int), 'parsed srate value not an integer'
        return srate, data_type

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
        srate, data_type = self.get_srate_and_datatype(exp_dict, rec_type)
        
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

    def __init__( self, datafile, src_t, trg_t, step_t, transform=None, device='cpu' ):
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

        self.datafile = datafile
        self.src_len = src_len
        self.trg_len = trg_len
        self.step_len = step_len
        self.sample_len = sample_len
        self.src_t = src_t
        self.trg_t = trg_t
        self.step_t = step_t
        self.sample_t = sample_t
        self.sample_start_idx = sample_start_idx
        self.sample_start_t = sample_start_t
        self.transform = transform
        self.device = device

    def __len__( self ):
        return len(self.sample_start_idx)

    def __getitem__( self, idx, ch_idx=None ):
        sample = self.datafile.read(t_start=self.sample_start_t[idx],t_len=self.sample_t,ch_idx=ch_idx)
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

    def get_data_loaders( self, partition=(0.8,0.2,0.0), batch_size=1, rand_part=False, rand_seed=42 ):
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
        train_loader = DataLoader(self,batch_size=batch_size,sampler=train_sampler)
        valid_loader = DataLoader(self,batch_size=batch_size,sampler=valid_sampler)
        test_loader = DataLoader(self,batch_size=batch_size,sampler=test_sampler)

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