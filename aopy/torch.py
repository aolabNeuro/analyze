# torch.py
# code using the pytorch library which isn't installed by default. to install use `pip install torch`

import torch

# - - -- --- ----- -------- ---PYTORCH--- -------- ----- --- -- - -
# - - -- --- ----- -------- ---DATASETS-- -------- ----- --- -- - -

class TensorDataset(torch.utils.data.Dataset):
    r"""Dataset wrapping tensors. Implements the pytorch Dataset parent class.

    Each sample will be retrieved by indexing tensors along the first dimension. These samples are collected and returned in a list of tensors in the same order as their source tensors.

    Args:
        *tensors (Tensor): pytorch tensors that have the same size of the first dimension. This first dimension indexes over individual trials. Standard convention has the 2nd dimension index over time samples while extra dimensions index over channels or spatial dimentions.
    
    Parameters:
        device (str): Memory location to place input tensors. (default: 'cpu')
        transform (torch.nn.Module): Transform class for augmenting or transforming tensor samples. (default: None)
        transform_mast ([bool]): List of boolean values indicating which input tensors are transformed when sampled. (default: None)

    Returns:
        sample ([Tensor]): List of pytorch Tensors sampled from an input index (int).
    """

    def __init__(self, *tensors, device='cpu', transform=None, transform_mask=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.device = device
        self.transform = transform
        if transform_mask:
            assert len(self.tensors) == len(transform_mask), f'transform_mask length ({len(transform_mask)}) must match number of tensors ({len(tensors)}).'
        else:
            transform_mask = [True] * len(self.tensors) # all-hot mask
        self.transform_mask = transform_mask


    def __getitem__(self, index):
        # get samples
        sample = [tensor[index] for tensor in self.tensors]
        # apply transform
        if self.transform:
            for idx, s in enumerate(sample):
                if self.transform_mask[idx]:
                    sample[idx] = self.transform(s)
        # assign device
        sample = recursive_assign_device(sample,self.device)
        return sample

    def __len__(self):
        return self.tensors[0].size(0)

def recursive_assign_device(x, device: str):
    """Recursively assign tensor elements in a nested list or tuple of arbitrary depth to a specified device memory location.

    Args:
        x ([Tensor,[Tensor],...]): List of tensors or lists of tensors. Can be arbitrarily deep.
        device (str): Memory location on the current system (ex: 'cpu', 'cuda:0')

    Returns:
        [Tensor,[Tensor],...]: Copy of input x, memory relocated to designated location.
    """
    if isinstance(x,(list,tuple)):
        x = [recursive_assign_device(_x,device) for _x in x]
    else:
        x = x.to(device)
    return x

# transforms - -
class DropChannels(object):
    '''
        Dataset transform to randomly drop channels (i.e. set all values to zero) within a sample.
        The number of dropped channels is determined by the drop ratio:
            n_drop = floor(drop_ratio*n_ch)
        Channel dimension is assumed to be the last indexed tensor dimension. This may need to be
        adjusted for multidimensional time series data, e.g. spectrograms.
    '''
    def __init__(self,drop_ratio=0.1):
        self.drop_ratio = drop_ratio

    def __call__(self,sample):
        n_ch = sample.shape[-1]
        n_ch_drop = floor(self.drop_ratio*n_ch)
        drop_ch_idx = torch.randperm(n_ch)[:n_ch_drop]
        sample[:,drop_ch_idx] = 0.
        return sample

# z-scoring for tensors in pytorch.
def tensor_zscore(x,dim=0):
    mean = x.mean(dim=dim).expand([50,-1,-1]).permute(1,0,2)
    std = x.std(dim=dim).expand([50,-1,-1]).permute(1,0,2)
    return (x - mean) / std

#-------------------------------------------------------------------
#-------------------------------------------------------------------