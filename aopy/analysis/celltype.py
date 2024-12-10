# celltype.py
# 
# Cell-type specific computations, e.g. classification by spike width

import numpy as np
from sklearn.mixture import GaussianMixture

from .base import find_outliers, get_pca_dimensions, interpolate_extremum_poly2

'''
Cell type classification analysis
'''
def classify_cells_spike_width(waveform_data, samplerate, std_threshold=3, pca_varthresh=0.75, min_wfs=10):
    '''
    Calculates waveform width and classifies units into putative exciatory and inhibitory cell types based on pulse width.
    Units with lower spike width are considered inhibitory cells (label: 0) and higher spike width are considered excitatory cells (label: 1)
    The pulse width is defined as the time between the waveform trough to the waveform peak. (trough-to-peak time)
    Assumes all waveforms are recorded for the same number of time points. 

    This function conducts the following processing steps:

        | **1. ** For each unit, project each waveform into the top PCs. Number of PCs determined by 'pca_varthresh'
        | **2. ** For each unit, remove outlier spikes. Outlier threhsold determined by 'std_threshold'. If the number of waveforms is less than 'min_wf', no waveforms are removed.
        | **3. ** For each unit, average remaining waveforms.
        | **4. ** For each unit, calculate spike width using a local polynomial interpolation.
        | **5. ** Use a gaussian mixture model to classify all units

    Args:
        waveform_data (nunit long list of (nt x nwaveforms) arrays): Waveforms of each unit. Each element of the list is a 2D array for each unit. Each 2D array contains the timeseries of all recorded waveforms for a given unit.
        samplerate (float): sampling rate of the points in each waveform. 
        std_threshold (float): For outlier removal. The maximum number of standard deviations (in PC space) away from the mean a given waveform is allowed to be. Defaults to 3
        pca_varthresh (float): Variance threshold for determining the number of dimensions to project spiking data onto. Defaults to 0.75.
        min_wfs (int): Minimum number of waveform samples required to perform outlier detection.

    Returns:
        tuple: A tuple containing
            | **TTP (nunit):** Spike width of each unit. [us]
            | **unit_labels (nunit):** Label of each unit. 0: low spike width (inhibitory), 1: high spike width (excitatory)
            | **avg_wfs (nunit, nt):** Average waveform of accepted waveforms for each unit
            | **sss_unitid (1D):*** Unit index of spikes with a lower number of spikes than allowed by 'min_wfs'
    '''
    TTP = [] 
    sss_unitid = []

    # Get data size parameters.
    nt, _ = waveform_data[0].shape
    nunits = len(waveform_data)

    # Initialize array for average waveforms
    avg_wfs = np.zeros((nt, nunits)) 

    # Iterate through all units
    for iunit in range(nunits): 
        iwfdata = waveform_data[iunit] # shape (nt, nunit) - waveforms for each unit
        
        # Use PCA and kmeans to remove outliers if there are enough data points
        if iwfdata.shape[1] >= min_wfs:
            # Use each time point as a feature and each spike as a sample.
            _, _, iwfdata_proj = get_pca_dimensions(iwfdata.T, max_dims=None, VAF=pca_varthresh, project_data=True)
            good_wf_idx, _ = find_outliers(iwfdata_proj, std_threshold)
        else:
            good_wf_idx = np.arange(iwfdata.shape[1])
            sss_unitid.append(iunit)
            
        iwfdata_good = iwfdata[:,good_wf_idx]

        # Average good waveforms
        iwfdata_good_avg = np.mean(iwfdata_good, axis = 1)    
        avg_wfs[:,iunit] = iwfdata_good_avg

        # Calculate 1st order TTP approximation
        troughidx_1st, peakidx_1st = find_trough_peak_idx(iwfdata_good_avg)

        # Interpolate peaks with a parabolic fit
        troughidx_2nd, _, _  = interpolate_extremum_poly2(troughidx_1st, iwfdata_good_avg, extrap_peaks=False)
        peakidx_2nd, _, _ = interpolate_extremum_poly2(peakidx_1st, iwfdata_good_avg, extrap_peaks=False)

        # Calculate 2nd order TTP approximation
        TTP.append(1e6*(peakidx_2nd - troughidx_2nd)/samplerate)    
    
    gmm_proc = GaussianMixture(n_components = 2, random_state = 0).fit(np.array(TTP).reshape(-1, 1))
    unit_labels = gmm_proc.predict(np.array(TTP).reshape(-1, 1))
    
    # Ensure lowest TTP unit is inhibitory (0)
    minttpidx = np.argmin(TTP)
    if unit_labels[minttpidx] == 1:
        unit_labels = 1 - unit_labels
    
    return TTP, unit_labels, avg_wfs, sss_unitid

def find_trough_peak_idx(unit_data):
    '''
    This function calculates the trough-to-peak time at the index level (0th order) by finding the minimum value of
    the waveform, and identifying that as the trough index. To calculate the peak index, this function finds the 
    index corresponding to the first negative derivative of the waveform. If there is no next negative derivative of
    the waveform, this function returns the last index as the peak time.
    
    Args:
        unit_data (nt, nch): Array of waveforms (Can be a 1D array with dimension nt)

    Returns:
        tuple: A tuple containing
            | **troughidx (nch):** Array of indices corresponding to the trough time for each channel
            | **peakidx (nch):** Array of indices corresponding ot the peak time for each channel. 
    '''
    # Handle condition where the input data is a 1D array
    if len(unit_data.shape) == 1:
        troughidx = np.argmin(unit_data)
        
        wfdecreaseidx = np.where(np.diff(unit_data[troughidx:])<0)
        
        if np.size(wfdecreaseidx) == 0:
            peakidx = len(unit_data)-1
        else:
            peakidx = np.min(wfdecreaseidx) + troughidx

    # Handle 2D input data array  
    else:
        troughidx = np.argmin(unit_data, axis = 0)
        peakidx = np.empty(troughidx.shape)

        for trialidx in range(len(peakidx)):
            
            wfdecreaseidx = np.where(np.diff(unit_data[troughidx[trialidx]:,trialidx])<0)

            # Handle the condition where there is no negative derivative.
            if np.size(wfdecreaseidx) == 0:
                peakidx[trialidx] = len(unit_data[:,trialidx])-1
            else:
                peakidx[trialidx] = np.min(wfdecreaseidx) + troughidx[trialidx]
        
    return troughidx, peakidx


def get_unit_spiking_mean_variance(spiking_data):
    '''
    This function calculates the mean spiking count and the spiking count variance in spiking data across 
    trials for each unit. 

    Args:
        spiking_data (ntime, nunits, ntr): Input spiking data

    Returns:
        Tuple:  A tuple containing
            | **unit_mean:** The mean spike counts for each unit across the input time
            | **unit_variance:** The spike count variance for each unit across the input time
    '''

    counts = np.sum(spiking_data, axis=1) # Counts has the shape (nunits, ntr)
    unit_mean = np.mean(counts, axis=1) # Averge the counts for each unit across all trials
    unit_variance = np.var(counts, axis=1) # Calculate the count variance for each unit across all trials

    return unit_mean, unit_variance
