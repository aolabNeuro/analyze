import os
import numpy as np

from .. import data as aodata

def calc_presence_ratio(data, min_trial_prop=0.9, return_details=False):
    '''
    Find which units are active on a high proportion of trials.
    
    Args:
        data (ntime, nunit, ntrials): trial aligned binned spikes
        min_trial_prop (float): proportion of trials a unit must have a spike on 
        
    Returns:
        presence_ratio (nunit): Proportion of trials that have a spike for each unit
        present_units (nunit): Binary mask if a unit is present or not
        presence_details (ntrials, nunit): Optional if 'return_details=True' Identifies which trials each unit is active on 
    '''        
    _, _, ntrials = data.shape
    
    present_trials = np.sum(np.max(data>0, axis=0), axis=1) # Number of trials with a spike for each unit
    
    presence_ratio = (present_trials/ntrials)
    if return_details:
        return presence_ratio, presence_ratio>=min_trial_prop, np.max(data>0, axis=0)
    else:
        return presence_ratio, presence_ratio>=min_trial_prop
    

def get_units_without_refractory_violations(spike_times, ref_perc_thresh=1, min_ref_period=1, start_time=0, end_time=None):
    '''
    Identify units with refractory period violations from spike times.

    This function loads spike data from the preprocessed directory, calculates the number of refractory period violations 
    for each unit, and returns the units whose percentage of violations are above a specified threshold.

    Args:
        spike_times (dict): Loaded using data.load_preproc_spike_data(). Each key in the dictionary is a unit label string and the value is the 1D array of spike times. 
        ref_perc_thresh (float, optional): Threshold for the percentage of spikes that are allowed to violate the refractory period (default is 1%).
        min_ref_period (float, optional): The minimum refractory period in milliseconds (default is 1 ms).
        start_time (float, optional): Start time (in seconds) for the time window to consider spikes (default is 0).
        end_time (float, optional): End time (in seconds) for the time window to consider spikes (default is None, meaning all spikes after `start_time` are considered).

    Returns:
        tuple: A tuple containing:
            - good_unit_labels (ngoodunit long list of str): List of unit labels (IDs) that have an acceptable number of refractory period violations.
            - ref_violations (numpy.ndarray): An array of the percentage of refractory period violations for each unit.
    '''
    unit_labels = list(spike_times.keys())     
    nunits = len(unit_labels)
    
    ref_violations = np.zeros(nunits)*np.nan
    good_unit_labels = []
    for iunit, unit_lbl in enumerate(spike_times):
        nspikes = len(spike_times[unit_lbl])
        
        if end_time is None:
            relevant_spike_times = spike_times[unit_lbl][spike_times[unit_lbl] >= start_time]
        else:
            relevant_spike_times = spike_times[unit_lbl][np.logical_and(spike_times[unit_lbl] >= start_time, spike_times[unit_lbl] <= end_time)]
        
        # Only continue analysis if there are relevant spike times:
        nrelevant_spikes = len(relevant_spike_times)
        if nrelevant_spikes > 0:
            ref_violations[iunit] = np.sum(np.diff(relevant_spike_times) < (min_ref_period/1000))/nrelevant_spikes # convert from [ms] to [s]
            
            if (ref_violations[iunit]) <= (ref_perc_thresh/100): # Only return units if the proportion of spikes within the refractory period of a previous spike is too high. Also convert ref_perc_thresh to proportion.
                good_unit_labels.append(unit_lbl)
    
    return good_unit_labels, ref_violations*100

def get_high_amplitude_units(preproc_dir, subject, te_id, date, port, amp_thresh=50, start_time=0, end_time=None, include_bad_unit_wfs=False):
    '''
    Identifies which units pass the waveform amplitude threshold.
    Calculates peak to peak amplitude for each unit across all channels the unit is recorded on, 
    then ony returns units whose amplitude passes the threshold.

    Args:
        preproc_dir (str): The directory containing the preprocessed data.
        subject (str): The subject name.
        te_id (int): The experiment task entry number.
        date (date): The date of the experiment.
        port (int): The port number identifying which probe to look at data from.
        amp_thresh (float, optional): The amplitude threshold (in microvolts) used to filter out units with low peak-to-peak amplitudes. Defaults to 50.
        start_time (float, optional): Start time (in seconds) for the time window to consider spikes (default is 0).
        end_time (float, optional): End time (in seconds) for the time window to consider spikes (default is None, meaning all spikes after `start_time` are considered).
        include_bad_unit_wfs (bool, optional): If the waveforms of bad units should be included in the output array/

    Returns:
        tuple: A tuple containing:
            - good_unit_labels (ngoodunit long list of str): List of unit labels for units with a peak-to-peak amplitude greater than the specified threshold.
            - amplitudes (ngoodunit): Computed amplitude of each unit.
            - mean_wfs (ntime, ngoodunit or nunit): The mean waveform taken from the channel with the highest peak-to-peak amplitude for each unit that passes the amplitude threshold.
    '''
    filename_mc = aodata.get_preprocessed_filename(subject, te_id, date, 'spike')
    spike_times = aodata.load_hdf_group(os.path.join(preproc_dir, subject), filename_mc, f'drive{port}/spikes')
    ap_metadata = aodata.load_hdf_group(os.path.join(preproc_dir, subject), filename_mc, f'drive{port}/metadata')
    waveforms = aodata.load_hdf_group(os.path.join(preproc_dir, subject), filename_mc, f'drive{port}/waveforms')
    unit_labels = list(waveforms.keys())
    
    # For historical purposes.
    if 'bit_volts' in list(ap_metadata.keys()):
        microvoltsperbit = ap_metadata['bit_volts']
    elif 'microvoltsperbit' in list(ap_metadata.keys()):
        microvoltsperbit = ap_metadata['microvoltsperbit']
    elif 'voltsperbit' in list(ap_metadata.keys()):
        microvoltsperbit = 1e6*ap_metadata['voltsperbit']

    # Initialize variables
    nunits = len(unit_labels)
    nwf_time = waveforms[str(unit_labels[0])].shape[1] # Number of time points in each waveform.
    good_unit_labels = []
    amplitudes = []
    mean_wfs = []
    for iunit, unit_lbl in enumerate(unit_labels):
        if end_time is None:
            relevant_spike_mask = spike_times[unit_lbl] >= start_time
        else:
            relevant_spike_mask = np.logical_and(spike_times[unit_lbl] >= start_time, spike_times[unit_lbl] <= end_time)
        
        
        relevant_wfs = waveforms[str(unit_lbl)][relevant_spike_mask,:,:]
        cent_wfs = relevant_wfs - np.mean(relevant_wfs, axis=1)[:,None,:] # Center each spike on each channel
        mean_wf = np.nanmean(cent_wfs, axis=0)*microvoltsperbit[0] # Mean across all spikes for each channel. becomes (ntime, nch) array
        
        p2p = np.abs(np.max(mean_wf, axis=0) - np.min(mean_wf, axis=0)) # Peak to peak amplitude for each channel
        
        if np.max(p2p) > amp_thresh:
            good_unit_labels.append(unit_lbl)
            amplitudes.append(np.max(p2p))
            mean_wfs.append(mean_wf[:,np.argmax(p2p)])
        elif include_bad_unit_wfs:
            mean_wfs.append(mean_wf[:,np.argmax(p2p)])
            
    return good_unit_labels, np.array(amplitudes), np.array(mean_wfs).T


def extract_ks_template_amplitudes(preproc_dir, subject, te_id, date, port, data_source='Neuropixel', start_time=0, end_time=None):
    '''
    Extract the template amplitude for each spike from each unit. The template amplitude for a single spike is the amplitude of the projection of the waveform onto PC1 of the waveform template (chosen by kilosort).
    
    Args:
        preproc_dir (str): The directory containing the preprocessed data.
        subject (str): The subject name.
        te_id (int): The experiment task entry number.
        date (date): The date of the experiment.
        port (int): The port number identifying which probe to look at data from.
        start_time (float, optional): Start time (in seconds) for the time window to consider spikes (default is 0).
        end_time (float, optional): End time (in seconds) for the time window to consider spikes (default is None, meaning all spikes after `start_time` are considered).

    Returns:
        dict: Dictionary of template amplitudes for each spike. Each key is a unit label string and each value is a 1D array of template amplitudes for each spiek..
    '''    
    # Define filepaths 
    kilosort_dir = os.path.join(preproc_dir, 'kilosort')
    ks_folder_name = os.path.join(aodata.get_kilosort_foldername(subject, te_id, date, data_source), f"port{port}/kilosort4")
    merged_ks_path = os.path.join(kilosort_dir, ks_folder_name)
    
    # Load synchronized spike times
    filename_mc = aodata.get_preprocessed_filename(subject, te_id, date, 'spike')
    spike_times = aodata.load_hdf_group(os.path.join(preproc_dir, subject), filename_mc, f'drive{port}/spikes') # Find relevant spikes based on the synchronized (preprocessed) spike times 
    
    # Load kilosort data
    cluster_labels = np.load(os.path.join(merged_ks_path, 'spike_clusters.npy')) # Unpreprocessed output from kilosort
    template_features = np.load(os.path.join(merged_ks_path, 'amplitudes.npy'))

    # Separate into clusters
    template_amps = {}
    for icluster in np.unique(cluster_labels):
        
        # Subselect features between start_time and end_time
        if end_time is None:
            relevant_spike_mask = spike_times[str(icluster)] >= start_time
        else:
            relevant_spike_mask = np.logical_and(spike_times[str(icluster)] >= start_time, spike_times[str(icluster)] <= end_time)
        
        icluster_mask = cluster_labels == icluster
        
        template_amp_temp = template_features[icluster_mask][relevant_spike_mask] # template features has the projection onto multiple PCs of the template. We are only interested in PC1.
            
        template_amps[str(icluster)] = template_amp_temp
            
    return template_amps

def apply_noise_cutoff_thresh(template_amps, bin_width=0.2, low_bin_thresh=0.1, uhq_std_thresh=5, min_spikes=10):
    '''
    From the IBL white paper. This analyzes if each unit has a Gaussian distribution of template amplitudes.
    If a unit doesn't, spikes were likely removed by the intenal Kilosort noise threshold and may bias analysis.
    For a unit to pass this metric, its histogram of template amplitudes must satisfy two criteria:
        1. The count in the lowest bin in the histogram must be less than or equal to 10% of bin with the highest count.
        2. The lowest bin must be <= 5 standard deviations (sd) away from the mean. The mean and sd are compute from upper quartile of template amplitudes.

    Laboratory, International Brain (2022). Spike sorting pipeline for the International Brain Laboratory. 
    figshare. Online resource. https://doi.org/10.6084/m9.figshare.19705522.v4

    Args:
        template_amps (dict): A dictionary of template amplitudes with each entry corresponding to a unit.
        bin_width (float): Bin width for computing the histogram of template amplitudes. 
        low_bin_threshold (float): The count in the lowest bin must be below this proportion of the count of the highest bin.
        uhq_std_thresh (float): How many standard deviations away from the mean the lowest bin must be to be acceptable. 
            the mean and sd are computed only from the upper quartile (>75%) bins
        min_spikes (int): Minimum number of spikes for a unit for this analysis to be applied. If a unit has below this number of 
            spikes it is not included in the good units that are output.

    Returns:
        tuple: A tuple containing:
            - good_unit_labels (ngoodunit long list of str): List of unit labels for units with a peak-to-peak amplitude greater than the specified threshold.
            - low_bin_perc (nunits): What percentage the lowest bin is of the maximum (criteria 1)
            - cutoff_metric (nunits): How many upper quartile sd away from the upper quartile mean the lowest bin is (criteria 2)

    '''
    unit_labels = list(template_amps.keys())
    low_bin_perc = np.zeros(len(unit_labels))*np.nan
    cutoff_metric = np.zeros(len(unit_labels))*np.nan
    result = np.zeros(len(unit_labels), dtype=bool)
    
    good_unit_labels = []
    for ii, unit_lbl in enumerate(unit_labels):
        
        if len(template_amps[unit_lbl]) > min_spikes:
            bins = np.arange(np.min(template_amps[unit_lbl]), np.max(template_amps[unit_lbl]), bin_width)
            hist, bin_edges = np.histogram(template_amps[unit_lbl], bins=bins)
            peak_value = np.max(hist)
            low_bin = hist[0]
            low_bin_perc_temp = low_bin/peak_value
            low_bin_perc[ii] = low_bin_perc_temp
            
            # Next check if the count of the lowest bin is less than 5 standard deviations away from the UHQ bins.
            # Compute upper-half-quantile metrics (uhq)
            max_amp = bin_edges[np.argmax(hist)]+(bin_width/2)
            uhq_thresh = np.quantile(template_amps[unit_lbl][template_amps[unit_lbl] > max_amp], 0.75)
            uhq_amps = template_amps[unit_lbl][template_amps[unit_lbl] > uhq_thresh]
            
            if len(uhq_amps) > 0: # If there are no amplitudes in the upper half quartile, it can't be a good unit
                uhq_hist, _ = np.histogram(uhq_amps, bins=np.arange(np.min(uhq_amps), np.max(uhq_amps), bin_width))
                uhq_mean = np.mean(uhq_hist)
                uhq_std = np.std(uhq_hist)
                cutoff_metric_temp = np.abs(low_bin-uhq_mean)/uhq_std # how many std away from the UHQ mean is the lowest bin?
                cutoff_metric[ii] = cutoff_metric_temp

                if low_bin_perc_temp <= low_bin_thresh and cutoff_metric_temp <= uhq_std_thresh:
                    good_unit_labels.append(unit_lbl)
    
    return good_unit_labels, low_bin_perc, cutoff_metric