import numpy as np

def calc_presence_ratio(data, min_trial_prop=0.9, return_details=False):
    '''
    Find which units are active on a high proportion of trials.
    
    Args:
        data (ntime, ntrials, nunit):
        min_trial_prop (float): proportion of trials a unit must have a spike on 
        
    Returns:
        presence_ratio (nunit): Proportion of trials that have a spike for each unit
        present_units (nunit): Binary mask if a unit is present or not
        presence_details (ntrials, nunit): Optional if 'return_details=True' Identifies which trials each unit is active on 
    '''        
    _, ntrials, _ = data.shape
    
    present_trials = np.sum(np.max(data>0, axis=0), axis=0) # Number of trials with a spike for each unit
    
    presence_ratio = (present_trials/ntrials)
    if return_details:
        return presence_ratio, presence_ratio>=min_trial_prop, np.max(data>0, axis=0)
    else:
        return presence_ratio, presence_ratio>=min_trial_prop


def get_units_without_refractory_violations(preproc_dir, subject, te_id, date, ref_perc_thresh=0.01, min_ref_period=1):
    '''
    Load data directly from kilosort output to find units with refractory period violations
    
    Args:
        preproc_dir (str): 
        date:
        te:
        subject:
        min_ref_period (float): [ms]
        
    Returns: 
        good_unit_labels:        
    '''
    if te_id < 18000:
        filename = aopy.data.get_preprocessed_filename(subject, te_id, date, 'ap')
    else:
        filename = aopy.data.get_preprocessed_filename(subject, te_id, date, 'ap')
        filename = filename[:-4] + '_port1.hdf'
    ap_data = aopy.data.load_hdf_group(os.path.join(preproc_dir, subject), filename, 'ap')
    nunits = len(ap_data['unit'].keys())
    
    ref_violations = np.zeros(nunits)*np.nan
    good_unit_labels = []
    for iunit, unit_lbl in enumerate(list(ap_data['unit'].keys())):
        nspikes = len(ap_data['unit'][unit_lbl])
        ref_violations[iunit] = np.sum(np.diff(ap_data['unit'][unit_lbl]) < (min_ref_period/1000)) # convert from [ms] to [s]
        
        if (ref_violations[iunit]/nspikes) <= ref_perc_thresh:
            good_unit_labels.append(unit_lbl)
    
    return good_unit_labels, ref_violations


def get_high_amplitude_units(preproc_dir, subject, te_id, date, amp_thresh=50):
    '''
    Calculates peak to peak amplitude for all channels then uses the largest one
    '''
    # Load data
    if te_id < 18000:
        filename = aopy.data.get_preprocessed_filename(subject, te_id, date, 'ap')
    else:
        filename = aopy.data.get_preprocessed_filename(subject, te_id, date, 'ap')
        filename = filename[:-4] + '_port1.hdf'
    ap_data = aopy.data.load_hdf_group(os.path.join(preproc_dir, subject), filename, 'ap')
    ap_metadata = aopy.data.load_hdf_group(os.path.join(preproc_dir, subject), filename, 'metadata') # TODO different preproc dir?

    # Initialize variables
    nunits = len(ap_data['unit'].keys())
    nwf_time = ap_data['waveform'][str(ap_data['unique_label'][0])].shape[1]
    ref_violations = np.zeros(nunits)*np.nan
    good_unit_labels = []
    mean_wfs = []
    for iunit, unit_lbl in enumerate(list(ap_data['unit'].keys())):
        cent_wfs = ap_data['waveform'][str(unit_lbl)] - np.mean(ap_data['waveform'][str(unit_lbl)], axis=1)[:,None,:] # Center each spike on each channel
        mean_wf = np.mean(cent_wfs, axis=0)*ap_metadata['bit_volts'][0] # Mean across all spikes for each channel. becomes (ntime, nch) array
        
        p2p = np.abs(np.max(mean_wf, axis=0) - np.min(mean_wf, axis=0)) # Peak to peak amplitude for each channel
        
        if np.max(p2p) > amp_thresh:
            good_unit_labels.append(unit_lbl)
            mean_wfs.append(mean_wf[:,np.argmax(p2p)])
            
            
    return good_unit_labels, mean_wfs


    def extract_ks_template_amplitudes(kilosort_dir, subject, te, date_ks, rec_site, max_time=None):
    '''
    Estimate if the amplitude distribution of a unit is cutoff by the noise threshold in the kilosort deconvolution
    
    Args:
        kilosort_dir (str):
        subject (str):
        te (list):
        date_ks (date):
        rec_site (int):
        max_time (float): [s]
    '''
    from datetime import date
    ks_folder_name_cutoff = date(2023, 8, 15)
    ks_folder_name_cutoff2 =  date(2024, 8, 15)
    samplerate = 30000 # Neuropixel AP data acq samplerate. Shouldn't change    
    if date_ks < ks_folder_name_cutoff2:
        if date_ks > ks_folder_name_cutoff:
            merged_ks_path = os.path.join(kilosort_dir, f"{date_ks}_Neuropixel_ks_{subject}_site{rec_site}_bottom_port1")
            entry_ks_path = [os.path.join(kilosort_dir, f"{date_ks}_Neuropixel_ks_{subject}_site{rec_site}_bottom_port1_{ite}") for ite in te]
        else:
            merged_ks_path = os.path.join(kilosort_dir, f"{date_ks}_Neuropixel_ks_{subject}_bottom_port1")
            entry_ks_path = [os.path.join(kilosort_dir, f"{date_ks}_Neuropixel_ks_{subject}_bottom_port1_{ite}") for ite in te]

        # Load combined trains to go into and grab what we need
        spike_times = np.load(os.path.join(merged_ks_path, 'kilosort_output/spike_times.npy'))
        template_features = np.load(os.path.join(merged_ks_path, 'kilosort_output/template_features.npy'))
        paths = np.load(os.path.join(merged_ks_path, 'task_path.npy'))
        combined_tes = [int(path[-len(str(te[0])):]) for path in paths]
        datasize = np.load(os.path.join(merged_ks_path, 'datasize_entry.npy'))        
    else:
        merged_ks_path = os.path.join(kilosort_dir, f"{date_ks}_Neuropixel_{subject}_te{te[0]}/port1/kilosort4/")
        spike_times = np.load(os.path.join(merged_ks_path, 'spike_times.npy'))
        spike_labels = np.load(os.path.join(merged_ks_path, 'spike_clusters.npy'))
        tf = np.load(os.path.join(merged_ks_path, 'tF.npy'))
        kept_spikes = np.load(os.path.join(merged_ks_path, 'kept_spikes.npy'))
        template_features = tf[kept_spikes,0,:]
        combined_tes = [te[0]]
        datasize = np.nan
    
    template_amps = {}
    for ii, ite in enumerate(te):
        if date_ks < ks_folder_name_cutoff2:
            spike_times_entry = np.load(os.path.join(entry_ks_path[ii], 'spike_indices_entry.npy'))
            spike_labels = np.load(os.path.join(entry_ks_path[ii], 'spike_clusters_entry.npy'))
        else:
            spike_times_entry = spike_times
            spike_labels = spike_labels
        
        # Find offset based on which TE is being looked at
        ientry_combined = np.where(np.isin(combined_tes, ite))[0][0]
        
        if ientry_combined > 0:
            offset = np.sum(datasize[:ientry_combined])
        else:
            offset = 0
        
        start_idx = np.where(spike_times == int(offset+np.min(spike_times_entry)))[0][0] # First spike of the combined spike train for this TE
        end_idx = np.where(spike_times == int(offset+np.max(spike_times_entry)))[0][-1] # Last spike of the combined spike train for this TE
        template_features_te = template_features[start_idx:end_idx+1,:]

        # Separate into clusters
        cluster_labels = np.unique(spike_labels)
        for icluster in cluster_labels:
            
            icluster_mask = np.isin(spike_labels, icluster)
            # print(spike_times_entry.shape, icluster_mask.shape, template_features_te.shape)
            spike_times_cluster = spike_times_entry[icluster_mask]
            template_amp_temp = np.max(template_features_te[icluster_mask,:], axis=1)
            
            if max_time is not None:
                template_amp_temp = template_amp_temp[spike_times_cluster < int(max_time*samplerate)]
                
            if icluster in list(template_amps.keys()):
                template_amps[icluster] = np.concatenate((template_amps[icluster], template_amp_temp))
            else:
                template_amps[icluster] = template_amp_temp
                
    return template_amps

def apply_noise_cutoff_thresh(template_amps, bin_width=0.1, low_bin_thresh=0.1, uhq_std_thresh=5):
    unit_labels = list(template_amps.keys())
    low_bin_perc = np.zeros(len(unit_labels))*np.nan
    cutoff_metric = np.zeros(len(unit_labels))*np.nan
    result = np.zeros(len(unit_labels), dtype=bool)
    
    for ii, unit_lbl in enumerate(unit_labels):
        if len(amps[unit_lbl]) > 10:
            bins = np.arange(np.min(amps[unit_lbl]), np.max(amps[unit_lbl]), bin_width)
            hist, bin_edges = np.histogram(amps[unit_lbl], bins=bins)
            peak_value = np.max(hist)
            low_bin = hist[0]
            low_bin_perc_temp = low_bin/peak_value
            low_bin_perc[ii] = low_bin_perc_temp
            
            # Compute upper-half-quantile metrics (uhq)
            max_amp = bin_edges[np.argmax(hist)]+(bin_width/2)
            uhq_thresh = np.quantile(amps[unit_lbl][amps[unit_lbl] > max_amp], 0.75)
            uhq_amps = amps[unit_lbl][amps[unit_lbl] > uhq_thresh]
            
            if len(uhq_amps) > 0: # If there are no amplitudes in the upper half quartile, it can't be a good unit
                uhq_hist, _ = np.histogram(uhq_amps, bins=np.arange(np.min(uhq_amps), np.max(uhq_amps), 0.1))
                uhq_mean = np.mean(uhq_hist)
                uhq_std = np.std(uhq_hist)
                cutoff_metric_temp = np.abs(low_bin-uhq_mean)/uhq_std
                cutoff_metric[ii] = cutoff_metric_temp

                if low_bin_perc_temp <= low_bin_thresh and cutoff_metric_temp <= uhq_std_thresh:
                    result[ii] = True
        
    return result, low_bin_perc, cutoff_metric