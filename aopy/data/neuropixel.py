import os
import glob

import numpy as np
from open_ephys.analysis import Session
import xml.etree.ElementTree as ETree
import h5py

def load_neuropixel_configuration(data_dir, data_folder, ex_idx=0, port_number=1):
    '''
    get neuropixel probe information from xml condiguration files made by OpenEphys
    channel number and electrode x pos is sorded in the order of y pos when saved by openephys
    This function also sorts x pos and y pos in the order of channel number
    
    Args:
        data_dir (str): where to find the file
        data_folder (str): the xml file that describes neuropixel probe configuration
        ex_idx (int): experiment idx. This is usually 0.
        port_number (int): port number which a probe connected to. natural number from 1 to 4.
    
    Returns:
        config (dict) : dictionary thet contains electrode configuration
    '''

    data_path = os.path.join(data_dir, data_folder)
    xmlfile_path = glob.glob(os.path.join(data_path,'**/*.xml')) # take xml file path
    tree = ETree.parse(xmlfile_path[ex_idx])
    root = tree.getroot()

    config_list = []
    for a in root.iterfind('SIGNALCHAIN'):
        for b in a.iterfind('PROCESSOR'):
            for c in b.iterfind('EDITOR'):
                for d in c.iterfind('NP_PROBE'):
                    config = d.attrib
                    for e in d.iterfind('CHANNELS'):
                        ch_number = np.array([int(a[2:]) for a in e.attrib.keys()]) # only extract number from ch##
                        ch_bank = np.array([int(a) for a in e.attrib.values()])
                        config['channel'] = np.sort(ch_number)
                        config['ch_bank'] = ch_bank[np.argsort(ch_number)] # sort in the order of ch_number

                    for e in d.iterfind('ELECTRODE_XPOS'):
                        x_pos = np.array([int(a) for a in e.attrib.values()])
                        config['xpos'] = x_pos[np.argsort(ch_number)] # sort in the order of ch_number

                    for e in d.iterfind('ELECTRODE_YPOS'):
                        y_pos = np.array([int(a) for a in e.attrib.values()])
                        config['ypos'] = y_pos[np.argsort(ch_number)] # sort in the order of ch_number
                        
                    config_list.append(config)
                    
    return config_list[port_number-1] # Only take configuration that correspond to port nubmer

def load_neuropixel_data(data_dir, data_folder, datatype, node_idx=0, ex_idx=0, port_number=1):
    '''
    Load neuropixel data object and metadata. The data obeject has 4 properties of samples, sample_numbers, timestamps, and metadata.
    See this link: https://github.com/open-ephys/open-ephys-python-tools/tree/main/src/open_ephys/analysis
    
    Args:
        data_dir (str): data directory where the data files are located
        data_folder (str): data folder where 1 experiment data is saved
        datatype (str): datatype. 'ap' or 'lfp'
        node_idx (int): record node index. This is usually 0.
        ex_idx (int): experiment index. This is usually 0.
        port_number (int): port number which a probe connected to. natural number from 1 to 4. 
    
    Returns:
        tuple: Tuple containing:
            | **rawdata (object):** data object
            | **metadata (dict):** metadata
    '''
    
    if datatype == 'ap':
        datatype_idx = 2*(port_number-1) # even numbers correspond to action potential
    elif datatype == 'lfp':
        datatype_idx = 2*port_number-1 # odd numbers correspond to lfp
    else:
        raise ValueError(f"Unknown datatype {datatype}")
    
    # Load data and metadata
    data_path = os.path.join(data_dir, data_folder)
    session = Session(data_path)
    data = session.recordnodes[node_idx].recordings[ex_idx].continuous[datatype_idx]   
    metadata = data.metadata
    
    # Add electrode configuration to metadata
    config = load_neuropixel_configuration(data_dir, data_folder, ex_idx=ex_idx, port_number=port_number)
    metadata['slot'] = config['slot']
    metadata['port'] = config['port']
    metadata['dock'] = config['dock']
    metadata['channel'] = config['channel']
    metadata['ch_bank'] = config['ch_bank']
    metadata['xpos'] = config['xpos']
    metadata['ypos'] = config['ypos']
    metadata['referenceChannel'] = config['referenceChannel']
    metadata['probe_serial_number'] = config['probe_serial_number']
    metadata['probe_part_number'] = config['probe_part_number']
    metadata['bs_serial_number'] = config['bs_serial_number']
    metadata['bs_part_number'] = config['bs_part_number']
    metadata['bsc_serial_number'] = config['bsc_serial_number']
    metadata['bsc_part_number'] = config['bsc_part_number']
    metadata['headstage_serial_number'] = config['headstage_serial_number']
    metadata['headstage_part_number'] = config['headstage_part_number']
     
    if datatype == 'ap':
        metadata['apGainValue'] = config['apGainValue']
    else:
        metadata['lfpGainValue'] = config['lfpGainValue']
    
    return data, metadata
    
def load_neuropixel_event(data_dir, data_folder, datatype, node_idx=0, ex_idx=0, port_number=1):
    '''
    Load neuropixel's event data saved by openephys, accroding to datatype
    
    Args:
        data_dir (str): data directory where the data files are located
        data_folder (str): data folder where 1 experiment data is saved
        datatype (str): datatype. 'ap' or 'lfp'
        node_idx (int): record node index. This is usually 0.
        ex_idx (int): experiment index. This is usually 0.
        port_number (int): port number which a probe connected to. natural number from 1 to 4. 
    
    Returns:
        events (ndarray) : events data
    '''
    
    if (datatype != 'ap')&(datatype != 'lfp'):
        raise ValueError(f"Unknown datatype {datatype}")
    
    # Load data object
    data_path = os.path.join(data_dir, data_folder)
    session = Session(data_path)
    recording = session.recordnodes[node_idx].recordings[ex_idx]

    # Get probe name to search relevant events
    letter = chr(ord('@')+port_number) # convert port number to alphabetical value
    probe_name = f'Probe{letter}-{datatype.upper()}'
    
    # Initialize data structure for saving events
    ty = np.dtype([('line','i2'),('sample_number','i8'),('timestamp','f8'),('processor_id','i8'),
              ('stream_idx','i8'),('stream_name','S16'),('state','i8')])
    stream_name = recording.events.stream_name.values
    n_events = np.sum(stream_name==probe_name)
    events = np.zeros(n_events,dtype=ty)
    
    # Get event data relevant with probe name
    events['line'] = recording.events.line.values[stream_name==probe_name]
    events['sample_number'] = recording.events.sample_number.values[stream_name==probe_name]
    events['timestamp'] = recording.events.timestamp.values[stream_name==probe_name]
    events['processor_id'] = recording.events.processor_id.values[stream_name==probe_name]
    events['stream_idx'] = recording.events.stream_index.values[stream_name==probe_name]
    events['stream_name'] = recording.events.stream_name.values[stream_name==probe_name]
    events['state'] = recording.events.state.values[stream_name==probe_name]
    
    # Sort events according to sample_numbers (time_index)
    sort_idx = np.argsort(events['sample_number'])
    events['line'] = events['line'][sort_idx]
    events['sample_number'] = events['sample_number'][sort_idx]
    events['timestamp'] = events['timestamp'][sort_idx]
    events['processor_id'] = events['processor_id'][sort_idx]
    events['stream_idx'] = events['stream_idx'][sort_idx]
    events['stream_name'] = events['stream_name'][sort_idx]
    events['state'] = events['state'][sort_idx]
    
    return events

def get_neuropixel_digital_input_times(data_dir, data_folder, datatype, node_idx=0, ex_idx=0, port_number=1):
    '''
    Computes the times when sync line come to the degital channel in openephys.
    Openephys recodings doesn't always begin with 0 time index.
    
    Args:
        data_dir (str): data directory where the data files are located
        data_folder (str): data folder where 1 experiment data is saved
        datatype (str): datatype. 'ap' or 'lfp'
        node_idx (int): record node index. This is usually 0.
        ex_idx (int): experiment index. This is usually 0.
        port_number (int): port number which a probe connected to. Natural number from 1 to 4. 
    
    Returns:
        tuple: Tuple containing:
            | **on_times (n_times):** times at which sync line turned on
            | **off_times (n_times):** times at which sync line turned off
    '''
        
    # Load data and metadata
    data, metadata = load_neuropixel_data(data_dir, data_folder, datatype, node_idx=node_idx, ex_idx=ex_idx, port_number=port_number)
    FS = metadata['sample_rate']
    
    # Openephys recording doesn't always begin with 0 time. This initial_timestamp is necessary
    initial_timestamp = data.sample_numbers[0]/FS
    
    # Get on_times and off_times in digital data, based on initial_timestamps
    events = load_neuropixel_event(data_dir, data_folder, datatype, node_idx=0, ex_idx=0)
    on_times = events['sample_number'][events['state']==1]/FS - initial_timestamp
    off_times = events['sample_number'][events['state']==0]/FS - initial_timestamp
    
    return on_times, off_times

def load_ks_output(kilosort_dir, concat_data_dir, flag='spike'):
    '''
    load kilosort output preprocessed by kilosort.
    If falg is 'spike', it loads spike information (spike indices, spike label)
    If flag is 'template', it loads template informaiton that was used for spike sorting in kilosort
    If flag is 'channel', it loads channel mapping information in kilosort
    If flag is 'rez', it loads rez.mat file, which contains drift map information
    
    Args:
        kilosort_dir (str): kilosort directory (ex. '/data/preprocessed/kilosort')
        concat_data_dir (str): data directory that contains concatenated data and kilosort_output (ex. '2023-06-30_Neuropixel_ks_affi_bottom_port1')
        flag (str, optional): Which data you load. 'spike','template','channel', or 'rez'
        
    Return:
        kilosort_output (dict): preprocessed data by kilosort
    '''

    # Get kilosort directory path
    data_path = os.path.join(kilosort_dir, concat_data_dir)
    
    # path for kilosort data
    if flag == 'spike':
        spike_times_path = glob.glob(os.path.join(data_path,'**/spike_times.npy'),recursive=True)[0]
        spike_clusters_path = glob.glob(os.path.join(data_path,'**/spike_clusters.npy'),recursive=True)[0]
        ks_label_path = glob.glob(os.path.join(data_path,'**/cluster_KSLabel.tsv'),recursive=True)[0]
        amplitudes_path = glob.glob(os.path.join(data_path,'**/amplitudes.npy'),recursive=True)[0]
        cluster_amplitude_path = glob.glob(os.path.join(data_path,'**/cluster_Amplitude.tsv'),recursive=True)[0]
        cluster_group_path = glob.glob(os.path.join(data_path,'**/cluster_group.tsv'),recursive=True)[0]
    elif flag == 'template':
        spike_templates_path = glob.glob(os.path.join(data_path,'**/spike_templates.npy'),recursive=True)[0]
        template_features_path = glob.glob(os.path.join(data_path,'**/template_features.npy'),recursive=True)[0]
        template_feature_ind_path = glob.glob(os.path.join(data_path,'**/template_feature_ind.npy'),recursive=True)[0]
        templates_path = glob.glob(os.path.join(data_path,'**/templates.npy'),recursive=True)[0]
    elif flag == 'channel':
        channel_map_path = glob.glob(os.path.join(data_path,'**/channel_map.npy'),recursive=True)[0]
        channel_positions_path = glob.glob(os.path.join(data_path,'**/channel_positions.npy'),recursive=True)[0]
    elif flag == 'rez':
        rez_path = glob.glob(os.path.join(data_path,'**/rez.mat'),recursive=True)[0]
    
    # load kilsort data
    kilosort_output = {}
    if flag == 'spike':
        kilosort_output['spike_indices'] = np.load(spike_times_path)
        kilosort_output['spike_clusters'] = np.load(spike_clusters_path)
        kilosort_output['ks_label'] = np.genfromtxt(ks_label_path,skip_header=1,dtype=h5py.special_dtype(vlen=str))
        kilosort_output['amplitudes'] = np.load(amplitudes_path)
        kilosort_output['cluster_amplitude'] = np.genfromtxt(cluster_amplitude_path,skip_header=1,dtype=h5py.special_dtype(vlen=str))
        kilosort_output['cluster_group'] = np.genfromtxt(cluster_group_path,skip_header=1,dtype=h5py.special_dtype(vlen=str))
    elif flag == 'template':
        kilosort_output['spike_templates'] = np.load(spike_templates_path)
        kilosort_output['template_features'] = np.load(template_features_path)
        kilosort_output['template_feature_ind'] = np.load(template_feature_ind_path)
        kilosort_output['templates'] = np.load(templates_path)
    elif flag == 'channel':
        kilosort_output['channel_map'] = np.load(channel_map_path)
        kilosort_output['channel_positions'] = np.load(channel_positions_path)
    elif flag == 'rez':
        f = h5py.File(rez_path,'r')
        kilosort_output = dict(f['rez'])
    
    return kilosort_output

def load_parsed_ksdata(kilosort_dir, data_dir):
    '''
    load kilosort data (spike indices, clusters, and label) parsed into the task entries
    This data is not still synchronized
    
    Args:
        kilosort_dir (str): kilosort directory (ex. '/data/preprocessed/kilosort')
        data_dir (str): data directory that contains parsed data (ex. '2023-06-30_Neuropixel_ks_affi_bottom_port1_9847')
        
    Returns:
        spike_indices (nspikes): spike indices detected by kilosort (not spike times)
        spike_clusters (nspikes): unit label detected  by kilsort
    '''
    
    # Path for loading spikes and clusters
    data_path = os.path.join(kilosort_dir, data_dir)
    spike_path = os.path.join(data_path,'spike_indices_entry.npy')
    cluster_path = os.path.join(data_path,'spike_clusters_entry.npy')
    label_path = os.path.join(data_path,'ks_label.npy')

    # Load spikes and clusters
    spike_indices = np.load(spike_path)
    spike_clusters = np.load(cluster_path)
    ks_label = np.load(label_path)
    
    return spike_indices, spike_clusters, ks_label

def get_channel_bank_name(ch_bank_data, ch_config_dir ='/data/channel_config_np', filename='channel_bank.npy'):
    '''
    Get the information about which channels are used for recording. This function assumes channel configuration is either of below,
    long-br, middle, long-tr, top, long-tl, long-bl, bottom.
    
    Args:
        ch_bank_data (nch): channel bank information contained in neuropixel
        ch_config_dir (str, optional): directory that contains the channel configuration file
        filename (str, optional): filename that includes all bank information.
        
    Returns:    
        chname (str): channel name (long-br, middle, long-tr, top, long-tl, long-bl, bottom)
    '''
    data_path = os.path.join(ch_config_dir, filename)

    # Load data about channel configuration
    ch_bank_info =  np.load(data_path, allow_pickle=True).item()
    ch_bank_list = list(ch_bank_info.keys())

    # Compare actual data with configuration files to see whch bank of each channel is used.
    for chname in ch_bank_list:
        if np.all(ch_bank_info[chname] == ch_bank_data):
            break
        
    return chname
