import numpy as np
from open_ephys.analysis import Session
import glob
import os
import xml.etree.ElementTree as ETree

def load_neuropixel_configuration(data_dir, data_folder, ex_idx = 0, port_number = 1):
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
            |** rawdata (object):** data object
            |** metadata (dict):** metadata
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
            |** on_times (n_times):** times at which sync line turned on
            |** off_times (n_times):** times at which sync line turned off
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