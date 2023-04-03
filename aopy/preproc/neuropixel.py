import numpy as np
from ..utils import extract_barcodes_from_times, get_first_last_times, sync_timestamp_offline

def sync_neuropixel_ecube(raw_timestamp,on_times_np,off_times_np,on_times_ecube,off_times_ecube,bar_duration=0.017):
    '''
    This function is specfic to synchronization between neuropixels and ecube.
    
    Args:
    raw_timestamp (nt) : raw timestamp in neuropixels that is not synchronized
    on_times_np (ndarray) : times when sync line rises to 1 in a neruopixel stream
    off_times_np (ndarray): times when sync line returns to 0 in a neruopixel stream
    on_times_ecube (ndarray): times when sync line rises to 1 in an ecube stream
    off_times_ecube (ndarray): times when sync line returns to 0 in an ecube stream
    
    Returns:
        tuple: tuple containing:
            |** sync_timestamps (nt):** synchronized timestamps
            |** scaling (float):** scaling factor between streams
    '''
    
    # Get each barcode timing and label
    barcode_ontimes_np,barcode_np = extract_barcodes_from_times(on_times_np, off_times_np, bar_duration=bar_duration)
    barcode_ontimes_ecube,barcode_ecube = extract_barcodes_from_times(on_times_ecube, off_times_ecube, bar_duration=bar_duration)
    
    # Check if barcodes are consistent across streams
    n_barcode = min(len(barcode_ecube),len(barcode_np))
    num_different_barcords = 0
    for idx in range(n_barcode):
        barcode = barcode_ecube[idx]
        if barcode != barcode_np[idx]:
            num_different_barcords += 1
            
    if num_different_barcords > 0:
        raise ValueError("Barcodes in a stream is different from the another stream")
    
    # Get the first and last barcode timing in the recording at each stream
    first_last_times_np, first_last_times_ecube = get_first_last_times(barcode_ontimes_np, barcode_ontimes_ecube, barcode_np, barcode_ecube) 

    return sync_timestamp_offline(raw_timestamp, first_last_times_np, first_last_times_ecube)