import numpy as np
from ..utils import extract_barcodes_from_times, get_first_last_times, sync_timestamp_offline

def sync_neuropixel_ecube(raw_timestamp,on_times_np,off_times_np,on_times_ecube,off_times_ecube,bar_duration=0.02,verbose=False):
    '''
    This function is specfic to synchronization between neuropixels and ecube.
    
    Args:
    raw_timestamp (nt) : raw timestamp that is not synchronized
    on_times_np (ndarray) : timings when sync line rises to 1 in neruopixel streams
    off_times_np (ndarray): timings when sync line returns to 0 in neruopixel streams
    on_times_ecube (ndarray): timings when sync line rises to 1 in ecube streams
    off_times_ecube (ndarray): timings when sync line returns to 0 in ecube streams
    bar_duration (float): duration of each bar that is sent to each stream
    verbose (bool): print barcode times and barcodes for each stream
    
    Returns:
        tuple: tuple containing:
            |** sync_timestamps (nt):** synchronized timestamps
            |** scaling (float):** scaling factor between streams
    '''
    
    while bar_duration >0.001:
        # Get each barcode timing and label
        barcode_ontimes_np,barcode_np = extract_barcodes_from_times(on_times_np,off_times_np,bar_duration=bar_duration)
        barcode_ontimes_ecube,barcode_ecube = extract_barcodes_from_times(on_times_ecube,off_times_ecube,bar_duration=bar_duration)

        # Check if barcodes are consistent across streams
        n_barcode = min(len(barcode_ecube),len(barcode_np))
        num_different_barcodes = 0
        for idx in range(n_barcode):
            barcode = barcode_ecube[idx]
            if barcode != barcode_np[idx]:
                num_different_barcodes += 1
        
        # If barcodes are the same across streams, break this loop
        if num_different_barcodes == 0:
            break
        bar_duration -= 0.0001
            
    if verbose:
        print(f'bar duration: {bar_duration}\n')
        print(f'neuropixel barcode times: {barcode_ontimes_np}\n')
        print(f'neuropixel barcode: {barcode_np}\n')
        print(f'ecube barcode times: {barcode_ontimes_ecube}\n')
        print(f'ecube barcode: {barcode_ecube}\n')
        
    # Get the first and last barcode timing in the recording at each stream
    first_last_times_np, first_last_times_ecube = get_first_last_times(barcode_ontimes_np,barcode_ontimes_ecube,barcode_np, barcode_ecube) 

    return sync_timestamp_offline(raw_timestamp, first_last_times_np, first_last_times_ecube)