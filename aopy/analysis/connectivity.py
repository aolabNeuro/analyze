import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp

from . import base
from .. import data as aodata
from .. import precondition
from . import accllr

def get_acq_ch_near_stimulation_site(stim_site, stim_layout='Opto32', electrode_layout='ECoG244', 
                                 dist_thr=1, return_idx=False):
    '''
    Get acquisition channels near a stimulation site. Use :func:`~aopy.data.load_chmap` to find the
    channels for the stimulation and electrode sites.

    Note: 
        This function returns channel
        numbers, which sometimes are 1-indexed. Set the return_idx flag to True to get the channel
        indices as well.

    Args:
        stim_site (int): stimulation site (must match a channel in the stim_layout)
        stim_layout (str): layout of stimulation sites, e.g. 'Opto32'. See 
            :func:`~aopy.data.load_chmap` for options.
        electrode_layout (str): layout of electrodes, e.g. 'ECoG244'. See
            :func:`~aopy.data.load_chmap` for options.
        dist_thr (float or tuple (min, max), optional): threshold for distance from stimulation site (in 
            the same units as the electrode layout, typically mm). If a tuple, the distance must be greater 
            than or equal to min and less than max. Default is 1.
        return_idx (bool, optional): if True, return the channel indices as well. Default is False.

    Returns:
        acq_ch: np.ndarray, acquisition channels near stimulation site
        | or **(acq_ch, idx) tuple**, if return_idx is True
    '''
    elec_pos, acq_ch, _ = aodata.load_chmap(electrode_layout)
    stim_pos, stim_ch, _ = aodata.load_chmap(stim_layout)
    stim_site_pos = stim_pos[stim_ch == stim_site]
    if stim_site_pos.size == 0:
        raise ValueError(f"stim_site site {stim_site} not found in layout {stim_layout}")
    
    dist = np.linalg.norm(elec_pos - stim_site_pos, axis=1)
    if np.size(dist_thr) == 2:
        idx = (dist < dist_thr[1]) & (dist >= dist_thr[0])
    elif np.size(dist_thr) == 1:
        idx = dist < dist_thr
    else:
        raise ValueError("dist_thr must be a float or tuple (min, max) of floats")

    if return_idx:
        return acq_ch[idx], np.where(idx)[0]
    else:
        return acq_ch[idx]


def calc_connectivity_coh(data_altcond_source, data_altcond_probe, n, p, k, 
                          samplerate, step, fk=250, pad=2,
                          imaginary=True, average=True):
    '''
    Calculate the average time-frequency cohereogram between multiple source 
    and probe channels. Iterates through every possible pair (order doesn't matter) 
    of source and probe channels and calculates the coherence between them. Optionally
    returns the average across all pairs. This function is called by 
    :func:`calc_connectivity_map_coh` to calculate the coherence between a single channel
    and multiple channels around the stimulation site. No re-referencing is done here,
    if you want to re-reference the data, do it before calling this function.

    Args:
        data_altcond_source (nt, n_source, ntrial): source erp data
        data_altcond_probe (nt, n_probe, ntrial): probe erp data
        n (float): window length in seconds
        p (float): standardized half bandwidth in hz
        k (int): number of DPSS tapers to use
        fs (float): sampling rate in Hz.
        step (float): window step size in seconds.
        fk (float, optional): frequency range to return in Hz ([0, fk]). Default is fs/2.
        pad (int, optional): padding factor for the FFT. This should be 1 or a multiple of 2.
            For nt=500, if pad=1, we pad the FFT to 512 points.
            If pad=2, we pad the FFT to 1024 points. 
            If pad=4, we pad the FFT to 2024 points.
            Default is 2.
        imaginary (bool, optional): if True, compute imaginary coherence.        
        average: bool, whether to average the coherence across all pairs

    Returns:
        tuple: tuple containing:
        | **f (n_freq):** frequency axis
        | **t (nt):** time axis
        | **coh (list of (n_freq,nt)):** magnitude squared coherence or imaginary coherence 
            (0 <= coh <= 1) between the pairs
        | **angle ((list of n_freq,nt)):** list of phase difference (in radians) between the pairs 
            (optional output, -pi <= angle <= pi)
        | **pair (list of tuples):** list of channel pairs
        | or **(freqs, time, coh, angle)** tuple**, if average is True
    '''
    data_altcond = np.concatenate((data_altcond_source, data_altcond_probe), axis=1)
    stim_coh = []
    stim_angle = []
    pair = []
    n_source = data_altcond_source.shape[1]
    n_probe = data_altcond_probe.shape[1]
    for source_idx in range(n_source):
        for probe_idx in range(n_probe):
            ch_pair = np.array([source_idx, n_source+probe_idx])
            if set(ch_pair) in pair: # skip the reciprocal pairs
                continue
            freqs, time, coh, angle = base.calc_mt_tfcoh(data_altcond, ch_pair, n, p, k, samplerate, 
                                                        step=step, fk=fk, pad=pad, imaginary=imaginary, 
                                                        ref=False, return_angle=True)
            stim_coh.append(coh)
            stim_angle.append(angle)
            pair.append(set(ch_pair))

    if average:
        return freqs, time, np.mean(stim_coh, axis=0), np.mean(stim_angle, axis=0)
    else:
        # Remove the offset in pair
        pair = [(tuple(p)[0], tuple(p)[1]-n_source) for p in pair]
        return freqs, time, stim_coh, stim_angle, pair

def calc_connectivity_map_coh(erp, samplerate, time_before, time_after, stim_ch_idx, n=0.06, 
                              step=0.03, bw=25, ref=True, parallel=False, imaginary=True):
    '''
    Map of coherence at every channel to the given stimulation channels. Input ERP data must include
    at least `n` seconds before and after events. Coherence is averaged across stimulation channels 
    if multiple are given. 

    Args:
        erp ((nt, nch, ntr) array): trial-aligned data
        samplerate (float): sampling rate of the erp
        time_before (float): time included before events in the erp (in seconds)
        time_after (float): time included after events in the erp (in seconds)
        stim_ch_idx (list of 0-indexed int): stimulation channel indices (where you want coherence to be calculated from)
        n (float): window length in seconds for the coherence calculation (default 0.06 s).
        step (float): window step size in seconds for the coherence calculation (default 0.03 s).
        bw (float): bandwidth for multitaper filter (default 25).
        ref (bool): re-referencing flag (default True).
        parallel (bool or mp.pool.Pool): whether to use parallel processing. Can optionally be a pool object
            to use an existing pool. If True, a new pool is created with the number of CPUs available. If False,
            computation is done serially (the default).
        imaginary (bool): if True, compute imaginary coherence (the default).
    
    Returns:
        tuple: tuple containing:
        | **freqs (n_freq):** frequency axis
        | **time (nt):** time axis
        | **coh_all (n_freq, nt, nch):** magnitude squared coherence or imaginary coherence 
            (0 <= coh <= 1) between the pairs at each channel
        | **angle_all (n_freq, nt, nch):** phase difference (in radians) between the pairs
            at each channel

    Note:
        This is not the most efficient way to compute pairwise coherence since we end up repeating
        the same calculations for each channel multiple times. Maybe a future enhancement. See the
        implementation in the package `spectral_connectivity` for a more time-efficient (but memory-
        inefficient) algorithm.

    Examples:
    
        Create a grid of channels with mostly noise but two channels have 50 Hz sine waves 
    
        .. code-block:: python

            grid_size = 3
            nch = grid_size**2
            T = 1
            fs = 1000
            nt = int(T*fs)
            ntr = 2
            time = np.linspace(0,T,nt)
            data = np.random.normal(0, 0.1, (nt,nch,ntr)) # start with noise
            stim_ch_idx = 0
            data[:,stim_ch_idx,0] += np.sin(2*np.pi*50*time) # 50 Hz sine
            data[:,stim_ch_idx,1] += np.sin(2*np.pi*50*time) 
            data[500:,4,0] += np.cos(2*np.pi*50*time[500:]) # 50 Hz cosine in second half of trial
            data[500:,4,1] += np.cos(2*np.pi*50*time[500:]) 

            n = 0.25
            w = 10
            step = 0.25
            f, t, coh_all, angle_all = aopy.analysis.connectivity.calc_connectivity_map_coh(data, fs, 0.5, 0.5, [stim_ch_idx], 
                                                                                    n=n, bw=w, step=step, ref=False)

            self.assertEqual(coh_all.shape, angle_all.shape)
            
            bands = [(40, 60), (100, 250)]
            x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
            elec_pos = np.zeros((nch,2))
            elec_pos[:,0] = x.reshape(-1)
            elec_pos[:,1] = y.reshape(-1)
            aopy.visualization.plot_tf_map_grid(f, t, coh_all, bands, elec_pos, clim=(0,1), interp_grid=None, 
                        cmap='viridis')
        
        .. image:: _images/connectivity_map_coh.png'
    '''
    assert erp.ndim == 3, "ERP data must be 3D (nt, nch, ntr)"
    assert time_before >= n, "time_before must be greater than or equal to n"
    assert time_after >= n, "time_after must be greater than or equal to n"

    n, p, k = precondition.convert_taper_parameters(n, bw)
    print(f"using {k} tapers for tfcoh")
    
    altcond_window = (-n, n)
    nullcond_window = (0, n)
    data_altcond, _, _, _ = accllr.prepare_erp(
        erp, erp, samplerate, time_before, time_after, nullcond_window, altcond_window)
    
    # Re-reference the data once
    if ref:
        data_altcond = data_altcond - np.mean(data_altcond, axis=1, keepdims=True) # mean across channels

    # Create a parallel pool if requested
    pool = None
    if parallel is True: # create a parallel pool
        pool = mp.Pool(min(mp.cpu_count()//2, erp.shape[1]))
    elif type(parallel) is mp.pool.Pool: # use an existing pool
        pool = parallel

    # Calculate coherence for each channel
    coh_all = []
    angle_all = []
    freqs = None
    time = None
    if pool:
        
        # call apply_async() without callback
        result_objects = [pool.apply_async(calc_connectivity_coh, 
                          args=(data_altcond[:,[ch],:], data_altcond[:,stim_ch_idx,:], n, p, k, samplerate, step),
                          kwds={'imaginary': imaginary})
                          for ch in range(erp.shape[1])]

        # result_objects is a list of pool.ApplyResult objects
        results = [r.get() for r in result_objects]
        freqs, time, coh_all, angle_all = zip(*results)
        freqs = freqs[0]
        time = time[0]
        if parallel is True:
            pool.close()

    else:
        for ch in tqdm(range(erp.shape[1])):

            freqs, time, coh_avg, angle_avg = calc_connectivity_coh(
                data_altcond[:,[ch],:], data_altcond[:,stim_ch_idx,:], n, p, k, samplerate, step,
                imaginary=imaginary
            )
            coh_all.append(coh_avg)
            angle_all.append(angle_avg)

    # Move time to the first axis and channels to the end
    coh_all = np.array(coh_all).transpose(1,2,0)
    angle_all = np.array(angle_all).transpose(1,2,0)
    
    return freqs, time-n, coh_all, angle_all

