# controllers.py
# 
# Code relating to spectral analysis of feedforward & feedback controllers, based on the paper:
#
# Yamagami M, Peterson LN, Howell D, Roth E, Burden SA. Effect of Handedness on Learned Controllers 
# and Sensorimotor Noise During Trajectory-Tracking. IEEE Trans Cybern. 2023 Apr;53(4):2039-2050. 
# doi: 10.1109/TCYB.2021.3110187. Epub 2023 Mar 16. https://pubmed.ncbi.nlm.nih.gov/34587106/

import math as m
import numpy as np
from .base import calc_freq_domain_values

def get_machine_dynamics(freqs, order, exp_freqs=None):
    '''
    Returns the machine dynamics at each experimental frequency, given the system order. 

    Args:
        freqs (nt/2,): array of non-negative frequencies (essentially the x axis of a spectrogram)
        order (int): the order of the system (e.g. 0 for position control, 1 for velocity control, 2 for acceleration control)
        exp_freqs ((nfreq,), optional): list or array of frequencies used to generate experimental signals (reference and/or disturbance).
            If given, only the machine dynamics at these frequencies will be returned. Otherwise, by default, the machine dynamics at all
            frequencies will be returned.

    Returns:
        M ((nt/2,) or (nfreq,)): array of machine dynamics at each frequency

    '''
    s = 1.j*2*m.pi*freqs

    if order == 0:
        M = 1./np.ones((len(s),))
    elif order == 1:
        M = 1./(s)
    elif order == 2:
        M = 1./(s**2+s)
    else:
        raise ValueError(f"Order {order} not recognized!")

    if exp_freqs is not None:
        exp_idx = np.isin(freqs, exp_freqs)
        M = M[exp_idx]

    return M


def calc_transfer_function(input, output, samplerate, exp_freqs=None):
    '''
    Computes the frequency-domain transformation between time-domain input and ouput signals,
    at each frequency. Assumes output signals are produced by an LTI system.
    For math details, see Yamagami et al. IEEE Transactions on Cybernetics (2021).
    
    Args:
        input (nt, nch): time-domain input signal
        output (nt, nch): time-domain output signal
        samplerate (float): sampling rate of the data
        exp_freqs ((nfreq,), optional): list or array of frequencies used to generate experimental signals (reference and/or disturbance).
            If given, only the transformation at these frequencies will be returned. Otherwise, by default, the transformation at all
            frequencies will be returned.           

    Returns:
        tuple: Tuple containing:
            | **freqs ((nt/2,) or (nfreq,)):** array of frequencies (essentially the x axis of a spectrogram) 
            | **transfer_func ((nt/2, nch) or (nfreq, nch)):** array of complex numbers corresponding to the transformation at each of the above frequencies

    '''    
    assert len(input) == len(output), "Mismatched signal lengths"

    # Compute FFT of time-domain signals
    freqs, freq_input = calc_freq_domain_values(input, samplerate)
    _, freq_output = calc_freq_domain_values(output, samplerate)

    # Divide freq-domain output by freq-domain input at each frequency to find transformation
    transfer_func = np.divide(freq_output, freq_input)

    # Find the transformation at specific frequencies of interest
    if exp_freqs is not None:
        exp_freq_idx = np.isin(freqs.round(4), exp_freqs)
        freqs = freqs[exp_freq_idx]
        transfer_func = transfer_func[exp_freq_idx]

    return freqs.round(4), transfer_func


def pair_trials_by_frequency(ref_freqs, dis_freqs, max_trial_distance=1, limit_pairs_per_trial=True, max_pairs_per_trial=2):
    '''
    Finds pairs of trials with complementary frequency content (e.g. one trial's task signals consist of a subset
    of the experimental frequencies, and the other trial's task signals consist of the remaining experimental frequencies). 
    Assumes that reference and disturbance signals are made up of different frequencies within the same trial.

    Args:
        ref_freqs (ntrial,): list or array of frequencies used to generate the reference signal for each trial
        dis_freqs (ntrial,): list or array of frequencies used to generate the disturbance signal for each trial
        max_trial_distance (int, optional): maximum number of trials allowed between two trials identified as a pair
        limit_pairs_per_trial (bool, optional): whether to limit the number of pairs that any one trial can be included in
        max_pairs_per_trial (int, optional): maximum number of pairs that any one trial can be included in

    Returns:
        trial_pairs (npair, 2): array of trial indices corresponding to pairs of trials with complementary frequency content

    '''
    assert len(ref_freqs) == len(dis_freqs), "Mismatched number of trials"

    trial_pairs = []
    counts = np.zeros(len(ref_freqs),) # number of pairs each trial is included in

    for i in range(len(ref_freqs)):
        if limit_pairs_per_trial and counts[i] >= max_pairs_per_trial:
            continue
        curr_r = np.array(ref_freqs[i]) # convert to array in case in list format
        curr_d = np.array(dis_freqs[i])

        # Find next trial with complementary frequency content
        for j in range(i+1,len(ref_freqs)):
            if j-i > max_trial_distance:
                break
            if limit_pairs_per_trial and counts[j] >= max_pairs_per_trial:
                continue
            next_r = np.array(ref_freqs[j])
            next_d = np.array(dis_freqs[j])

            if (next_r != curr_r).all() and (next_d != curr_d).all():
                trial_pairs.append([i,j])
                counts[i] += 1
                counts[j] += 1
                    
    return np.array(trial_pairs)


def calc_F_B_controllers(usr, ref, dis, exp_freqs, ref_freqs, dis_freqs, samplerate, system_order, trial_pairs=None):
    '''
    Estimates a user's frequency-domain feedforward and feedback controllers at each experimental frequency in a 
    reference-tracking, disturbance-rejection task. Assumes the user responds to reference & disturbance signals like an LTI system.
    For math details, see Yamagami et al. IEEE Transactions on Cybernetics (2021).
    
    Args:
        usr (ntrial, nt): array of time-domain user response signals from each trial
        ref (ntrial, nt): array of time-domain reference signals from each trial 
        dis (ntrial, nt): array of time-domain disturbance signals from each trial
        exp_freqs (nfreq,): list or array of frequencies used to generate experimental signals (reference and/or disturbance)   
        ref_freqs (ntrial,): list or array of frequencies used to generate the reference signal for each trial
        dis_freqs (ntrial,): list or array of frequencies used to generate the disturbance signal for each trial
        samplerate (float): sampling rate of the data
        system_order (int): the order of the system (e.g. 0 for position control, 1 for velocity control, 2 for acceleration control)
        trial_pairs ((npair, 2), optional): list or array of trial indices corresponding to pairs of trials with complementary frequency content.
            If given, controllers will be computed over these trial pairs. Otherwise, by default, controllers will be computed over trials paired 
            up using the default parameters of :func:`~aopy.analysis.controllers.pair_trials_by_frequency`.   
        
    Returns:
        tuple: tuple containing:
            | **F (npair, nfreq):** array of complex numbers corresponding to the user's feedforward controller at each experimental frequency 
            | **B (npair, nfreq):** array of complex numbers corresponding to the user's feedback controller at each experimental frequency
            | **Tur (npair, nfreq):** array of complex numbers corresponding to the ref-->user transformation at each experimental frequency
            | **Tud (npair, nfreq):** array of complex numbers corresponding to the dis-->user transformation at each experimental frequency
            | **M (nfreq,):** array of machine dynamics at each experimental frequency

    '''
    assert len(usr) == len(ref) == len(dis), "Mismatched number of trials"

    # Calculate transfer functions from ref-->user and dis-->user
    trial_Tur = np.array([np.squeeze(calc_transfer_function(ref[i], usr[i], samplerate, exp_freqs)[1]) for i in range(len(usr))])
    trial_Tud = np.array([np.squeeze(calc_transfer_function(dis[i], usr[i], samplerate, exp_freqs)[1]) for i in range(len(usr))])  

    # Get machine dynamics, given system order
    freqs = calc_freq_domain_values(usr[0], samplerate)[0]
    M = get_machine_dynamics(freqs.round(4), system_order, exp_freqs)

    # Pair up trials by frequency content
    if trial_pairs is None:
        trial_pairs = pair_trials_by_frequency(ref_freqs, dis_freqs)

    # Find transfer functions and controllers at the experimental frequencies for each trial pair
    Tur = np.zeros((len(trial_pairs),len(exp_freqs)), dtype=complex)*np.nan
    Tud = np.zeros((len(trial_pairs),len(exp_freqs)), dtype=complex)*np.nan
    B = np.zeros((len(trial_pairs),len(exp_freqs)), dtype=complex)*np.nan
    F = np.zeros((len(trial_pairs),len(exp_freqs)), dtype=complex)*np.nan
    
    for pair_id, (trial_a, trial_b) in enumerate(trial_pairs):
        ref_ind_a = np.isin(exp_freqs, ref_freqs[trial_a])
        dis_ind_a = np.isin(exp_freqs, dis_freqs[trial_a])
        ref_ind_b = np.isin(exp_freqs, ref_freqs[trial_b])
        dis_ind_b = np.isin(exp_freqs, dis_freqs[trial_b])

        Tur[pair_id, ref_ind_a] = trial_Tur[trial_a, ref_ind_a]
        Tur[pair_id, ref_ind_b] = trial_Tur[trial_b, ref_ind_b]
        Tud[pair_id, dis_ind_a] = trial_Tud[trial_a, dis_ind_a]
        Tud[pair_id, dis_ind_b] = trial_Tud[trial_b, dis_ind_b]

        # Compute feedback (B) and feedforward (F) controllers - see eq. 6a and 6b in Yamagami et al., 2023
        # B = -Tud/(M*(1 + Tud))
        # F = (1 + B*M)*(Tur) - B
        B[pair_id,:] = np.divide( -Tud[pair_id], np.multiply( M, np.ones(Tud[pair_id].shape,dtype=complex)+Tud[pair_id] ) )
        F[pair_id,:] = np.multiply( Tur[pair_id], (1+0j)+np.multiply( B[pair_id,:], M ) ) - B[pair_id,:]

    return F, B, Tur, Tud, M


def calc_crossover_freq(freqs, M, B):
    '''
    Estimate crossover frequency, the maximum frequency at which a user can properly 
    track the target. Above crossover, reference-tracking and disturbance-rejection 
    performance degrades. 
    Based on the McRuer crossover model, crossover frequency is the experimental frequency
    at which the gain of the user's open-loop transfer function equals 1 (see McRuer & Jex, 1967).
    Below crossover, the gain is >1. Above crossover, the gain is <1.
    
    Args:
        freqs ((nfreqs,) array): experimental frequencies used to generate reference and disturbance trajectories
        M ((nfreqs,) array): machine dynamics corresponding to the system order (0th order, 1st order, etc.)
        B ((ntrials, nfreqs) complex array): user's feedback controller for each trial (including more trials produces more accurate estimate)
    
    Returns:
        tuple: tuple containing:
            | **xover_freq (float):** estimated crossover frequency
            | **L ((nfreqs,) array):** gain of user's open-loop transfer function
    
    '''
    # estimate the user's open-loop transfer function
    L = np.zeros((len(freqs),))
    for i in range(len(freqs)):
        # |L(w)| = |M(w)*B(w)|
        L[i] = np.mean(abs(M[i]*B[:,i]))
    above_xover_ind = np.where(L<1)[0]
    if len(above_xover_ind) < 1: 
        # crossover occurs at a freq higher than tested
        xover_freq = np.nan
    else:
        # crossover is the lowest freq where L<1
        xover_freq = freqs[above_xover_ind[0]]
    return xover_freq, L