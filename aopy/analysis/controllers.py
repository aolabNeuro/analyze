# controllers.py
# 
# Code relating to spectral analysis of feedforward & feedback controllers, based on the paper:
#
# Yamagami M, Peterson LN, Howell D, Roth E, Burden SA. Effect of Handedness on Learned Controllers 
# and Sensorimotor Noise During Trajectory-Tracking. IEEE Trans Cybern. 2023 Apr;53(4):2039-2050. 
# doi: 10.1109/TCYB.2021.3110187. Epub 2023 Mar 16. https://pubmed.ncbi.nlm.nih.gov/34587106/

import numpy as np
import math as m

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