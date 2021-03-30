
import numpy as np
import math
from dpsschk import dpsschk
from nextPowerOfTwo import nextPowerOfTwo
from dmtspec_pt import dmtspec_pt


def tfspec_pt(dN, tapers=None, sampling=1.0, dn=None, fk=None, pad=2, pval=0.05, flag=True, errorchk=False):

    # TFSPEC_PT Moving - window time - frequency point process multitaper spectrum.
    #
    # [SPEC, RATE, F, ERR] = TFSPEC_PT(dN, TAPERS, SAMPLING, DN, FK, PAD, PVAL, FLAG)
    #
    # Inputs: dN = Point process array in [Space / Trials, Time] form.
    # TAPERS = Data tapers in [K, TIME], [N, W], or [N, P, K] form.
    # Defaults to[N, 5, 9] where N is NT / 10.
    # SAMPLING = Sampling rate of point process, dN, in Hz.
    # Defaults to 1.
    # DN = Overlap in time of neighbouring windows.
    # Defaults to N. / 10
    # FK = Frequency range to return in Hz in
    # either[F1, F2] or [F2] form.
    # In[F2] form, F1 is set to 0.
    # Defaults to[0, SAMPLING / 2]
    # PAD = Padding factor for the FFT.
        # i.e.For N = 500, if PAD = 2, we pad the FFT
    # to 1024 points if PAD = 4, we pad the FFT
    # to 2048 points.
    # Defaults to 2.
    # PVAL = P - value to calculate error bars for .
    # Defaults to 0.05 i.e. 95
    #  confidence.
    #
    # FLAG = 0:    calculate SPEC seperately for each channel / trial.
        # FLAG = 1:    calculate SPEC by pooling across channels / trials.
    #
    # Outputs: SPEC = Spectrum of dN in [Space / Trials, Time, Freq] form.
    # RATE = Rate of point process.
    # F = Units of Frequency axis for SPEC.
        # ERR = Error bars in [Hi / Lo, Space, Time, Freq]
    # form given by the Jacknife - t interval for PVAL.
    #

    # Origninal MATLAB Author: Bijan Pesaran,
    # Version date 15 / 10 / 98.

    # Translated by Seth Richards
    # Version Date 2020-06-10

    if isinstance(dN, dict):   # Deprecated behavior from original matlab code, unsure of use case
        raise Exception('Functionality not currently supported, dN should be numpy array')

    sdN = dN.shape
    ntr = sdN[0]  # calculate the number of trials
    nt = sdN[1]
    nch = ntr

    if tapers is None:
        n = math.floor(nt/10)
        tapers, throwAway = dpsschk([n, 5, 9])
        tapers = np.array(tapers)

    if len(tapers[0]) == 2:
        n = tapers[0]
        w = tapers[1]
        p = n * w

        k = math.floor(2 * p - 1)
        if k < 1:
            raise Exception('Must choose N and W so that K > 1')
        if k < 3:
            raise Exception('Warning:  Less than three tapers being used')

        tapers = np.array([n, p, k])

    if len(tapers[0]) == 3:
        n = tapers[0]
        tapers[0] = math.floor(np.multiply(tapers[0],sampling))
        tapers, throwAway = dpsschk(tapers)
        tapers = np.array(tapers)

    tapers = np.around(tapers, 5)

    if dn is None:
        dn = n/10

    if fk is None:
        fk = [0, sampling / 2.]

    if np.size(fk) == 1:
        fk = [0, fk]

    n = math.floor(n * sampling)
    dn = math.floor(np.multiply(dn, sampling))
    nf = np.maximum(256, pad * 2 ** nextPowerOfTwo(n + 1))
    temp = np.multiply(sampling, nf)
    fk = np.true_divide(fk,sampling)
    nfk = np.floor(np.multiply(fk, temp))
    nwin = np.floor(np.true_divide((nt - n), dn))  # calculate the number of windows

    f = np.linspace(fk[0], fk[1], np.diff(nfk)[0])

    if not flag:  # No pooling across trials
        spec = np.zeros([nch, int(nwin), int(np.diff(nfk)[0])])
        rate = np.zeros([nch, int(nwin)])
        err = np.zeros(shape=(2, int(nch), int(nwin), int(np.diff(nfk)[0])), dtype=float)

        for ch in range(nch):

            for win in range(int(nwin)):

                lowerbound = (win + 1) * dn
                upperbound = (win + 1) * dn + n
                dN_tmp = dN[ch, lowerbound:upperbound]

                if not errorchk:  # Don't estimate error bars
                    ftmp, rate_tmp = dmtspec_pt(dN_tmp, tapers, sampling, fk, pad, preslice=True)
                    spec[ch, win, :] = ftmp
                    rate[ch, win] = rate_tmp
                    err = 0

                else:  # Estimate error bars
                    ftmp, rate_tmp, dum, err_tmp = dmtspec_pt(dN_tmp, tapers, sampling, fk, pad, pval, errorchk, preslice=True)
                    spec[ch, win, :] = ftmp
                    rate[ch, win] = rate_tmp
                    err[0, ch, win, :] = err_tmp[0, :]
                    err[1, ch, win, :] = err_tmp[1, :]

    if flag:  # Pooling across trials
        spec = np.zeros([int(nwin), int(np.diff(nfk)[0])])
        rate = np.zeros([int(nwin)])
        err = np.zeros(shape=(2, int(nwin), int(np.diff(nfk)[0])), dtype=float)

        for win in range(int(nwin)):

            lowerbound = (win+1) * dn
            upperbound = (win+1) * dn + n
            dN_tmp = dN[:, lowerbound:upperbound]

            if not errorchk:  # Don't estimate error bars
                ftmp, rate_tmp, throwAway1, throwAway2 = dmtspec_pt(dN_tmp, tapers, sampling, fk, pad, pval, flag)
                spec[win, :] = ftmp
                rate[win] = rate_tmp
                err = 0
            else:  # Estimate error bars
                ftmp, rate_tmp, dum, err_tmp = dmtspec_pt(dN_tmp, tapers, sampling, fk, pad, pval, flag, errorchk)
                spec[win, :] = ftmp
                rate[win] = rate_tmp

                err[0, win, :] = err_tmp[0, :]
                err[1, win, :] = err_tmp[1, :]

    return spec, rate, f, err
