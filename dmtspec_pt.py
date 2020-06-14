
import numpy as np
import math
import nextPowerOfTwo as np2
import dpsschk
from scipy.stats import t
from extendArrayWithCurrentData import extendArrayWithCurrentData


def dmtspec_pt(dN, tapers=None, sampling=1, fk=None, pad=2, pval=0.05, flag=False, errorchk=False,preslice=False):

    # DMTSPEC_PT Point process spectrum using multitaper techniques
    #
    # [SPEC, RATE, F, ERR] = DMTSPEC_PT(dN, TAPERS, SAMPLING, FK, PAD, PVAL, FLAG)
    #
    # Inputs: dN = Point process array in [Space / Trials, Time] form.
    # TAPERS = Data tapers in [K, TIME], [N, W], or [N, P, K] form.
    # Defaults to[N, 5, 9] where N is duration of X.
    # SAMPLING = Sampling rate of point process dN in Hz.
    # Defaults to 1.
    # FK = Frequency range to return in Hz in
    # either[F1, F2] or [F2] form.
    # In[F2] form, F1 is set to 0.
    # Defaults to[0, SAMPLING / 2]
    # PAD = Padding factor for the FFT.
        # i.e.For N = 500, if PAD = 2, we pad the FFT
    # to 1024 points; if PAD = 4, we pad the FFT
    # to 2048 points.
    # Defaults to 2.
    # PVAL = P - value to calculate error bars for .
    # Defaults to 0.05 i.e. 95 percent confidence.
    #
    # FLAG = 0:    calculate SPEC seperately for each channel / trial.
        # FLAG = 1:    calculate SPEC by pooling across channels / trials.
    # ERRORCHK = Error bars calculation if true
    # PRESLICE = links functionality with tfspec_pt flag = 0

    # Outputs: SPEC = Spectrum of dN in [Space / Trials, Freq] form.
    # RATE = Mean rate of dN in Hz.
    # F = Units of Frequency axis for SPEC.
    # ERR = Error bars for SPEC in [Hi / Lo, Space / Trials, Freq]
        # form given by a Jacknife - t interval for PVAL.

    # Modification History: Rewritten by Bijan Pesaran 02 / 04 / 00
    # June 2004: Added cell array spike time inputs

    # Translated by Seth Richards from original MATLAB code
    # Version Date 2020-06-14

    # No tapers == 1 condition, assuming tapers given from tfspec_pt.py

    tapers = np.asarray(tapers)

    if len(tapers[0]) == 2:
        n = tapers[0]
        w = tapers[1]
        p = n * w
        k = math.floor(2 * -1)
        tapers = [n, p, k]

    if len(tapers[0]) == 3:
        tapers[0] = math.floor(np.multiply(tapers[0], sampling))
        tapers, v = dpsschk.dpsschk(tapers)

    if fk is None:
        fk = [0, sampling/2]

    if np.size(fk) == 1:
        fk = [0, fk]

    nt = len(tapers[0])

    if isinstance(dN, dict):
        raise Exception('Functionality not currently supported')

    nch = dN.shape[0]
    N = len(tapers[0])

    if N != nt:
        raise Exception('Length of time series and tapers must be equal');

    K = len(tapers[1])
    nf = np.maximum(256, pad * 2 ** (np2.nextPowerOfTwo(N + 1)))
    temp1 = np.multiply(fk, np.true_divide(nf, sampling))
    temp2 = np.array(temp1, dtype=float)
    nfk = (np.floor(temp2))
    dof = 2. * nch * K

    # Determine outputs
    f = np.linspace(fk[0], fk[1], np.diff(nfk)[0])

    ntapers = np.multiply(tapers, np.sqrt(sampling))

    # Calculate the Slepian transforms.
    inputArray = np.transpose(ntapers[:, 0: K])
    H = np.fft.fft(inputArray, int(nf))
    H = np.conj(H)  # fft produces conjugate of MATLAB fft

    if not flag:  # No pooling across channels / trials

        spec = np.zeros(shape=(int(nch), int(np.diff(nfk)[0])), dtype=float)
        err = np.zeros(shape=(2, int(nch), int(np.diff(nfk)[0])), dtype=float)
        rate = np.zeros(shape=(1, int(nch)), dtype=float)

        #
        # This is the fourier transform loop
        # The difference between spectral analysis for
        # continuous and point processes is here.
        # We take the tapered fourier transform and
        # subtract the mean number of spikes multiplied
        # by | H | ^ 2 which is the projection of DC into the frequency
        # domain.
        #

        for ch in range(nch):

            if preslice:  # original <--- from tfspec_pt ch should be pre-sliced if flag = 0
                tmp = dN
            else:
                tmp = dN[ch, :]

            fftInput = np.multiply(ntapers[:, 0:K], extendArrayWithCurrentData(tmp, 0, K))
            fftInput = np.transpose(fftInput)

            dNk = np.fft.fft(fftInput, int(nf))
            hproduct = np.multiply(H, tmp.mean(axis=0))
            dNk = np.conj(dNk) - hproduct

            lower = int(nfk[0])
            upper = int(nfk[1])
            dNk = dNk[:, lower:upper]

            dNkSquared = np.absolute(dNk)
            dNkSquared = np.multiply(dNkSquared, dNkSquared)

            Sk = np.mean(dNkSquared, axis=0)

            spec[ch, :] = Sk.mean(axis=0)

            rateInputProduct = np.multiply(ntapers[:, 0:K], extendArrayWithCurrentData(tmp, 0, K))
            rateInputProduct = np.around(rateInputProduct, 4)

            rateInput = np.multiply(rateInputProduct, rateInputProduct)
            rateInput = np.sum(rateInput, 0)
            newRate = np.mean(rateInput)

            rate[ch, :] = newRate

            if errorchk:  # Estimate error bars using Jacknife

                jlsp = np.zeros(shape=(int(K * nch), dNk.shape[1]), dtype=float)

                for ik in range(K):
                    indices = np.setdiff1d(np.arange(0, K), ik)
                    dNj = dNk[indices, :]

                    dNjSquared = np.absolute(dNj)
                    dNjSquared = np.multiply(dNjSquared, dNjSquared)
                    jlspInside = np.mean(dNjSquared, axis=0)

                    jlsp[ik, :] = np.log(jlspInside)

                lsig = np.multiply(math.sqrt(K - 1), np.std(jlsp, axis=0))

                # Determine the scaling factor, using student's t cdf inverse
                crit = t.ppf(1 - np.true_divide(pval, 2), dof - 1)
                critlsig = np.multiply(crit, lsig)
                err[0, ch, :] = np.exp(np.log(spec[ch, :]) + critlsig)
                err[1, ch, :] = np.exp(np.log(spec[ch, :]) - critlsig)

    if flag:  # Pooling across trials
        err = np.zeros(shape=(2, int(np.diff(nfk)[0])), dtype=float)
        rate = 0

        dNk = np.zeros(shape=(int(nch * K), int(np.diff(nfk)[0])), dtype=complex)

        for ch in range(nch):

            if preslice:  # original <--- from tfspec_pt ch should be pre-sliced if flag = 0
                tmp = dN
            else:
                tmp = dN[ch, :]

            fftInput = np.multiply(ntapers[:, 0:K], extendArrayWithCurrentData(tmp, 0, K))
            fftInput = np.transpose(fftInput)

            xk = np.fft.fft(fftInput, int(nf))
            hproduct = np.multiply(H, tmp.mean(axis=0))
            xk = np.conj(xk) - hproduct

            rateInputProduct = np.multiply(ntapers[:, 0:K], extendArrayWithCurrentData(tmp, 0, K))
            rateInputProduct = np.around(rateInputProduct, 4)
            rateInput = np.multiply(rateInputProduct, rateInputProduct)
            rateInput = np.sum(rateInput, 0)
            newRate = np.mean(rateInput)

            rate = rate + newRate

            dNk[ch*K:(ch+1)*K, :] = xk[:, int(nfk[0]): int(nfk[1])]

        dNkSquared = np.absolute(dNk)
        dNkSquared = np.multiply(dNkSquared,dNkSquared)

        spec = np.mean(dNkSquared, axis=0)

        rate = np.true_divide(rate, nch)

        if errorchk:  # Estimate error bars using Jacknife

            jlsp = np.zeros(shape=(int(K * nch), dNk.shape[1]), dtype=float)

            for ik in range(nch * K):
                indices = np.setdiff1d(np.arange(0, K*nch), ik)
                dNj = dNk[indices, :]

                dNjSquared = np.absolute(dNj)
                dNjSquared = np.multiply(dNjSquared, dNjSquared)
                jlspInside = np.mean(dNjSquared, axis=0)
                jlsp[ik, :] = np.log(jlspInside)

            lsig = np.multiply(np.sqrt((nch * K) - 1), np.std(jlsp,axis=0))
            crit = t.ppf(1 - np.true_divide(pval, 2), dof - 1)  # Determine the scaling factor
            critlsig = np.multiply(crit, lsig)

            err[0, :] = np.exp(np.log(spec) + critlsig)
            err[1, :] = np.exp(np.log(spec) - critlsig)

    return spec, rate, f, err
