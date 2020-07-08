import numpy as np
import math
import nextPowerOfTwo as np2
import dpsschk
from scipy.stats import t,chi2
from extendArrayWithCurrentData import extendArrayWithCurrentData


def dmtspec(X, tapers = None, sampling = 1, fk = None, pad = 2, pval = 0.05, flag = 0, Errorbar = False):
    # function[spec, f, err] = dmtspec(X, tapers, sampling, fk, pad, pval, flag, Errorbar)
    # DMTSPEC calculates the direct multitaper spectral estimate for time series.
    #
    # [SPEC, F, ERR] = DMTSPEC(X, TAPERS, SAMPLING, FK, PAD, PVAL, FLAG, ERRORBAR)
    #
    # Inputs: X = Time series array in [Space / Trials, Time] form.
    # TAPERS = Data tapers in [K, TIME], [N, P, K] or [N, W] form.
    # Defaults to[N, 3, 5] where N is duration of X.
    # SAMPLING = Sampling rate of time series X in Hz.
    # Defaults to 1.
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
    # Defaults to 0.05 i.e. 95 # confidence.
    #
    # FLAG = 0:    calculate SPEC seperately for each channel / trial.
    # FLAG = 1:    calculate SPEC by pooling across channels / trials.
    #
    # ERRORBAR = Structure.ERRRORBAR.Type - 'Jacknife' or 'Chi-squared'
    # Defaults to Chi Squared if false
    # for Jacknife, enter True
    #
    #
    #
    # Outputs: SPEC = Spectrum of X in [Space / Trials, Freq] form.
    # F = Units of Frequency axis for SPEC.
    # ERR = Error bars for SPEC in [Hi / Lo, Space / Trials, Freq]
    # form given by a Jacknife - t interval for PVAL.
    #

    # Origninal MATLAB Code adapted from:
    # Modification History:
    # Written by: Bijan Pesaran, 0 8 / 97
    # Modified: Added error bars BP 08 / 27 / 98

    # Author: Seth Richards
    # Version Date: 2020/06/14

    jlsp = 0
    sX = X.shape
    nt = sX[1]
    nch = sX[0]

    errorchk = True

    if Errorbar:
        errorbarType = 'Jacknife'
    else:
        errorbarType = 'Chi-squared'

    # Set the defaults
    nt = np.true_divide(nt, sampling)
    if tapers == None:
        tapers = [nt, 3, 5]

    if len(tapers[0]) == 2:
        n = tapers[0]
        w = tapers[1]
        p = n * w
        k = math.floor(2 * p - 1)
        tapers = [n, p, k]
        # disp(['Using ' num2str(k) ' tapers.'])

    if len(tapers[0]) == 3:
        tapers[0] = math.floor(np.multiply(tapers[0], sampling))
        tapers, v = dpsschk.dpsschk(tapers)

    if fk is None:
        fk = [0, np.true_divide(sampling, 2)]

    if np.size(fk) == 1:
        fk = [0, fk]

    N = np.size(tapers)/tapers.shape[1]
    nt = math.floor(np.multiply(nt, sampling))

    if N != nt:
        raise Exception('Error:  Length of time series and tapers must be equal')
    #K = tapers(1,:).shape[1]
    K = tapers.shape[1]
    nf = np.maximum(256, pad * 2 ** (np2.nextPowerOfTwo(N + 1)))
    temp1 = np.multiply(fk,np.true_divide(nf,sampling))
    temp2 = np.array(temp1, dtype=float)
    nfk = (np.floor(temp2))
    dof = 2. * nch * K

    # Determine outputs
    f = np.linspace(fk[0], fk[1], np.diff(nfk)[0])
    if not flag:  # No pooling across trials
        # disp('No pooling across trials')

        spec = np.zeros(shape=(int(nch),int(np.diff(nfk)[0])), dtype=float)
        err = np.zeros(shape=(2, int(nch), int(np.diff(nfk)[0])),dtype=float)

        if nch == 1:
            mX = np.true_divide(np.sum(X,0), nt)
        else:
            mX = np.true_divide(np.sum(X, 0), nch)

        for ch in range(nch):
            tmp = np.transpose((X[ch, :] - mX))

            # Assumes tmp is an array of size (X,1)
            tmp = extendArrayWithCurrentData(tmp, 0, K)
            inputArray = np.multiply(tapers[:, 0:K+1], tmp)
            inputArray = np.transpose(np.around(inputArray, 4))

            # N-point Fourier transform
            xk = np.fft.fft(inputArray, int(nf))

            lowerBound = int(nfk[0])
            upperBound = int(nfk[1])
            xk = xk[:, lowerBound:upperBound]

            Sk = np.multiply(xk, xk.conjugate())

            # Casting complex to real (Ignore Warning)
            Sk = np.around(np.array(Sk, np.float32), 6)

            spec[ch, :] = np.true_divide(np.sum(Sk, 0), K)

            if errorchk:  # Estimate error bars using Jacknife
                if errorbarType == 'Jacknife':
                    for ik in range(K):
                        indices = np.setdiff1d(np.arange(0, K),ik)
                        xj = xk[indices, :]

                        xjSquared = np.absolute(xj)
                        xjSquared = np.multiply(xjSquared, xjSquared)
                        jlspInside = np.mean(xjSquared, axis=0)

                        jlsp[ik,:] = math.log(np.true_divide(jlspInside, (K - 1)))

                    lsig = np.multiply(math.sqrt(K - 1), np.std(jlsp, axis=0))
                    crit = t.ppf(1 - np.true_divide(pval, 2), dof - 1)  # Determine the scaling factor, using student's t cdf inverse
                    critlsig = np.multiply(crit, lsig)
                    err[0, ch, :] = np.exp(np.log(spec[ch, :])+critlsig)
                    err[1, ch, :] = np.exp(np.log(spec[ch, :])-critlsig)

                elif errorbarType == 'Chi-squared':  # if == 'Chi-squared'

                    a = chi2.ppf(1 - np.true_divide(pval, 2), dof)
                    b = chi2.ppf(np.true_divide(pval, 2), dof)
                    err[0, ch,:] = np.true_divide(np.multiply(spec[ch, :], dof), b)
                    err[1, ch,:] = np.true_divide(np.multiply(spec[ch, :], dof), a)

    if flag:  # Pooling across trials
        err = np.zeros(2, np.diff(nfk)[0])

        Xk = np.zeros(nch * K, np.diff(nfk)[0])
        mX = np.true_divide(np.sum(X, 0), nch)

        for ch in range(nch):
            tmp = np.transpose((X[ch, :] - mX))

            # Assumes tmp is an array of size (X,1)
            tmp = extendArrayWithCurrentData(tmp, 0, K)
            inputArray = np.multiply(tapers[:, 0:K + 1], tmp[:])
            inputArray = np.transpose(np.around(inputArray, 4))

            # N-point Fourier transform (I know all the transposes are weird.
            # python dimensions are "interesting"
            lowerBound = int(nfk[0])
            upperBound = int(nfk[1])
            xk = np.fft.fft(inputArray, int(nf))
            xk = xk[:, lowerBound:upperBound]
            Xk[ch*K:(ch+1)*K, :] = xk

        Sk = np.multiply(Xk, Xk.conjugate())

        # Casting complex to real (Ignore Warning)
        Sk = np.around(np.array(Sk, np.float32), 6)

        spec = np.true_divide(np.sum(Sk, 1), (np.multiply(K, nch)))

        if errorchk:  # Estimate error bars
            if errorbarType == 'Jackknife':

                for ik in range(nch*K):
                    if np.mod(ik, 1000) == 0:
                        print("Jacknife iter: %d of %d" % ik, (nch * K))

                    indices = np.setdiff1d(np.arange(0,K*nch), ik)
                    xj = Xk[indices, :]

                    xjSquared = np.absolute(xj)
                    xjSquared = np.multiply(xjSquared, xjSquared)
                    jlspInside = np.mean(xjSquared, axis=0)

                    jlsp[ik,:] = math.log(np.true_divide(jlspInside, (K * nch - 1)))

                lsig = np.multiply( np.sqrt((nch * K) - 1), np.std(jlsp,axis=0))
                crit = t.ppf(1 - np.true_divide(pval, 2), dof - 1) # Determine the scaling factor
                critlsig = np.multiply(crit, lsig)
                err[0,:] = np.exp(np.log(spec) + critlsig)
                err[1,:] = np.exp(np.log(spec) - critlsig)

            elif errorbarType == 'Chi-squared':

                a = chi2.ppf(1 - np.true_divide(pval, 2), dof)
                b = chi2.ppf(np.true_divide(pval, 2), dof)
                err[0, :] = np.true_divide(np.multiply(spec, dof), b)
                err[1, :] = np.true_divide(np.multiply(spec, dof), a)

    return spec, f, err
