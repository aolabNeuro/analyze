
import numpy as np
import math
from dpsschk import dpsschk
from nextPowerOfTwo import nextPowerOfTwo
from extendArrayWithCurrentData import extendArrayWithCurrentData
from scipy.stats import t

def coherency(X, Y, tapers = None, sampling = 1, fk = None, pad = 2, pval = 0.05, flag = 11, contflag = 0,errorchk = True):

    # COHERENCY calculates the coherency between two time series, X and Y
    #
    # [COH, F, S_X, S_Y,, COH_ERR, SX_ERR, SY_ERR] = ...
    # COHERENCY(X, Y, TAPERS, SAMPLING, FK, PAD, PVAL, FLAG, CONTFLAG)
    #
    # Inputs: X = Time series array in [Space / Trials, Time] form.
    # Y = Time series array in [Space / Trials, Time] form.
    # TAPERS = Data tapers in [K, TIME], [N, P, K] or [N, W] form.
    # Defaults to[N, 5, 9] where N is the duration
    # of X and Y.
    # SAMPLING = Sampling rate of time series X, in Hz.
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
    # Defaults to 0.05 i.e. 95 % confidence.
    #
    # FLAG = 0:    calculate COH separately for each channel / trial.
    # FLAG = 1:    calculate COH by pooling across channels / trials.
    # FLAG = 11 calculation is done as for FLAG = 1
    # but the error bars cannot be calculated to save memory.
    # Defaults to FLAG = 11.
    # CONTFLAG = 1; There is only a single continuous signal coming in.
    # Defaults to 0.
    #
    # Outputs: COH = Coherency between X and Y in [Space / Trials, Freq].
    # F = Units of Frequency axis for COH
        # S_X = Spectrum of X in [Space / Trials, Freq] form.
    # S_Y = Spectrum of Y in [Space / Trials, Freq] form.
    # COH_ERR = Error bars for COH in [Hi / Lo, Space, Freq]
        # form given by the Jacknife - t interval for PVAL.
            # SX_ERR = Error bars for S_X.
            # SY_ERR = Error bars for S_Y.
            #

    # Original Matlab code written by: Bijan Pesaran Caltech 1998
    # Modified: September 2003.

    # Python translated by: Seth Richards
    # Version Date 2020/06/14

    # LAG = 0 may not function as intended

    sX = X.shape
    nt1 = sX[1]
    nch1 = sX[0]

    sY = Y.shape
    nt2 = sY[1]
    nch2 = sY[0]

    if nt1 != nt2:
        raise Exception('Error: Time series are not the same length')

    if nch1 != nch2:
        raise Exception('Error: Time series are incompatible')

    nt = nt1
    nch = nch1

    nt = np.true_divide(nt,sampling)
    if tapers is None:
        tapers = np.array([nt, 5, 9])

    if len(tapers[0]) == 2:
        n = tapers[0]
        w = tapers[1]
        p = n * w
        k = math.floor(2 * p - 1)
        tapers = [n, p, k]
        print(['Using ', tapers, ' tapers.'])

    if len(tapers[0]) == 3:
        tapers[0] = math.floor(np.multiply(tapers[0], sampling))
        tapers, throwAway = dpsschk(tapers)

    if fk is None:
        fk = [0, np.true_divide(sampling, 2)]

    if np.size(fk) == 1:
        fk = [0, fk]

    K = tapers.shape[1]
    N = tapers.shape[0]

    if N != nt * sampling:
        raise Exception('Error: Tapers and time series are not the same length');

    nf = np.maximum(256, pad * 2 ** nextPowerOfTwo(N + 1))

    temp = np.multiply(sampling, nf)
    fk = np.true_divide(fk,sampling)
    nfk = np.floor(np.multiply(fk, temp))
    Nf = int(np.diff(nfk)[0])

    # Determine outputs
    f = np.linspace(fk[0], fk[1], Nf)

    # Default variables if not checking for error
    coh_err = None
    SX_err = None
    SY_err = None

    if flag == 0:
        coh = np.zeros([nch, Nf])
        S_X = np.zeros([nch, Nf])
        S_Y = np.zeros([nch, Nf])

        if errorchk:
            coh_err = np.zeros([2, nch, Nf])
            SX_err = np.zeros([2, nch, Nf])
            SY_err = np.zeros([2, nch, Nf])

        if contflag == 0:
            m1 = np.sum(X, axis=0)
            mX = np.transpose(np.true_divide(m1, nch))
            mY = np.sum(Y, axis=0)

        for ch in range(nch):
            if contflag == 1:
                tmp1 = np.transpose(X[ch, :]) - np.true_divide(X[ch, :].sum(axis=0), N)

                tmp2 = np.transpose(Y[ch, :]) - np.true_divide(Y[ch, :].sum(axis=0), N)
            else:
                tmp1 = np.transpose(X[ch, :]) - mX
                tmp2 = np.transpose(Y[ch, :]) - mY

            extendedArrayX = extendArrayWithCurrentData(tmp1, ch, K)
            inputArrayX = np.multiply(tapers[:, 0:K], extendedArrayX)
            Xk = np.fft.fft(np.transpose(inputArrayX), int(nf))
            lowerBoundX = int(nfk[0])
            upperBoundX = int(nfk[1])
            Xk = Xk[:,lowerBoundX:upperBoundX]

            extendedArrayY = extendArrayWithCurrentData(tmp2, ch, K)
            inputArrayY = np.multiply(tapers[:, 0:K], extendedArrayY)
            Yk = np.fft.fft(np.transpose(inputArrayY), int(nf))
            lowerBoundY = int(nfk[0])
            upperBoundY = int(nfk[1])
            Yk = Yk[:, lowerBoundY:upperBoundY]

            SXk = (Xk * np.conj(Xk)).real
            SYk = (Yk * np.conj(Yk)).real
            SXTemp = np.sum(SXk, axis=0)

            S_X[ch, :] = np.transpose(np.true_divide(SXTemp, K))
            SYTemp = np.sum(SYk, axis=0)
            S_Y[ch, :] = np.true_divide(SYTemp, K)

            cohTemp = np.sum(np.multiply(Xk, np.conj(Yk)), axis=0)
            cohTemp1 = np.sqrt(np.multiply(S_X[ch, :], S_Y[ch, :]))
            coh[ch, :] = np.true_divide(np.true_divide(cohTemp.real, K), cohTemp1.real)

            if errorchk:  # Estimate error bars using Jacknife
                jcoh = np.zeros([K, Nf])
                jXlsp = np.zeros([K, Nf])
                jYlsp = np.zeros([K, Nf])
                for ik in range(K):
                    tempArray = range(0, K)
                    indices = np.setdiff1d(tempArray, ik)
                    Xj = Xk[indices, :]
                    Yj = Yk[indices, :]
                    tmpx = np.true_divide(np.sum(np.multiply(Xj,np.conj(Xj)),axis=0), K-1)
                    tmpy = np.true_divide(np.sum(np.multiply(Yj,np.conj(Yj)),axis=0), K-1)

                    jcohTemp = np.sum(np.multiply(Xj, np.conj(Yj)), axis=0)
                    jcoh[ik, :] = np.arctanh(np.true_divide(np.abs(np.true_divide(jcohTemp,(K - 1))), np.sqrt(np.multiply(tmpx,tmpy)))).real
                    jXlsp[ik, :] = np.log(tmpx.real)
                    jYlsp[ik, :] = np.log(tmpy.real)

                lsigX = np.multiply(np.sqrt(K - 1), np.std(jXlsp, axis=0))
                lsigY = np.multiply(np.sqrt(K - 1), np.std(jYlsp, axis=0))
                lsigXY = np.multiply(np.sqrt(K - 1), np.std(jcoh, axis=0))
                crit = t.ppf(1 - np.true_divide(pval,2), K - 1) # Determine the scaling factor
                coh_err[0, ch, :] = np.tanh(np.arctanh(np.abs(coh)) + np.multiply(crit,  lsigXY))
                coh_err[1, ch, :] = np.tanh(np.arctanh(np.abs(coh)) - np.multiply(crit, lsigXY))
                SX_err[0, ch, :] = np.exp(np.log(S_X) + np.multiply(crit, lsigX))
                SX_err[1, ch, :] = np.exp(np.log(S_X) - np.multiply(crit, lsigX))
                SY_err[0, ch, :] = np.exp(np.log(S_Y) + np.multiply(crit, lsigY))
                SY_err[1, ch, :] = np.exp(np.log(S_Y) - np.multiply(crit, lsigY))

    if flag == 1:  # Pooling across trials
        Xk = np.zeros([nch * K, Nf], dtype=np.complex)
        Yk = np.zeros([nch * K, Nf], dtype=np.complex)
        if not contflag:
            mX = np.transpose(np.true_divide(np.sum(X, axis=0), nch))
            mY = np.transpose(np.true_divide(np.sum(Y, axis=0), nch))

        for ch in range(nch):
            if contflag:
                tmp1 = np.transpose(X[ch, :]) - np.true_divide(np.sum(X[ch, :]), N)
                tmp2 = np.transpose(Y[ch, :]) - np.true_divide(np.sum(Y[ch, :]), N)
            else:
                tmp1 = np.transpose(X[ch, :]) - mX
                tmp2 = np.transpose(Y[ch, :]) - mY

            extendedArrayx = extendArrayWithCurrentData(tmp1, ch, K)
            inputArrayx = np.multiply(tapers[:, 0:K], extendedArrayx)
            xk = np.fft.fft(np.transpose(inputArrayx), int(nf))
            Xk[int(ch * K):int((ch+1) * K), :] = xk[:, int(nfk[0]): int(nfk[1])]

            extendedArrayy = extendArrayWithCurrentData(tmp2, ch, K)
            inputArrayy = np.multiply(tapers[:, 0:K], extendedArrayy)
            yk = np.fft.fft(np.transpose(inputArrayy), int(nf))
            Yk[int(ch * K): int((ch+1) * K), :] = yk[:, int(nfk[0]): int(nfk[1])]

        S_X = np.true_divide(np.sum(np.multiply(Xk,np.conj(Xk)), axis=0), K)
        S_Y = np.true_divide(np.sum(np.multiply(Yk,np.conj(Yk)), axis=0), K)
        cohTemp = np.sqrt(np.multiply(S_X, S_Y))

        coh = np.true_divide(np.true_divide(np.sum(np.multiply(Xk, np.conj(Yk)), axis=0), K), cohTemp)

        if errorchk:  # Estimate error bars using Jacknife
            jcoh = np.zeros([nch * K, Nf])
            jXlsp = np.zeros([nch * K, Nf])
            jYlsp = np.zeros([nch * K, Nf])
            coh_err = np.zeros([2, Nf])
            SX_err = np.zeros([2, Nf])
            SY_err = np.zeros([2, Nf])
            for ik in range(nch*K):

                indices = np.setdiff1d(np.multiply(range(0, nch), K), ik)
                Xj = Xk[indices, :]
                Yj = Yk[indices, :]

                tx = np.true_divide(np.sum(np.multiply(Xj, np.conj(Xj)), axis=0), (nch * K - 1))
                ty = np.true_divide(np.sum(np.multiply(Yj, np.conj(Yj)), axis=0), (nch * K - 1))


                # Use atanh variance stabilizing transformation for coherence
                jcohTemp = np.true_divide(np.sum(np.multiply(Xj,np.conj(Yj)), axis=0), (nch * K - 1))
                jcohTemp1 = np.true_divide(jcohTemp, np.sqrt(np.multiply(tx, ty)))
                jcoh[ik, :] = np.arctanh(np.abs(jcohTemp1))
                jXlsp[ik, :] = np.log(tx.real)
                jYlsp[ik, :] = np.log(ty.real)

            lsigX = np.multiply(np.sqrt(nch * K - 1), np.std(jXlsp, axis=0))
            lsigY = np.multiply(np.sqrt(nch * K - 1), np.std(jYlsp, axis=0))
            lsigXY = np.multiply(np.sqrt(nch * K - 1), np.std(jcoh, axis=0))
            crit = t.ppf(1 - np.true_divide(pval, 2), K * nch - 1)  # Determine the scaling factor

            coh_err[0, :] = np.tanh(np.arctanh(np.abs(coh)) + np.multiply(crit, lsigXY))
            coh_err[1, :] = np.tanh(np.arctanh(np.abs(coh)) - np.multiply(crit, lsigXY))
            SX_err[0, :] = np.exp(np.log(S_X.real) + np.multiply(crit, lsigX))
            SX_err[1, :] = np.exp(np.log(S_X.real) - np.multiply(crit, lsigX))
            SY_err[0, :] = np.exp(np.log(S_Y.real) + np.multiply(crit, lsigY))
            SY_err[1, :] = np.exp(np.log(S_Y.real) - np.multiply(crit, lsigY))

    if flag == 11:  # Pooling across trials saving memory
        S_X = np.zeros([1, Nf])
        S_Y = np.zeros([1, Nf])
        coh = np.zeros([1, Nf])
        if not contflag:
            mX = np.transpose(np.true_divide(np.sum(X, axis=0), nch))
            mY = np.transpose(np.true_divide(np.sum(Y, axis=0), nch))

        for ch in range(nch):
            if contflag:
                tmp1 = np.transpose(X[ch, :]) - np.true_divide(np.sum(X[ch, :], axis=0), N)
                tmp2 = np.transpose(Y[ch, :]) - np.true_divide(np.sum(Y[ch, :], axis=0), N)
            else:
                tmp1 = np.transpose(X[ch, :]) - mX
                tmp2 = np.transpose(Y[ch, :]) - mY

            extendedArrayx = extendArrayWithCurrentData(tmp1, ch, K)
            inputArrayx = np.multiply(tapers[:, 0:K], extendedArrayx)
            Xk = np.fft.fft(np.transpose(inputArrayx), int(nf))

            extendedArrayy = extendArrayWithCurrentData(tmp2, ch, K)
            inputArrayy = np.multiply(tapers[:, 0:K], extendedArrayy)
            Yk = np.fft.fft(np.transpose(inputArrayy), int(nf))

            S_XTemp = Xk[:,int(nfk[0]):int(nfk[1])]
            S_XTemp2 = np.sum(np.multiply(S_XTemp, np.conj(S_XTemp)), axis=0)
            S_X = S_X + np.true_divide(np.true_divide(S_XTemp2, K), nch)
            S_X = S_X.real

            S_YTemp = Yk[:, int(nfk[0]) : int(nfk[1])]
            S_Y = S_Y + np.true_divide(np.true_divide(np.sum(np.multiply(S_YTemp,np.conj(S_YTemp)), axis = 0), K), nch)
            S_Y = S_Y.real

            coh = coh + np.true_divide(np.true_divide(np.sum(np.multiply(S_XTemp,np.conj(S_YTemp)), axis = 0), K), nch)

        coh = np.true_divide(coh, (np.sqrt(np.multiply(S_X, S_Y))))

    S_X = S_X.real
    S_Y = S_Y.real

    return coh, f, S_X, S_Y, coh_err, SX_err, SY_err
