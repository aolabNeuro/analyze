import numpy as np
import math
from dpsschk import dpsschk
from nextPowerOfTwo import nextPowerOfTwo
from dmtspec import dmtspec
from scipy import signal
from extendArrayWithCurrentData import extendArrayWithCurrentData

# for debugging
import matplotlib.pyplot as plt


def tfspec(X, tapers=None, sampling=1, errorchk=False, dn=None, fk=None, pad=2, pval=0.05, flag= False, contflag=False, Errorbar="Chi-squared"):

    # TFSPEC Moving window time - frequency spectrum using multitaper techniques.
    #
    # [SPEC, F, TI, ERR] = TFSPEC(X, TAPERS, SAMPLING, DN, FK, PAD, PVAL, FLAG, CONTFLAG, ERRORBAR)
    #
    # Inputs: X = Time series array in [Space / Trials, Time] form.
    # TAPERS = Data tapers in [K, TIME], [N, P, K] or [N, W] form.
    # [N, W] Form: N = duration of analysis window in s.
    # W = bandwidth of frequency smoothing in Hz.
    # Defaults to[N, 3, 5] where N is NT / 10
    # and NT is duration of X.
    #
    # SAMPLING = Sampling rate of time series X in Hz.
    # Defaults to1.
    # ERRORCHK = Estimate error bars - Not functional
    # DN = Window step.
    # Defaults to N. / 10
    # FK = Frequency range to return in Hz in
    # either[F1, F2] or [F2] form.
    # In[F2] form, F1 is set to 0.
    # Defaults to[0, SAMPLING / 2]
    # PAD = Padding factor for the FFT.
    # i.e.For  N = 500,  if PAD = 2, we pad the FFT
    # to 1024 points; if PAD = 4, we pad the FFT
    # to 2048 points.
    # Defaults to 2.
    # PVAL = P - value to calculate error bars for .
    # Defaults to 0.05 i.e. 95 # confidence.
    #
    # FLAG = 0:    calculate SPEC seperately for each channel / trial.
        # FLAG = 1:    calculate   SPEC  by  pooling  across   channels / trials.
    # CONTFLAG = 1; There is only a single continuous signal coming in.
    # ERRORBAR = error bar estimate method
    #
    # Outputs: SPEC = Spectrum of X in [Space / Trials, Time, Freq] form.
    # F = Units of Frequency axis for SPEC.
        # ERR = Error  bars in [Hi / Lo, Space, Time, Freq]
    # form given by the Jacknife - t or Chi2 interval for PVAL.
        # TI =
    #
    # See also DPSS, PSD, SPECGRAM.

    # Origninal MATLAB Code adapted from:
    # Written by: Bijan Pesaran, 15 / 10 / 98.
    # Modification History:
    # Optimized when not computing error bars.

    # Author: Seth Richards
    # Version Date: 2020/06/14

    #finds the dimensions of array X, rows by columns

    X = np.asarray(X)
    nt = X.shape[1]  # calculate the number of points
    nch = X.shape[0]  # calculate the number of channels

#     n = math.floor(np.true_divide(np.true_divide(nt,10), sampling))
    n = nt//sampling//10
    #array n equals array nt divided by 10 times sampling - default window size

    if tapers is None:
        tapers = np.array([n, 3, 5])

    if len(tapers) == 2:
        n = tapers[0]
        w = tapers[1]
        p = n * w
        k = math.floor(2 * p - 1)
        tapers = [n, p, k]
        print(['Using ', tapers, ' tapers.'])

    if len(tapers) == 3:
        tapers[0] = math.floor(tapers[0] *sampling)
        dpss_tapers,throwAway = dpsschk(tapers)

    if dn is None :
        dn = np.true_divide(n, 10)

    if fk is None:
        fk = [0, np.true_divide(sampling, 2.)]

    if np.size(fk) == 1:
        fk = [0, fk]
    
    K = dpss_tapers.shape[0] # number of tapers
    N = dpss_tapers.shape[1] # number of time points, probably shouldn't be overwrote

    if N > nt:
        raise ValueError('Error: Tapers are longer than time series')

    dn = math.floor(dn*sampling)
    nf = np.maximum(256, pad * 2 ** nextPowerOfTwo(N + 1))
    # temp = np.multiply(sampling, nf)
    # fk = np.true_divide(fk,sampling)
    nfk = np.floor(np.array(fk) * nf / sampling)

    nwin = np.floor(np.true_divide((nt - N), dn))  # calculate the number of windows
    nfr = np.int(np.diff(nfk)[0])
    f = np.linspace(fk[0],fk[1],nfr)
    
    if not flag:  # No pooling across trials
        spec = np.zeros((nch,int(nwin),nfr))
        err = 0  # errorchk nonfunctional, returns zero for error estimate

        if errorchk:
            errorchk = False
            print("This code was not implemented/removed in MATLAB version")

        else:  # Don't estimate error bars

            err = None
            for win in range(0, int(nwin)):
                # Here the optimized spectral loop starts.
                if contflag:
                    tmp = signal.detrend(X[:,dn*win:(dn*win+N)]).T

                    if tmp.shape[1] > N :# machine precision work-around? added by alo for weird behavior 181000020
                        tmp = tmp[1:N, :]

                else:
                    mX = X[:,dn*win:(dn*win+N)].mean(axis=0)

#                     extendedArray = extendArrayWithCurrentData(mX, 0, nch, False)
                    tmp = (X[:,dn*win:(dn*win+N)] - mX).T # N x nch
                
                
                # this can all be done in a single pass. Don't for-loop.
                lowerBound = int(nfk[0])
                upperBound = int(nfk[1])
                inputArray = np.einsum('ij,ik->ijk',tmp,dpss_tapers.T) # N x nch x k
                Xk = np.fft.fft(inputArray,axis=0,n=int(nf))
                Xk = Xk[lowerBound:upperBound,]
                XkSquare = (Xk * np.conj(Xk)).real
                spec[:,win,:] = XkSquare.mean(axis=-1).T
#                 for ch in range(nch):

#                     extendedArray = extendArrayWithCurrentData(tmp, ch, K)
#                     inputArray = np.multiply(dpss_tapers, extendedArray)
#                     inputArray = np.einsum('ij,ik->ijk',tmp,dpss_tapers) # N x nch x k

#                     Xk = np.fft.fft(np.transpose(inputArray), int(nf))
#                     Xk = np.transpose(Xk)

#                     Xk = Xk[lowerBound:upperBound, :]

#                     XkSquare = Xk * np.conj(Xk)
#                     XkSquare = XkSquare.real
#                     specSliceTemp = np.sum(XkSquare, axis=1)
#                     specSliceTemp = np.true_divide(specSliceTemp, K)

#                     spec[ch, win, :] = np.transpose(specSliceTemp)

            # The optimized loop ends here end

    else:  # Estimate error bars - this is not optimized
            # Broken
            print("This code was not implemented in MATLAB originally, not supported in python yet")
            # ftmp, dum, err_tmp = dmtspec(np.transpose(tmp), tapers, sampling, fk, pad, pval);
            # spec[ch, win,:] = ftmp;
            # err[0, ch, win,:] = err_tmp[0,:];
            # err[1, ch, win,:] = err_tmp[1,:];

    if flag:  # Pooling across trials
        spec = np.zeros([nch, int(nwin), int(np.diff(nfk)[0])])
        err = np.zeros(shape=(2, int(nwin), int(np.diff(nfk)[0])), dtype=float)

        # disp('Flag = 11')
        # ind = repmat([1:nch], K, 1); ind = ind(:)
        for win in range(nwin):
        # The optimized loop starts here
            if contflag:
                tmp = X[:, dn * (win - 1) + 1: dn * (win - 1) + N]
            else:
                mX = np.sum(X[:, dn * (win - 1) + 1: dn * (win - 1) + N + 1], axis=0)
                mX = np.true_divide(mX, nch)

                extendedArray = extendArrayWithCurrentData(mX, 0, nch, False)
                tmp = (X[:, dn * (win - 1) + 1:dn * (win - 1) + N + 1] - extendedArray)
                tmp = np.transpose(tmp)

            if not errorchk: # Don't estimate error bars
                SX = np.zeros([int(np.diff(nfk)[0]), 1])
                for ch in range(nch):
                    extendedArray = extendArrayWithCurrentData(tmp, ch, K)
                    inputArray = np.multiply(dpss_tapers, extendedArray)

                    Xk = np.fft.fft(np.transpose(inputArray), int(nf))
                    Xk = np.transpose(Xk)

                    lowerBound = int(nfk[0])
                    upperBound = int(nfk[1])
                    Xk = Xk[lowerBound:upperBound, :]

                    XkSquare = Xk * np.conj(Xk)
                    XkSquare = XkSquare.real
                    specSliceTemp = np.sum(XkSquare, axis=1)

                    SX = SX + specSliceTemp

                spec[win, :] = np.true_divide(np.transpose(SX), np.multiply(K, nch))

            # The optimized loop ends here
            else: # Estimate error bars - This is not optimized
                ftmp, dum, err_tmp = dmtspec(np.transpose(tmp), dpss_tapers, sampling, fk, pad, pval, flag, Errorbar)
                spec[win, :] = ftmp
                err = np.zeros([2,err_tmp[1].shape])
                err[0, win, :] = err_tmp[0, :]
                err[1, win, :] = err_tmp[1, :]
    
    ti = np.linspace(N / 2, nt - N / 2, np.int(nwin))

    if spec.shape[1] == 1 and (spec.shape[1]).shape[1] > 2:
        spec = spec.squeeze()

    return spec, f, ti, err
