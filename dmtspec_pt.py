

# Functionality Not Currently Supported ---------------------------------



import numpy as np
import math
import nextPowerOfTwo as np2
import dpsschk
from scipy.stats import t,chi2
from extendArrayWithCurrentData import extendArrayWithCurrentData

def dmtspec_pt(dN, tapers = None, sampling = 1, fk = None, pad = 2, pval = 0.05, flag = False, errorchk = False):

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
#
# Outputs: SPEC = Spectrum of dN in [Space / Trials, Freq] form.
# RATE = Mean rate of dN in Hz.
# F = Units of Frequency axis for SPEC.
    # ERR = Error bars for SPEC in [Hi / Lo, Space / Trials, Freq]
        # form given by a Jacknife - t interval for PVAL.
        #

    # Modification History: Rewritten by Bijan Pesaran 02 / 04 / 00
    # June 2004: Added cell array spike time inputs
    
    raise Exception('Functionality not currently supported')
    
    if tapers == None:
        tapers = [dN.shape[1], 3, 5]

    if np.size(tapers) == 2:
        n = tapers[0]
        w = tapers[1]
        p = n * w
        k = math.floor(2 *  - 1)
        tapers = [n, p, k]
        # disp(['Using ' num2str(k) ' tapers.'])

    if np.size(tapers) == 3:
        tapers[0] = math.floor(np.multiply(tapers[0], sampling))
        tapers, v = dpsschk.dpsschk(tapers)

    if fk is None:
        fk =[0, sampling/2]

    if np.size(fk) == 1:
        fk = [0, fk];

    nt = np.size(tapers[:,0])

    if isinstance(dN, dict):
        #dN = sp2ts(dN, [0, nt. / sampling, sampling]);
        raise Exception('Functionality not currently supported')

    nch = dN.shape[0]

    N = np.size(tapers[:, 0])

    if N is not nt:
        raise Exception('Length of time series and tapers must be equal');

    K = tapers.shape[1]
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

    if not flag: # No pooling across channels / trials

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

            tmp = np.transpose(dN[ch,:])
            fftInput = np.transpose(np.multiply(ntapers[:,0:K], extendArrayWithCurrentData(tmp,0,K)))

            dNk = np.transpose(np.fft.fft(fftInput, int(nf))) -tmp.mean(axis=0)
            dNk = np.multiply(np.transpose(dNk),H)
            lower = int(nfk[0])
            upper = int(nfk[1])
            dNk = dNk[:, lower:upper]
            Sk = np.multiply(np.abs(dNk),np.abs(dNk))
            spec[ch,:] = Sk.mean(axis=0)

            rateInput = np.multiply(extendArrayWithCurrentData(tmp,0,K),extendArrayWithCurrentData(tmp,0,K))
            rateInput = np.multiply(rateInput, ntapers[:, 0:K])
            rateInput = np.sum(rateInput,0)

            rate[ch] = np.mean(rateInput)

            if errorchk: # Estimate error bars using Jacknife
                jlsp = 0
                for ik in range(K):
                    indices = np.setdiff1d(np.arange(0, K), ik)
                    dNj = dNk[indices, :]
                    jlsp[ik,:] = np.log(np.mean(np.multiply(np.abs(dNj),np.abs(dNj)), 1))

                lsig = np.multiply(math.sqrt(K - 1), np.std(jlsp))
                crit = t.ppf(1 - np.true_divide(pval, 2),dof - 1)  # Determine the scaling factor, using student's t cdf inverse
                critlsig = np.multiply(crit, lsig)
                err[0, ch, :] = np.exp(math.log(spec[ch, :]) + critlsig)
                err[1, ch, :] = np.exp(math.log(spec[ch, :]) - critlsig)


    if flag:   # Pooling across trials
        spec = np.zeros(1, np.diff(nfk)[0])
        err = np.zeros(2, np.diff(nfk)[0])
        rate = 0

        dNk = np.zeros([nch * K, np.diff(nfk)[0]])

        for ch in range(nch):
            tmp = dN[ch,:]

            fftInput = np.transpose(np.multiply(ntapers[:, 0:K], extendArrayWithCurrentData(tmp, 0, K)))

            xk = np.transpose(np.fft.fft(fftInput, int(nf))) - tmp.mean(axis=0)
            xk = np.multiply(dNk, H)

            rateInput = np.multiply(extendArrayWithCurrentData(tmp, 0, K), extendArrayWithCurrentData(tmp, 0, K))
            rateInput = np.multiply(rateInput, ntapers[:, 0:K])
            rateInput = np.sum(rateInput, 0)
            newRate = np.mean(rateInput)

            rate = rate + newRate
            dNk[(ch-1)*K+1:ch*K,:] = xk[:, nfk[0]: nfk[1]]

        dNkSquared = np.multiply(np.abs(dNk),np.abs(dNk))
        spec = dNkSquared.mean(axis = 0)
        rate = np.true_divide(rate,nch)

        if errorchk: # Estimate error bars using Jacknife
            jlsp = 0
            for ik in range(nch * K):
                indices = np.setdiff1d(np.arange(0,K*nch) ,ik)
                dNj = dNj[indices,:]
                jlsp[ik,:] = np.log(spec)

            lsig = np.multiply(np.sqrt((nch * K) - 1), np.std(jlsp))
            crit = t.ppf(1 - np.true_divide(pval, 2), dof - 1)  # Determine the scaling factor
            critlsig = np.multiply(crit, lsig)
            err[0, :] = np.exp(math.log(spec) + critlsig)
            err[1, :] = np.exp(math.log(spec) - critlsig)

    return spec, rate, f, err
