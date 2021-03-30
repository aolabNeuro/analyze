
import numpy as np
import math
import time
import gausswin


def tfwavelet(X, sampling=1, FREQPAR =None, waveflag = False):
    #
    # NOTE: Assumes X is in Python friendly readable form --> .DAT file needs to conform to Python dTypes

    # function[WAVEPAR, spec] = tfwavelet(X, sampling, FREQPAR, waveflag)

    # TFWavelet Moving window time - frequency decomposition using Morlet wavelets
    #
    # [WAVEPAR, SPEC] = TFWAVELET(X, FREQPAR, sampling, waveflag)
    #
    # INPUTS:
    # X | Time series array in [Trials, Time]form.
    # |
    # FREQPAR | Structure containing frequency transform parameters
    # | FREQPAR.foi: Frequencies of interest, defaults to 2. ** [1: 1 / 4:8]
    # | FREQPAR.bw: Frequency smoothing in fractional octaves, defaults to 0.5
    # | 0.5 octaves roughly correspond to foi / dfoi  not 5.83
    # | dfoi = 1 / timewin: higher smoothing, shorter time windows
    # | The size of the analysis window for the
    # | the distance of the first and last window centers to
    # | the start and end of the data for all frequencies
    # | FREQPAR.stepsize: Stepsize of the analysis windows in s, defaults to
    # | half overlapping windows for the highest frequency
    # | FREQPAR.win_centers: A vector of samples that specify the analysis window centers.
    # | Defaults to steps half of the size of the analysis window
    # | for the highest frequency
    # |
    # |
    # SAMPLING | Sampling rate of time series X in Hz.Defaults to 1
    # WAVELAG | If 1 spectral decomposition is not done and only WAVEPAR is returned
    #
    # OUTPUTS:
    # WAVEPAR | Structure containing wavelet parameters
    # | WAVEPAR.foi: Same as FREQPAR.foi
    # | WAVEPAR.bw: Same as FREQPAR.bw
    # | WAVEPAR.dfoi: Bandwidth around center frequencies
    # | WAVEPAR.timewin: Time window size for each frequency in s
    # | WAVEPAR.win_centers: Samples of analysis window centers, not returned with waveflag == 1
    # SPEC | Spectrum of X in [Trials, Time, Freq] form.
    #
    #
    # Origninal MATLAB Code .py adapted from:
    # Author: David Hawellek, version date April 23, 2013.
    #
    # Author: Seth Richards
    # Version Date: 2020/05/05

    X = np.asarray(X) #Makes certain array is Numpy array

    ntr = X.shape[0]  # number of trials / channels
    nti = X.shape[1]  # number of timepoints

    tic = 0

    if not False:
        tic = time.perf_counter()
        print('TFWAVELET: Data Transform\n')
    else :
        tic = time.perf_counter()
        print('TFWAVELET: Checking Parameters\n')

    if FREQPAR == None:
        FREQPAR = {'foi': None, 'bw': None, 'dfoi': None, 'stepsize': None, 'win_centers': None}
        FREQPAR['foi'] = np.power(2, np.arange(1, 8.25, 0.25))  # # Center frequencies
        FREQPAR['bw'] = 0.5  # Frequency resolution
        print('Setting foi/bw to default\n')

    if not 'foi' in FREQPAR:  # does foi appear in FREQPAR
        FREQPAR['foi'] = np.power(2, np.arange(1, 8.25, 0.25))
        print('Setting foi to default\n')

    if not 'bw' in FREQPAR:
        FREQPAR['bw'] = 0.5
        print('Setting bw to default\n')
    if not 'win_centers' in FREQPAR:
        FREQPAR['win_centers'] = []
        print('Setting window centers to default\n')

    print('#.2fHz to #.2fHz')
    # Parameters
    foi = FREQPAR['foi']
    bw = FREQPAR['bw']
    foi_min = (2 * foi) / (2 ** bw + 1)
    foi_max = 2 * foi / (2 ** -bw + 1)
    dfoi = foi_max - foi_min  # 2 * std in foi domain
    timewin = np.true_divide(6/math.pi, dfoi)

    timewin = np.around(timewin * 1000) / 1000
    toffset = timewin[0] / 2


    # Outpt structure
    WAVEPAR = {'foi': None, 'bw': None, 'dfoi': None, 'timewin': None}
    WAVEPAR['foi'] = foi
    WAVEPAR['bw'] = bw
    WAVEPAR['dfoi'] = dfoi
    WAVEPAR['timewin'] = timewin


    if waveflag:
        print('#.2fHz to #.2fHz with #.2f oct smoothing\n#.2fs largest to #.4fs smallest analysis window\n', foi[0],
              foi[-1], bw, timewin[0], timewin[-1])
        dospec = False
    else:
        dospec = True

    if dospec:
        if nti < np.multiply(timewin[0], sampling):
            print(nti)
            print(timewin(1))
            raise Exception(
                'At east #.2fs needed for analyzing #.2fHz with the specified parameters\nTrials just have #.2fs. Consider data padding.',
                timewin[0], foi[0], np.true_divide(nti, sampling))

        # Parameters in samples
        if not FREQPAR['win_centers']:
            tshift = timewin[-1] / 2
            nshift = round(tshift * sampling)
            noff = math.ceil(np.multiply(toffset, sampling))
            win_centers = np.arange(noff +1, nti + nshift, nshift- (noff + 1))   # win_centers = noff+1:nshift:nti-(noff+1) order of operations

            nsection = np.size(win_centers)
        else:
            win_centers = FREQPAR['win_centers']
            nsection = np.size(win_centers)

        # Update output parameters
        WAVEPAR['win_centers'] = win_centers

        # Memory allocation
        x = ntr
        y = nsection
        z = np.size(foi)

        #creates empty array to be populated later
        spec = np.empty((x, y, z), dtype = np.complex)
        spec[:] = np.nan
        # spec = complex(nan(ntr.shape[1], nsection.shape[1], np.size(foi), 'single'))     I have no idea what this is

        for (ifoi) in range(np.size(foi)):
            # Define frequency kernel
            n_win = round(timewin[ifoi] * sampling)
            TAPER = np.transpose(gausswin.gausswin(n_win, 3))
            TAPER = np.true_divide(TAPER,TAPER.sum(axis=0))

            iEXP = np.true_divide(((np.arange(1, n_win + 1) - n_win / 2 - 0.5)* foi[ifoi] * 2 * math.pi),(sampling ))
            iEXP = iEXP.astype('complex')

            len_iEXP = (len(iEXP))
            i = 0
            while i < len_iEXP:
                iEXP[i] = iEXP[i] * 1j
                i = i + 1

            iEXP = np.exp(iEXP)

            KERNEL = np.transpose(np.multiply(TAPER, iEXP))

            seccount = 0
            for isection in win_centers:

                colTwo = np.arange(isection - math.floor(n_win / 2), isection + math.ceil(n_win / 2) - 1 + 1)
                colTwo = int(colTwo[0])
                section = X[:,colTwo]
                seccount = seccount + 1

                spec[:,seccount,ifoi] = np.multiply(section,KERNEL)

    toc = time.perf_counter()
    timeElapsed = toc - tic
    print('done (#.2fs)\n', timeElapsed)

    return WAVEPAR, spec
