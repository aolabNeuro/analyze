
import numpy as np
from minmax import minmax


def sp2ts(sp, bn=None, binwidth=1):
    #  SP2TS Converts spike times to time series.
    #
    #  TS = SP2TS(SP, BN, BINWIDTH) returns a time series with bin
    #  width, BINWIDTH, with boundaries, BN(1:2), when given a vector 
    #  of spike times, SP.  
    #       If omitted, BN defaults to [minmax(SP),1].
    # Spike times are in ms and 

    # Origninal MATLAB Code adapted from:
    # Bijan Pesaran

    # Author: Seth Richards
    # Version Date: 2020/06/11

    # THIS VERSION DOES NOT WORK WITH NON INTEGER BIN SIZE

    if bn is None:
        bn = [np.transpose(minmax(sp)), 1]

    if bn.shape[1] < 3:
        bn[2] = 1e3
    
    ts = []
    diffTemp = np.multiply(abs(bn[1] - bn[0]), bn[2])
    x = np.linspace(bn[0], bn[1], np.true_divide(diffTemp, binwidth+1))
    
    if isinstance(sp, dict):

        for tr in sp:
            historgamSet = np.true_divide(sp[tr], bn[2])

            # throwAway sets values of ts_tmp to bin size per bin
            ts_tmp,cthrowAway = np.histogram(historgamSet, np.int(x))
            ts1 = [0, ts_tmp[1:-2], 0]
            ts = np.array([ts, ts1])

    else:
        # throwAway sets values of ts_tmp to bin size per bin
        ts_tmp,throwAway = np.histogram(sp, x)
        ts = [0, ts_tmp[1:-2], 0]

    return ts
