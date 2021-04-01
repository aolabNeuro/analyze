import numpy as np
import types
from .arraymethods import *

class ChunkedStream:
    def __init__(self, chunkemitter=None):
        if chunkemitter is None:
            raise StopIteration('No dataset emitter source specified.')
        
        if not isinstance(chunkemitter, types.GeneratorType):
            raise TypeError('ChunkedStream must be initialized on chunk emitters.')

        self.chunkemitter = iter(chunkemitter)

    def __next__(self):
        return next(self.chunkemitter)

    def __iter__(self):
        return self

    def find_all_nonzero(self, chmask=None, discontchunklength=0.1, threshold=None, multiplier=None):

        for emitchunk in self:
            if chmask is not None:
                chunk = emitchunk[chmask, :]
            else:
                chunk = emitchunk

            disconts = np.nonzero(chunk != 0)

            print("{} discontinuities found in block".format(len(disconts[1])))
            yield(disconts[0], disconts[1], chunk[disconts])

    def find_unique_discont(self, chmask=None, discontchunklength=0.1, threshold=None, multiplier=None):
        discontlist = []
        discontchunklist = []
        windowsize = int(discontchunklength*25000/2)

        for emitchunk in self:
            if chmask is not None:
                chunk = emitchunk[chmask, :]
            else:
                chunk = emitchunk

            if chunk.metadata['isstart'] is True:
                # start of a sequence
                t_chunk_offset = 0
                disconts, state = find_discont(chunk, threshold=threshold, multiplier=multiplier)
                windowsize = int(discontchunklength*chunk.metadata['samplerate']/2)
            else:
                disconts, state = find_discont(chunk, threshold=threshold, multiplier=multiplier, contstate=state)

            chunkdisconts = unique_disconts(disconts)
            print("Discontinuities: {}".format(chunkdisconts) if len(chunkdisconts) > 0 else "... no discontinuities found.")
            if len(chunkdisconts) > 0:
                for pt in chunkdisconts:
                    # index within chunk (indexable to data)
                    pt_ch = pt[0]
                    pt_t = pt[1]

                    # index within whole sequence, across multiple chunks
                    # (cannot be used, for printing information only)
                    pt_print = (pt_ch, pt_t + t_chunk_offset)

                    t_from = max(0, pt_t - windowsize)
                    t_to = min(chunk.shape[1]-1, pt_t + windowsize)
                    ptchunk = chunk[pt_ch, t_from:t_to]
                    yield pt_print, ptchunk

            discontlist.extend(chunkdisconts)
            t_chunk_offset += chunk.shape[1]

        return discontlist

def find_discont(indata, threshold=None, multiplier=None, contstate=None):
    if threshold is None:
        threshold = 99
    if multiplier is None:
        multiplier = 2

    if contstate is not None:
        prevrow = contstate["data"]
        if prevrow.shape[0] == indata.shape[0] and prevrow.shape[1] == 1:
            data = np.hstack((prevrow, indata))
        else:
            raise TypeError('contstate["data"] must be [channels x 1]')

        prevthres = contstate["thresval"]
    else:
        data = indata.astype(int)
        prevthres = np.inf

    diffs = np.diff(data)
    thresval = np.percentile(abs(diffs), threshold)*multiplier
    thresval = min(thresval, prevthres)

    discont_tuples = np.nonzero(abs(diffs) > thresval)
    timesortidx = np.argsort(discont_tuples[1])

    retstate = {"data": data[:,-1:], "thresval": thresval}

    return (discont_tuples[0][timesortidx], discont_tuples[1][timesortidx], diffs[discont_tuples]), retstate

def unique_disconts(discont_tuples, dup_dt=500):
    uniq_discont = []
    discont_ch = discont_tuples[0]
    discont_t = discont_tuples[1]

    if len(discont_ch) != len(discont_t):
        raise IndexError('Number of channels and timepoints do not match')

    if len(discont_t) > 0:
        # find a list of time changes greater than dup_dt samples (250 samples==1ms)
        # collapse ch listing if <= dup_dt samples
        discont_changepts = np.nonzero(np.diff(discont_t) > dup_dt)[0]
        # 0 gets first element of of discont_t
        # discont_changepts + 1 gets element after change occurs in discont_t
        discont_t_list = np.hstack((0,discont_changepts+1))

        # iterate through discont_t_list
        for t_list_i in range(len(discont_t_list)):
            # t is the smallest continuous block of time
            t = discont_t[discont_t_list[t_list_i]]

            # if there are more discont_t_list elements after this one, current t captures
            # all ch between this one and the next one
            if t_list_i < len(discont_t_list)-1:
                ch = np.unique(discont_ch[discont_t_list[t_list_i]:discont_t_list[t_list_i+1]])
            # if this is the last discont_t_list element (or the only one), capture
            # all the rest of the ch
            else:
                ch = np.unique(discont_ch[discont_t_list[t_list_i]:])

            # add to the list of discontinuities
            uniq_discont.append((ch, t))
    return uniq_discont