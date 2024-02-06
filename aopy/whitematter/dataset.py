import os
import glob
import re
import psutil
import numpy as np
from .arraymethods import *

class Dataset:

    def __init__(self, path=None, source=None, algs=None, datatype=None, maxchunksize=None, samplerate=None, verbose=False):
        self.verbose = False
        self.path = "."
        self.source = None
        self.datatype = "bin"
        self.availsources = set()
        self.maxchunksize = int(psutil.virtual_memory()[1]/10)
        self.samplerate = 25000
        self.sampleodometer = 0

        self.nextchunk = []

        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path

        if datatype is not None:
            self.datatype = datatype

        if samplerate is not None:
            self.samplerate = samplerate

        if self.datatype == "bin":
            self.__init_bin__()

        if maxchunksize is not None:
            self.maxchunksize = maxchunksize

        if verbose is True:
            self.verbose = verbose
            print("> Dataset initialized in {} with available sources: ".format(self.path))
            for asource in self.listsources():
                print("    {}".format(asource))
            print("  Reading {} bytes at a time.".format(self.maxchunksize))

        if source is not None:
            self.selectsource(source, verbose=verbose)

    def __init_bin__(self):
        self.re_bin_source  =   re.compile(r'.*[/\\](\w{,64})_\d{,4}_[cC]hannels_\w{1,5}\d{0,2}_.*')
        self.re_bin_attr    =   re.compile(r'.*[/\\]\w{,64}_(\d{,4})_[cC]hannels_(\w{1,5}\d{0,2})_.*')

        self.re_hsw_source  =   re.compile(r'(HSW)_(?:(?:\d{2,4}\.|\d{2,4})*_)*(?:\d+(?:hr|hour|min|sec))*_\d{,5}ch(?:_\d{,3}k+[sS]ps)?.*')
        self.re_hsw_attr    =   re.compile(r'HSW_(?:(?:\d{2,4}\.|\d{2,4})*_)*(?:\d+(?:hr|hour|min|sec))*_(\d{,5})ch_(\d{,3})(k+)[sS]ps.*')

    def __verboseprint__(self, overrideverbose):
        if overrideverbose is True or (overrideverbose is None and self.verbose is True):
            return True
        else:
            return False

    ### Listing data files

    def listdata(self, overridepath=None, overridepattern=None):
        return self.__binlistdata__(overridepath, overridepattern)

    def __binlistdata__(self, overridepath, overridepattern):
        searchpath = self.path
        searchpattern = "*." + self.datatype

        if overridepath is not None:
            searchpath = overridepath
        if overridepattern is not None:
            searchpattern = overridepattern
        elif self.source is not None:
            searchpattern = "*" + self.source + searchpattern

        searchpattern = os.path.join(searchpath, searchpattern)
        binfiles = glob.glob(searchpattern)
        self.filelist = sorted(binfiles)
        return self.filelist

    ### Listing sources

    def listsources(self, overridepath=None):
        return self.__binlistsources__(overridepath)

    def __binlistsources__(self, overridepath):
        path = self.path
        if overridepath is not None:
            path = overridepath

        binlist = self.__binlistdata__(path, overridepattern="*.*")
        for files in binlist:
            sourcesfound = self.re_bin_source.findall(files)
            hswfound = self.re_hsw_source.findall(files)
            self.availsources.update(sourcesfound)
            self.availsources.update(hswfound)
        return sorted(self.availsources)

    def selectsource(self, source=None, verbose=None):
        # Just in case selectsource() is called before __init__ for some reason
        if len(self.availsources) < 1:
            self.listsources()
        self.nextchunk = []
        self.sampleodometer = 0

        if source is not None:
            if source in self.availsources:
                self.source = source
                if self.__verboseprint__(verbose):
                    print("> Selected source: {}".format(source))
            else:
                raise LookupError('Source is not a member of available sources.')
        else:
            self.source = None
            print("> Selecting all sources.")

    ### Reading individual files

    def listrecordings(self, verbose=None):
        filelist = self.listdata()
        recordings = []

        sumbytes = 0
        prevfile = tuple()

        if len(filelist) == 0:
            raise IOError('Cannot find any files in {}.'.format(self.path))

        for fileattr in (self.parseattr(file) for file in filelist):
            if len(prevfile) == 0 \
            or (prevfile[0] == fileattr[0] \
            and prevfile[1] == fileattr[1] \
            and prevfile[2] < fileattr[2]):
                sumbytes += fileattr[3]
                prevfile = fileattr
            else:
                recordings.append((prevfile[0], prevfile[1], sumbytes))
                sumbytes = fileattr[3]
                prevfile = fileattr
        recordings.append((fileattr[0], fileattr[1], sumbytes))

        if self.__verboseprint__(verbose):
            print("> {} selected recording{} in {}:".format(\
                len(recordings),\
                "(s)" if len(recordings) > 1 else "",\
                self.path))
            for recording in recordings:
                print("    {0:4d} channels of {1}:\t{2:} samples ({3:8.3f} s)".format(recording[1], recording[0], recording[2], recording[2]/self.samplerate))
        return recordings

    def parseattr(self, filename):
        return self.__binparseattr__(filename)

    def __binparseattr__(self, filename):

        # find source type
        binsource = self.re_bin_source.findall(filename)
        if len(binsource) < 1:
            binsource = self.re_hsw_source.findall(filename)
            if len(binsource) < 1:
                raise LookupError('Cannot parse data source type.')

        # find source attributes: if even matches
        binattr = []
        if binsource[0] == "HSW":
            binattr = self.re_hsw_attr.findall(filename)
        else:
            binattr = self.re_bin_attr.findall(filename)
        
        if len(binattr) < 1:
            raise LookupError('Cannot parse binary file channel count and datatype from filename {}.'.format(filename))

        # find source attributes: convert types
        if binsource[0] == "HSW":
            samplerate = 0
            try:
                chancount = int(binattr[0][0])
                dattype = parsedattype('uint16') # hard coded for HSW
                samplerate = int(binattr[0][1])
                if len(binattr[0]) > 2 and binattr[0][2] == 'k':
                    samplerate = samplerate * 1000
                if samplerate > 0:
                    self.samplerate = samplerate
            except:
                raise LookupError('Cannot parse binary file channel count and datatype from filename {}.'.format(filename))
        else:
            try:
                chancount = int(binattr[0][0])
                dattype = parsedattype(binattr[0][1])
            except:
                raise LookupError('Cannot parse binary file channel count and datatype from filename {}.'.format(filename))

        if (binsource[0] == "DigitalPanel" and dattype[1] == 64):
            chancount = 1

        if (chancount * dattype[1]) % 8 != 0:
            raise LookupError('Invalid binary file channel count / bit depth combination parsed in {}, incomplete bytes.'.format(filename))

        rowsize = int(chancount * dattype[1] / 8)
        datasize = os.stat(filename).st_size - 8

        if (datasize) % rowsize != 0:
            print("Warning: incomplete binary file samples detected in {}. Check for file corruption.".format(filename))

        timestamp = binopentimestamp(filename)
        numsamples = int(datasize / rowsize)

        return (binsource[0], chancount, timestamp, numsamples, rowsize, datasize, dattype[0])

    def __binnextchunk__(self, filelist):
        fileindex = 0
        fileoffset = 0
        startofset = True
        lastrecording = tuple() # (source, chan, timestamp)

        def isnewrecording(currfileattr, lastfileattr):
            # if name of source, channelcount differs, or if timestamp restarts,
            # determine it's a new recording
            if currfileattr[0] != lastfileattr[0] \
            or currfileattr[1] != lastfileattr[1] \
            or currfileattr[2] < lastfileattr[2]:
                return True
            else:
                return False

        while fileindex < len(filelist):
            fileattr = self.parseattr(filelist[fileindex])
            chunksizetarget = self.maxchunksize - (self.maxchunksize % fileattr[4])
            while chunksizetarget > 0 and fileindex < len(filelist):
                # break conditions:
                #   1. no more room remaining in memory chunk to read
                #   2. a new recording has started (see isnewrecording)
                #   3. end of filelist

                lastrecording = (fileattr[0], fileattr[1], fileattr[2])
                    # archive out to determine isnewrecording later

                chunksize = min(chunksizetarget, fileattr[5] - fileoffset)
                    # chunksize is the min of:
                    #   1. room remaining in memory chunk to read
                    #   2. bytes remaining at end of file

                if chunksize > 0:
                    self.nextchunk.append((fileindex, fileoffset, chunksize))
                    chunksizetarget -= chunksize
                    fileoffset += chunksize
                    # if there is in fact room in mem or file to read, add that
                    # then increment the offset for the next chunk

                if chunksizetarget <= 0:
                    break
                    # make this die preemptively before incrementing file if no more mem

                if fileattr[5] - fileoffset <= 0:
                    fileindex += 1
                    fileoffset = 0
                    # if end of file has been reached, go to next file
                    if fileindex < len(filelist):
                        # if there are still more files
                        fileattr = self.parseattr(filelist[fileindex])
                            # parse the next file
                        if isnewrecording(fileattr, lastrecording):
                            break
                            # if it's a new recording, end the chunk anyway even if there's more memory remaining
                    # else:
                    #   technically, this is fileindex >= len(filelist)
                    #   so this is 3rd break condition, but it's end of loop so it's not needed

            if len(self.nextchunk) > 0:
                chunklist = list(map(lambda chunkidx: (filelist[chunkidx[0]], chunkidx[1], chunkidx[2]), self.nextchunk))
                yield (chunklist, startofset)
                    # generator emits chunk list if there are chunks to emit. also, if it's a new recording
                prevchunk = self.nextchunk.pop()
                self.nextchunk.clear()
                fileindex = prevchunk[0]
                fileoffset = prevchunk[1] + prevchunk[2]
                    # save out the last of the chunk list for reference
                if isnewrecording(fileattr, lastrecording):
                    startofset = True
                else:
                    startofset = False

    def emitchunk(self, startat=None, debug=None):
        filelist = self.listdata()
        recordingnum = 0
        skipsamples = 0

        if startat is not None:
            sampletarget = startat*self.samplerate

        for chunkparams in self.__binnextchunk__(filelist):
            chunkidxlist = chunkparams[0]
            isstart = chunkparams[1]
            if isstart is True:
                recordingnum += 1

            if debug is True:
                chunksizesum = sum(list(map(lambda x: x[2], chunkidxlist)))
                print("> Emitting{}chunk of size {} bytes containing:".format(" starting " if isstart is True else " continuing ", chunksizesum))

            chunksamples = 0
            for chunkidx in chunkidxlist:
                filename = chunkidx[0]
                fileattr = self.parseattr(filename)
                offset = chunkidx[1]
                chunksize = chunkidx[2]
                rowsize = fileattr[4]

                chancount = fileattr[1]
                dattype = fileattr[6]

                if debug is True:
                    print("    {} [{}:{}]: {}ch x {}".format(\
                    os.path.basename(filename), \
                    offset, \
                    offset+chunksize, \
                    chancount, \
                    np.dtype(dattype)))

                numsamples = int(chunksize / rowsize)
                chunksamples += numsamples

            if startat is not None:
                if (self.sampleodometer + chunksamples) <= sampletarget:
                    self.sampleodometer += chunksamples
                    continue
                elif self.sampleodometer < sampletarget:
                    skipsamples = sampletarget - self.sampleodometer
                    print("sampletarget: {}, skipsamples: {}".format(sampletarget, skipsamples))
                    print("samples [{}:{}]:".format(self.sampleodometer+skipsamples, self.sampleodometer+chunksamples))
                    self.sampleodometer += chunksamples
                else:
                    skipsamples = 0
                    self.sampleodometer += chunksamples

            chunkbuffer = bytearray(b'').join(binbufferchunk(chunkidx[0], chunkidx[1], chunkidx[2]) for chunkidx in chunkidxlist)
            if dattype == '?':
                uint8array = np.frombuffer(chunkbuffer, dtype='<u1')
                flatarray = np.unpackbits(uint8array)
            else:
                flatarray = np.frombuffer(chunkbuffer, dtype=dattype)
            if len(flatarray) % chancount != 0:
                print(f"Warning: {len(flatarray) % chancount} samples dropped in {filename}.")     
                shapedarray = flatarray[:-(len(flatarray) % chancount)].reshape(-1, chancount).swapaxes(0,1)
            else:
                shapedarray = flatarray.reshape(-1, chancount).swapaxes(0,1)

            if startat is not None and skipsamples > 0:
                print(shapedarray.shape)
                shapedarray = shapedarray[:, skipsamples:]

            # construct metadata
            chunkmeta = {"isstart": isstart, "recordingnum": recordingnum-1, "channels": chancount, "samplerate": self.samplerate, "nptype": dattype}
            yield MetadataArray(shapedarray, metadata=chunkmeta)

def parsedattype(bindatstring):
    # for legacy naming reasons, 'int64' and 'u/int1' were used for type bool for each individual digital channel

    return {
    'int1': ('<u8', 64),
    'uint1': ('<u8', 64),
    'int64': ('<u8', 64),
    'uint8': ('<u1', 8),
    'uint16': ('<u2', 16),
    'uint32': ('<u4', 32),
    'uint64': ('<u8', 64),
    'int8': ('<i1', 8),
    'int16': ('<i2', 16),
    'int32': ('<i3', 32),
    'float32': ('f', 32),
    'single32': ('f', 32),
    'bool': ('<u8', 64)
    }[bindatstring]

def binopentimestamp(filename):
    with open(filename, 'rb') as f:
        tsbytes = f.read(8)
        timestamp = int.from_bytes(tsbytes, byteorder='little', signed=False)
    return timestamp

def binbufferchunk(filename, offset, chunksize):
    with open(filename, 'rb') as f:
        f.seek(8 + offset) # first uint64 is timestamp
        databuf = bytes(f.read(chunksize))
        if len(databuf) != chunksize:
            raise IndexError('EOF hit: Chunksize out of range of filesize')
    return databuf