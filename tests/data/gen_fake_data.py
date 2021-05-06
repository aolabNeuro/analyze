import numpy as np
import aopy.visualization as vis

filename = 'tests/data/fake ecube data/Headstages_8_Channels_int16_2021-05-06_11-47-02.bin'

# Generate a signal
samplerate = 25000
duration = 1
time = np.arange(0, duration, 1/samplerate)
frequency = 6
amplitude = 1
data = []
for i in range(8):
    theta = i*np.pi/8 # shift phase for each channel
    sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
    data.append(sinewave)
data = np.array(data).T

# Pack it into bits
voltsperbit = 1e-4
intdata = np.array(data/voltsperbit, dtype='<i2') # turn into integer data
flatdata = data.reshape(-1)
timestamp = [1, 2, 3, 4]
flatdata = np.insert(flatdata, timestamp, 0)

# Save it to the test file
with open(filename, 'wb') as f:
    f.write(np.byte(1))
    f.write(np.byte(1))
    f.write(np.byte(1))
    f.write(np.byte(1))
    f.write(np.byte(1))
    f.write(np.byte(1))
    f.write(np.byte(1))
    f.write(np.byte(1))
    for t in range(intdata.shape[0]):
        for ch in range(intdata.shape[1]):
            f.write(np.byte(intdata[t,ch]))
            f.write(np.byte(intdata[t,ch] >> 8))

# Load it again in the same way the whitematter code works
with open(filename, 'rb') as f:
    f.seek(8) # first uint64 is timestamp
    databuf = bytes(f.read())
    flatarray = np.frombuffer(databuf, dtype='<i2')
    shapedarray = flatarray.reshape(-1, 64).swapaxes(0,1)

# Save it as the ground truth figure
data = shapedarray.T
write_dir = 'tests/tmp'
figname = 'load_ecube_data_groundtruth.png'
vis.plot_timeseries(data, samplerate)
vis.savefig(write_dir, figname)