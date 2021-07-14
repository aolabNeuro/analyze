# utils.py
# all extra utility functions belong here
import numpy as np
import os
import pickle

def generate_test_signal(T, fs, freq, a):
    '''
    Generates a test time series signal with multiple frequencies, specified in freq, for T timelength at a sampling rate of fs

    Args:
        T (float): Time period in seconds
        fs (int): Sampling frequency in Hz
        freq (1D array): Frequencies to be mixed in the test signal. main frequency in the first element
        a (1D array) : amplitudes for each frequencies and last element of the array to be amplitude of noise (size : len(freq) + 1)

    Returns:
        x (1D array): cosine wave with multiple frequencies and noise
        t (1D array): time vector for x
    '''
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    # a = 0.02
    f0 = freq[0]
    # noise_power = 0.001 * fs / 2
    x = a[-1] * np.sin(2 * np.pi * 1.2 * np.sqrt(t))  # noise
    # x += np.random.normal(scale=np.sqrt(noise_power), size=t.shape)  # noise

    for i in range(len(freq)):
        x += a[i] * np.cos(2 * np.pi * freq[i] * t)

    return x, t


def pkl_write(file_to_write, values_to_dump, write_dir):
    '''
    This functions write data into a pickle file.
    
    Args:
        file_to_write (str): filename with '.pkl' extension
        values_to_dump (any): values to write in a pickle file
        write_dir (str): Path - where do you want to write this file

    Returns:
        None

    examples: pkl_write(meta.pkl, data, '/data_dir')
    '''
    file = os.path.join(write_dir, file_to_write)
    with open(file, 'wb') as pickle_file:
        pickle.dump(values_to_dump, pickle_file)


def pkl_read(file_to_read, read_dir):
    '''
    This function takes in path to a pickle file and returns data as it is stored
    
    Args:
        file_to_read (str): filename with '.pkl' extension
        read_dir (str): Path to folder where the file is stored

    Returns:
        data in a format as it is stored

    '''
    file = os.path.join(read_dir, file_to_read)
    this_dat = pickle.load(open(file, "rb"))
    return this_dat
