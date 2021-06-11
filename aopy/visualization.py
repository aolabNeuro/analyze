# visualization.py
# code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
import numpy as np
import os
import copy

from aopy import precondition
from scipy.signal import freqz


def savefig(base_dir, filename, **kwargs):
    '''
    Wrapper around matplotlib savefig with some default options
    Args:
        base_dir (str): where to put the figure
        filename (str): what to name the figure
        **kwargs (optional): arguments to pass to plt.savefig()
    '''
    if '.' not in filename:
        filename += '.png'
    fname = os.path.join(base_dir, filename)
    if 'dpi' not in kwargs:
        kwargs['dpi'] = 300.
    if 'facecolor' not in kwargs:
        kwargs['facecolor'] = 'none'
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'none'
    if 'transparent' not in kwargs:
        kwargs['transparent'] = True
    plt.savefig(fname, **kwargs)


def plot_timeseries(data, samplerate, ax=None):
    '''
    Plots data along time on the given axis
    Args:
        data (nt, nch): timeseries data, can also be a single channel vector
        samplerate (float): sampling rate of the data
        ax (pyplot axis, optional): where to plot
    '''
    if np.ndim(data) < 2:
        data = np.expand_dims(data, 1)
    if ax is None:
        ax = plt.gca()
    time = np.arange(np.shape(data)[0]) / samplerate
    for ch in range(np.shape(data)[1]):
        ax.plot(time, data[:, ch] * 1e6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (uV)')


def plot_freq_domain_power(data, samplerate, ax=None):
    '''
    Plots a power spectrum of each channel on the given axis
    Args:
        data (nt, nch): timeseries data, can also be a single channel vector
        samplerate (float): sampling rate of the data
        ax (pyplot axis, optional): where to plot
    '''
    if np.ndim(data) < 2:
        data = np.expand_dims(data, 1)
    if ax is None:
        ax = plt.gca()
    freq_data = np.fft.fft(data, axis=0)
    length = np.shape(freq_data)[0]
    freq = np.fft.fftfreq(length, d=1. / samplerate)
    data_ampl = abs(freq_data[freq > 1, :]) * 2 / length
    non_negative_freq = freq[freq > 1]
    for ch in range(np.shape(freq_data)[1]):
        ax.semilogx(non_negative_freq, data_ampl[:, ch] * 1e4)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (a.u.)')


def get_data_map(data, x_pos, y_pos):
    '''
    Organizes data according to the given x and y positions
    Args:
        data (nch): list of values
        x_pos (nch): list of x positions
        y_pos (nch): list of y positions
    Returns:
        2,n array: map of the data on the given grid
    '''
    data = np.reshape(data, -1)

    X = np.unique(x_pos)
    Y = np.unique(y_pos)
    nX = len(X)
    nY = len(Y)

    # Order Y into rows and X into columns
    data_map = np.empty((nY, nX), dtype=data.dtype)
    data_map[:] = np.nan
    for data_idx in range(len(data)):
        xid = np.where(X == x_pos[data_idx])[0]
        yid = np.where(Y == y_pos[data_idx])[0]
        data_map[yid, xid] = data[data_idx]

    return data_map


def calc_data_map(data, x_pos, y_pos, grid_size, interp_method='nearest', threshold_dist=None):
    '''
    Turns scatter data into grid data by interpolating up to a given threshold distance.
    Args:
        data (nch): list of values
        x_pos (nch): list of x positions
        y_pos (nch): list of y positions
        grid_size (tuple): number of points along each axis
        interp_method (str): method used for interpolation
        threshold_dist (float): distance to neighbors before disregarding a point on the image
    Returns:
        2,n array: map of the data on the given grid
    '''
    extent = [np.min(x_pos), np.max(x_pos), np.min(y_pos), np.max(y_pos)]
    x_spacing = (extent[1] - extent[0]) / (grid_size[0] - 1)
    y_spacing = (extent[3] - extent[2]) / (grid_size[1] - 1)
    xy = np.vstack((x_pos, y_pos)).T
    xq, yq = np.meshgrid(np.arange(extent[0], x_spacing * grid_size[0], x_spacing),
                         np.arange(extent[2], y_spacing * grid_size[1], y_spacing))
    X = griddata(xy, data, (np.reshape(xq, -1), np.reshape(yq, -1)), method=interp_method, rescale=False)

    # Construct kd-tree, functionality copied from scipy.interpolate
    tree = cKDTree(xy)
    xi = _ndim_coords_from_arrays((np.reshape(xq, -1), np.reshape(yq, -1)))
    dists, indexes = tree.query(xi)

    # Mask values with distances over the threshold with NaNs
    if threshold_dist:
        X[dists > threshold_dist] = np.nan

    data_map = np.reshape(X, grid_size)
    return data_map


def plot_spatial_map(data_map, x, y, ax=None, cmap='bwr'):
    '''
    Wrapper around plt.imshow for spatial data
    Args:
        data_map (2,n array): map of x,y data
        x (list): list of x positions
        y (list): list of y positions
        ax (int, optional): axis on which to plot, default gca
        cmap (str, optional): matplotlib colormap to use in image
    '''
    # Calculate the proper extents
    extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
    x_spacing = (extent[1] - extent[0]) / (data_map.shape[0] - 1)
    y_spacing = (extent[3] - extent[2]) / (data_map.shape[1] - 1)
    extent = np.add(extent, [-x_spacing / 2, x_spacing / 2, -y_spacing / 2, y_spacing / 2])

    # Set the 'bad' color to something different
    cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
    cmap.set_bad(color='black')

    # Plot
    if ax is None:
        ax = plt.gca()
    ax.imshow(data_map, cmap=cmap, origin='lower', extent=extent)
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')


def plot_rastor(data, plot_cue, cue_bin, ax):
    '''
       Create a rastor plot of neural data

       Args:
           data (n_trials, n_neurons, n_timebins): neural spiking data (not spike count- must contain only 0 or 1) in the form of a three dimensional matrix
           plot_cue : If plot_cue is true, a vertical line showing when this event happens is plotted in the rastor plot
           cue_bin : time bin at which an event occurs. For example: Go Cue or Leave center
            ax: axis to plot rastor plot
       Returns:
           rastor plot in appropriate axis
    '''
    n_trial = np.shape(data)[0]
    n_neurons = np.shape(data)[1]
    n_bins = np.shape(data)[2]

    color_palette = sns.set_palette("Accent", n_neurons)
    for n in range(n_neurons):  # set this to 1 if we need rastor plot for only one neuron
        for tr in range(n_trial):
            for t in range(n_bins):
                if data[n, tr, t] == 1:
                    x1 = [tr, tr + 1]
                    x2 = [t, t]
                    ax.plot(x2, x1, color=color_palette(n))
    if plot_cue:
        ax.axvline(x=cue_bin, linewidth=2.5, color='r')


'''
Plots to test filter performance
'''


def plot_filtered_signal(t, x, x_filter, low, high):
    # Plotting noisy test signal and filtered signal
    plt.plot(t, x, label='Noisy signal')
    plt.plot(t, x_filter, label='Filtered signal')
    plt.xlabel('time (seconds)')
    # plt.hlines([-self.a, self.a], 0, self.T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='best')
    plt.show()


def plot_phase_locking(t, a, f0, x_filter):
    # Plotting filtered signal with original signal frequency
    x_f0 = a * 2 * np.cos(2 * np.pi * f0 * t)
    plt.plot(t, x_f0, label='Original signal (%g Hz)' % f0)
    plt.plot(t, x_filter, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.ylabel('amplitude')
    plt.title('Comparison of Original Vs Filtered Signal')
    plt.legend(loc='best')
    plt.show()


def plot_freq_response_vs_filter_order(x, lowcut, highcut, fs):
    # Plot the frequency response for a few different orders
    for order in [2, 3, 4, 5,
                  6]:  # trying  different order of butterworth to see the roll off around cut-off frequencies
        precondition.butterworth_filter_data(x, lowcut, highcut, fs, order=order)

        b, a = precondition.butterworth_params(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Comparison of Frequency response for Diff. Orders of Butterworth Filter')
    plt.show()


def plot_psd(x, x_filter, fs):
    # Plot power spectral density of the signal
    f, psd = precondition.get_psd_welch(x, fs)
    f_filtered, psd_filtered = precondition.get_psd_welch(x_filter, fs)
    plt.semilogy(f, psd, label='test signal')
    plt.semilogy(f_filtered, psd_filtered, label='filtered output')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.title('Power Spectral Density Comparison')
    plt.show()


def plot_db_spectral_estimate(freq, psd, psd_filter, labels):
    psd = 10 * np.log10(psd)
    psd_filter = 10 * np.log10(psd_filter)
    plt.figure()
    precondition.plot_spectral_estimate(freq, psd, (psd_filter,), elabels=(labels,))
    plt.show()
