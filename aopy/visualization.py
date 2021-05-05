# visualization.py
# code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)

import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
import numpy as np
import os

def savefig(result_dir, filename, **kwargs):
    '''
    Wrapper around matplotlib savefig with some default options

    Args:
        result_dir (str): where to put the figure
        filename (str): what to name the figure
        **kwargs (optional): arguments to pass to plt.savefig()
    '''
    if '.' not in filename:
        filename += '.png'
    fname = os.path.join(result_dir, filename)
    if 'dpi' not in kwargs:
        kwargs['dpi']=300.
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
    time = np.arange(np.shape(data)[0])/samplerate
    for ch in range(np.shape(data)[1]):
        plt.plot(time, data[:,ch]*1e6)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')

def plot_freq_domain(freq_data, samplerate, ax=None):
    '''
    Plots data along frequency on the given axis

    Args:
        freq_data (nt, nch): frequency domain data, can also be a single channel vector
        samplerate (float): sampling rate of the data
        ax (pyplot axis, optional): where to plot
    '''
    if np.ndim(freq_data) < 2:
        freq_data = np.expand_dims(freq_data, 1)
    if ax is None:
        ax = plt.gca()
    length = np.shape(freq_data)[0]
    freq = np.fft.fftfreq(length, d=1./samplerate)
    data_ampl = abs(freq_data[freq>1,:])*2/length
    non_negative_freq = freq[freq>1]
    for ch in range(np.shape(freq_data)[1]):
        plt.semilogx(non_negative_freq, data_ampl[:,ch]*1e4)
        #plt.xscale('log', base=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (a.u.)')

def plot_data_on_pos(data, x_pos, y_pos, grid_size, ax=None, interp='nearest', cmap='bwr', threshold=0.01):
    '''
    Make an image out of scatter data. 

    Args:
        data (nch): list of values
        x_pos (nch): list of x positions
        y_pos (nch): list of y positions
        grid_size (2): number of points along each axis
        ax (int): optional axis on which to plot, default gca
        interp (str): method used for interpolation
        cmap (str): matplotlib colormap to use in image
        threshold (float): distance to neighbors before disregarding a point on the image
    '''
    extent = [np.min(x_pos), np.max(x_pos), np.min(y_pos), np.max(y_pos)]
    x_spacing = (extent[1]-extent[0])/(grid_size[0]-1)
    y_spacing = (extent[3]-extent[2])/(grid_size[1]-1)
    if ax is None:
        ax = plt.gca()
    xy = np.vstack((x_pos, y_pos)).T
    xq, yq = np.meshgrid(np.arange(extent[0],x_spacing*grid_size[0],x_spacing), np.arange(extent[2],y_spacing*grid_size[1],y_spacing))
    X = griddata(xy, data, (np.reshape(xq,-1), np.reshape(yq,-1)), method=interp, rescale=False)

    # Construct kd-tree, functionality copied from scipy.interpolate
    tree = cKDTree(xy)
    xi = _ndim_coords_from_arrays((np.reshape(xq,-1), np.reshape(yq,-1)))
    dists, indexes = tree.query(xi)

    # Copy original result but mask missing values with NaNs
    X[dists > threshold] = np.nan

    # Fix the extents to match what imshow expects
    extent = np.add(extent, [-x_spacing/2, x_spacing/2, -y_spacing/2, y_spacing/2])

    # Set the 'bad' color to something different
    cmap = matplotlib.cm.get_cmap(cmap).copy()
    cmap.set_bad(color='black')

    # Plot
    im = np.reshape(X, grid_size)
    plt.imshow(im, cmap=cmap, origin='lower', extent=extent)
    plt.xlabel('x position')
    plt.ylabel('y position')
