# visualization.py
# Code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)
import string
from typing import Tuple, Union
import warnings
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from scipy import signal
from scipy.stats import zscore
import numpy as np
import os
from PIL import Image
import copy
import pandas as pd
from tqdm import tqdm
import pandas as pd

from .. import analysis
from ..data import load_chmap
from .. import utils

def plot_mean_fr_per_target_direction(means_d, neuron_id, ax, color, this_alpha, this_label):
    '''
    generate a plot of mean firing rate per target direction

    '''
    sns.set_context('talk')
    ax.plot(np.array(means_d)[:, neuron_id], c=color, alpha=this_alpha, label=this_label)

    ax.legend()
    ax.set_xlabel("Target", fontsize=16)
    ax.set_ylabel("Spike Rate (Hz)", fontsize=16)
    plt.tight_layout()


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
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'none'
    if 'transparent' not in kwargs:
        kwargs['transparent'] = True
    if kwargs['transparent'] and 'facecolor' not in kwargs:
        kwargs['facecolor'] = 'none'
    plt.savefig(fname, **kwargs)

def subplots_with_labels(n_rows: int, n_cols: int, return_labeled_axes: bool = False,
                         rel_label_x: float = -0.25, rel_label_y: float = 1.1, label_font_size: int = 11,
                         constrained_layout: bool = True, **kwargs) -> Union[Tuple[plt.Figure, np.ndarray], Tuple[plt.Figure, np.ndarray, dict]]:
    """
    The goal is to augment plt.subplots() with the ability to label subplots with letters.
    Create a figure with subplots labeled with letters. 

    Example:
    generate a figure with 2 rows and 2 columns of subplots, labeled A, B, C, D
        .. code-block:: python
            fig, axes = subplots_with_labels(2, 2, constrained_layout=True)

        .. image:: _images/labeled_subplots.png

    Args:
    - n_rows: int, number of rows of subplots.
    - n_cols: int, number of columns of subplots.
    - return_labeled_axes: bool, whether to return the labeled axes.
    - rel_label_x: float, the relative x position of the subplot label.
    - rel_label_y: float, the relative y position of the subplot label.
    - label_font_size: int, the font size of the subplot label.
    - constrained_layout: bool, whether to use constrained layout.
    - **kwargs: additional keyword arguments to pass to plt.subplot_mosaic.

    Returns:
    - fig: Figure, the created figure.
    - axes: np.ndarray, the created axes.
    - labels_axes: dict, the labeled axes if return_labeled_axes is True.
    """
    # if more than 26 subplots, raise an error
    if n_rows * n_cols > 26:
        raise ValueError("More than 26 subplots requested, running out of single letters to label them with!")

    # make a list of letters to use as labels
    alphabets = string.ascii_uppercase
    labels = alphabets[:n_rows * n_cols]

    # tabulate the labels into n_rows by n_cols array
    labels = np.array(list(labels)).reshape((n_rows, n_cols))

    # make a string where rows are separated by semicolons
    labels = ";".join(["".join(row) for row in labels])

    # make the figure and axes
    fig, labels_axes = plt.subplot_mosaic(labels, constrained_layout=constrained_layout, **kwargs)

    for n, (key, ax) in enumerate(labels_axes.items()):
        ax.text(rel_label_x, rel_label_y, key, transform=ax.transAxes, size=label_font_size)

    # just annotate the axes
    axes = list(labels_axes.values())
    axes = np.array(axes).reshape((n_rows, n_cols))

    if return_labeled_axes:
        return fig, axes, labels_axes
    else:
        return fig, axes


def plot_timeseries(data, samplerate, ax=None, **kwargs):
    '''
    Plots data along time on the given axis. Default units are seconds and volts.

    Example:
        
        Plot 50 and 100 Hz sine wave.

        .. code-block:: python

            data = np.reshape(np.sin(np.pi*np.arange(1000)/10) + np.sin(2*np.pi*np.arange(1000)/10), (1000))
            samplerate = 1000
            plot_timeseries(data, samplerate)

        .. image:: _images/timeseries.png

    Args:
        data (nt, nch): timeseries data in volts, can also be a single channel vector
        samplerate (float): sampling rate of the data
        ax (pyplot axis, optional): where to plot
        kwargs (dict, optional): optional keyword arguments to pass to plt.plot
    '''
    if np.ndim(data) < 2:
        data = np.expand_dims(data, 1)
    if ax is None:
        ax = plt.gca()

    time = np.arange(np.shape(data)[0]) / samplerate
    for ch in range(np.shape(data)[1]):
        ax.plot(time, data[:, ch], **kwargs)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')

def gradient_timeseries(data, samplerate, n_colors=100, color_palette='viridis', ax=None, **kwargs):
    '''
    Draw gradient lines of timeseries data. Default units are seconds and volts.

    Args:
        data (nt, nch): timeseries to plot, can be 1d or 2d.
        samplerate (float): sampling rate of the data
        n_colors (int, optional): number of colors in the gradient. Default 100.
        color_palette (str, optional): colormap to use for the gradient. Default 'viridis'.
        ax (plt.Axis, optional): axis to plot the targets on
        kwargs (dict): keyword arguments to pass to the LineCollection function (similar to plt.plot)

    Raises:
        ValueError: if the data has more than 2 dimensions

    Example:
        .. code-block:: python

            data = np.reshape(np.sin(np.pi*np.arange(1000)/100), (1000))
            samplerate = 1000
            gradient_timeseries(data, samplerate)

        .. image:: _images/timeseries_gradient.png
    '''           
    if data.ndim == 1:
        data = np.expand_dims(data, 1)
    elif data.ndim > 2:
        raise ValueError('Data with more than 2 dimensions not supported!')
    if ax is None:
        ax = plt.gca()

    n_pt = data.shape[0]
    time = np.arange(n_pt) / samplerate
    colors = sns.color_palette(color_palette, min(n_colors, n_pt))

    # Segment the line
    labels = np.zeros((n_pt,), dtype='int')
    size = (n_pt // n_colors) * n_colors # largest size we can evenly split into n_colors
    labels[:size] = np.repeat(range(n_colors), n_pt // n_colors)
    labels[size:] = n_colors - 1 # leftovers also get the last color
    times, _ = utils.segment_array(time, labels, duplicate_endpoints=True)
    lines, line_labels = utils.segment_array(data, labels, duplicate_endpoints=True)

    # Use linecollections to plot each channel of data
    labels = np.array(line_labels).astype(int)
    colors = [colors[i] for i in labels]
    for dim in range(data.shape[1]):
        segments = [np.vstack([t, l[:,dim]]).T for t, l in zip(times, lines)]
        lc = LineCollection(segments, colors=colors, **kwargs)
        ax.add_collection(lc)
        
    ax.margins(0.05) # add_collections doesn't autoscale
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')

def plot_freq_domain_amplitude(data, samplerate, ax=None, rms=False):
    '''
    Plots a amplitude spectrum of each channel on the given axis. Just need to input time series
    data and this will calculate and plot the amplitude spectrum. 

    Example:

        Plot 50 and 100 Hz sine wave amplitude spectrum. 

        .. code-block:: python

            data = np.sin(np.pi*np.arange(1000)/10) + np.sin(2*np.pi*np.arange(1000)/10)
            samplerate = 1000
            plot_freq_domain_amplitude(data, samplerate) # Expect 100 and 50 Hz peaks at 1 V each

        .. image:: _images/freqdomain.png

    Args:
        data (nt, nch): timeseries data in volts, can also be a single channel vector
        samplerate (float): sampling rate of the data
        ax (pyplot axis, optional): where to plot
        rms (bool, optional): compute root-mean square amplitude instead of peak amplitude
    '''
    if ax is None:
        ax = plt.gca()
    non_negative_freq, data_ampl = analysis.calc_freq_domain_amplitude(data, samplerate, rms)
    for ch in range(np.shape(data_ampl)[1]):
        ax.semilogx(non_negative_freq, data_ampl[:,ch])
    ax.set_xlabel('Frequency (Hz)')
    if rms:
        ax.set_ylabel('RMS amplitude (V)')
    else:
        ax.set_ylabel('Peak amplitude (V)')

def get_data_map(data, x_pos, y_pos):
    '''
    Organizes data according to the given x and y positions

    Args:
        data (nch): list of values
        x_pos (nch): list of x positions
        y_pos (nch): list of y positions

    Returns:
        (m,n array): map of the data on the grid defined by x_pos and y_pos
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

    Example:
        Make a plot of a 10 x 10 grid of increasing values with some missing data.
        
        .. code-block:: python

            data = np.linspace(-1, 1, 100)
            x_pos, y_pos = np.meshgrid(np.arange(0.5,10.5),np.arange(0.5, 10.5))
            missing = [0, 5, 25]
            data_missing = np.delete(data, missing)
            x_missing = np.reshape(np.delete(x_pos, missing),-1)
            y_missing = np.reshape(np.delete(y_pos, missing),-1)

            interp_map, xy = calc_data_map(data_missing, x_missing, y_missing, [10, 10], threshold_dist=1.5)
            plot_spatial_map(interp_map, xy[0], xy[1])

        .. image:: _images/posmap_calcmap.png

    Args:
        data (nch): list of values
        x_pos (nch): list of x positions
        y_pos (nch): list of y positions
        grid_size (tuple): number of points along each axis
        interp_method (str): method used for interpolation
        threshold_dist (float): distance to neighbors before disregarding a point on the image

    Returns:
        tuple: tuple containing:
            | *data_map (grid_size array, e.g. (16,16)):* map of the data on the given grid
            | *xy (grid_size array, e.g. (16,16)):* new grid positions to use with this map

    '''
    extent = [np.min(x_pos), np.max(x_pos), np.min(y_pos), np.max(y_pos)]

    x_spacing = (extent[1] - extent[0]) / (grid_size[0] - 1)
    y_spacing = (extent[3] - extent[2]) / (grid_size[1] - 1)
    xy = np.vstack((x_pos, y_pos)).T
    xq, yq = np.meshgrid(np.arange(extent[0], extent[0] + x_spacing * grid_size[0], x_spacing),
                         np.arange(extent[2], extent[2] + y_spacing * grid_size[1], y_spacing))
    
    # Remove nan values
    non_nan = np.logical_not(np.isnan(data))
    data = data[non_nan]
    xy = xy[non_nan]
    
    # Interpolate
    new_xy = (np.reshape(xq, -1), np.reshape(yq, -1))
    X = griddata(xy, data, new_xy, method=interp_method, rescale=False)

    # Construct kd-tree, functionality copied from scipy.interpolate
    tree = cKDTree(xy)
    xi = _ndim_coords_from_arrays((np.reshape(xq, -1), np.reshape(yq, -1)))
    dists, indexes = tree.query(xi)

    # Mask values with distances over the threshold with NaNs
    if threshold_dist:
        X[dists > threshold_dist] = np.nan

    data_map = np.reshape(X, grid_size)
    return data_map, new_xy


def plot_spatial_map(data_map, x, y, alpha_map=None, ax=None, cmap='bwr', nan_color='black', clim=None):
    '''
    Wrapper around plt.imshow for spatial data

    Args:
        data_map ((2,n) array): map of x,y data
        x (list): list of x positions
        y (list): list of y positions
        alpha_map ((2,n) array): map of alpha values (optional, default alpha=1 everywhere). If the alpha
            values are outside of the range (0,1) they will be scaled automatically.
        ax (int, optional): axis on which to plot, default gca
        cmap (str, optional): matplotlib colormap to use in image. default 'bwr'
        nan_color (str, optional): color to plot nan values, or None to leave them invisible. default 'black'
        clim ((2,) tuple): (min, max) to set the c axis limits. default None, show the whole range

    Returns:
        mappable: image object which you can use to add colorbar, etc.

    Examples:
        
        Make a plot of a 10 x 10 grid of increasing values with some missing data.
        
        .. code-block:: python
        
            data = np.linspace(-1, 1, 100)
            x_pos, y_pos = np.meshgrid(np.arange(0.5,10.5),np.arange(0.5, 10.5))
            missing = [0, 5, 25]
            data_missing = np.delete(data, missing)
            x_missing = np.reshape(np.delete(x_pos, missing),-1)
            y_missing = np.reshape(np.delete(y_pos, missing),-1)

            data_map = get_data_map(data_missing, x_missing, y_missing)
            plot_spatial_map(data_map, x_missing, y_missing)

        .. image:: _images/posmap.png

        Make the same image but include a transparency layer

        .. code-block:: python

            data = np.linspace(-1, 1, 100)
            x_pos, y_pos = np.meshgrid(np.arange(0.5,10.5),np.arange(0.5, 10.5))
            missing = [0, 5, 25]
            data_missing = np.delete(data, missing)
            x_missing = np.reshape(np.delete(x_pos, missing),-1)
            y_missing = np.reshape(np.delete(y_pos, missing),-1)
            data_map = get_data_map(data_missing, x_missing, y_missing)
            plot_spatial_map(data_map, x_missing, y_missing, alpha_map=data_map)

        .. image:: _images/posmap_alphamap.png

    '''
    # Calculate the proper extents
    if data_map.size > 1:
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
        x_spacing = (extent[1] - extent[0]) / (data_map.shape[0] - 1)
        y_spacing = (extent[3] - extent[2]) / (data_map.shape[1] - 1)
        extent = np.add(extent, [-x_spacing / 2, x_spacing / 2, -y_spacing / 2, y_spacing / 2])
    else:
        extent = [np.min(x) - 0.5, np.max(x) + 0.5, np.min(y) - 0.5, np.max(y) + 0.5]

    # Set the 'bad' color to something different
    cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
    if nan_color:
        cmap.set_bad(color=nan_color)
    
    # If an alpha map is present, make an rgba image
    if alpha_map is not None:
        if clim is None:
            clim = (np.nanmin(data_map), np.nanmax(data_map))
        norm = cm.colors.Normalize(*clim)
        scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
        data_map = scalarMap.to_rgba(data_map)

        # Apply the alpha map after scaling from 0 to 1
        alpha_range = np.nanmax(alpha_map) - np.nanmin(alpha_map)
        if alpha_range > 1 or np.nanmax(alpha_map) > 1 or np.nanmin(alpha_map) < 0:
            alpha_map = (alpha_map - np.nanmin(alpha_map)) / alpha_range
        alpha_map[np.isnan(alpha_map)] = 0
        data_map[:,:,3] = alpha_map
        
    # Plot
    if ax is None:
        ax = plt.gca()
    image = ax.imshow(data_map, cmap=cmap, origin='lower', extent=extent)
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')

    return image

def plot_ECoG244_data_map(data, bad_elec=[], interp=True, cmap='bwr', theta=0, ax=None, **kwargs):
    '''
    Plot a spatial map of data from an ECoG244 electrode array from the Viventi lab.

    Args:
        data ((256,) array): values from the ECoG array to plot in 2D
        bad_elec (list, optional): channels to remove from the plot. Defaults to [].
        interp (bool, optional): flag to include 2D interpolation of the result. Defaults to True.
        cmap (str, optional): matplotlib colormap to use in image. Defaults to 'bwr'.
        theta (float): rotation (in degrees) to apply to positions. rotations are applied clockwise, 
            e.g., theta = 90 rotates the map clockwise by 90 degrees, -90 rotates the map anti-clockwise 
            by 90 degrees. Default 0.
        ax (pyplot.Axes, optional): axis on which to plot. Defaults to None.
        kwargs (dict): dictionary of additional keyword argument pairs to send to calc_data_map and plot_spatial_map.

    Returns:
        pyplot.Image: image returned by pyplot.imshow. Use to add colorbar, etc.

    Examples:

        .. code-block:: python

            data = np.linspace(-1, 1, 256)
            missing = [0, 5, 25]
            plt.figure()
            plot_ECoG244_data_map(data, bad_elec=missing, interp=False, cmap='bwr', ax=None)
            # Here the missing electrodes (in addition to the ones
            # undefined by the channel mapping) should be visible in the map.

            plt.figure()
            plot_ECoG244_data_map(data, bad_elec=missing, interp=False, cmap='bwr', ax=None, nan_color=None)
            # Now we make the missing electrodes transparent

            plt.figure()
            plot_ECoG244_data_map(data, bad_elec=missing, interp=True, cmap='bwr', ax=None)
            # Missing electrodes should be filled in with linear interp.

    '''
    
    if ax is None:
        ax = plt.gca()
    
    # Load the signal path files
    elec_pos, acq_ch, elecs = load_chmap(drive_type='ECoG244', theta=theta)

    # Remove bad electrodes
    bad_ch = acq_ch[np.isin(elecs, bad_elec)]-1
    data[bad_ch] = np.nan
        
    # Interpolate or directly compute the map
    if interp:
        interp_kwargs = {k: v for k, v in kwargs.items() if k in ['interp_method', 'threshold_dist']}
        data_map, xy = calc_data_map(data[acq_ch-1], elec_pos[:,0], elec_pos[:,1], (16, 16), **interp_kwargs)
    else:
        data_map = get_data_map(data[acq_ch-1], elec_pos[:,0], elec_pos[:,1])
        xy = [elec_pos[:,0], elec_pos[:,1]]

    # Plot
    plot_kwargs = {k: v for k, v in kwargs.items() if k in ['alpha_map', 'nan_color', 'clim']}
    im = plot_spatial_map(data_map, xy[0], xy[1], cmap=cmap, ax=ax, **plot_kwargs)
    return im

def annotate_spatial_map(elec_pos, text, color, fontsize=6, ax=None, **kwargs):
    '''
    Simple wrapper around plt.annotate() to add text annotation to a 2d position. 

    Args:
        elec_pos ((x,y) tuple): position where text should be placed on 2d plot
        text (str): annotation text
        color (plt.Color): the color to make the text
        fontsize (int, optional): the fontsize to make the text. Defaults to 6.
        ax (pyplot.Axes, optional): axis on which to plot. Defaults to None.
        kwargs (dict): additional keyword arguments to pass to plt.annotate()

    Returns:
        plt.Annotation: annotation object
    '''
    if ax is None:
        ax = plt.gca()
    return ax.annotate(text, elec_pos, color=color, fontsize=fontsize, ha='center', va='center', **kwargs)
    
def annotate_spatial_map_channels(acq_idx=None, acq_ch=None, drive_type='ECoG244', theta=0, color='k', fontsize=6, 
                                  ax=None, **kwargs):
    '''
    Given acq_idx (indices) or acq_ch (channel numbers), prints either indices or channel numbers
    on top of a spatial map.

    Args:
        acq_idx ((nacq,) array or list, optional): If provided, specifies the acquisition indices to
            be annotated. If neither acq_idx nor acq_ch is provided, all channel numbers will be 
            annotated by default.
        acq_ch ((nacq,) array or list, optional): If provided, specifies the acquisition channel numbers to
            be annotated. If neither acq_idx nor acq_ch is provided, all channel numbers will be 
            annotated by default.
        drive_type (str, optional): Drive type of the channels to plot. See :func:`aopy.data.base.load_chmap`.
        color (str, optional): color to display the channels. Default 'k'.
        fontsize (int, optional): the fontsize to make the text. Defaults to 6.
        print_zero_index (bool, optional): if True (the default), prints channel numbers indexed by 0. 
            Otherwise prints directly from the channel map (which should use 1-indexing).
        ax (pyplot.Axes, optional): axis on which to plot. Defaults to None.
        kwargs (dict): additional keyword arguments to pass to plt.annotate()

    Example:

        .. code-block:: python

            aopy.visualization.plot_ECoG244_data_map(np.zeros(256,), cmap='Greys')
            aopy.visualization.annotate_spatial_map_channels(drive_type='ECoG244', color='k')
            aopy.visualization.annotate_spatial_map_channels(drive_type='Opto32', color='b')
            plt.axis('off')

        .. image:: _images/ecog244_opto32.png

    Note: 
        The acq_ch returned from `func::aopy.data.load_chmap` are generally 1-indexed lists of acquisition 
        channels connected to electrodes. In python, however, the acquisition indices start at 0, so we
        give the option to select channels based on either an index (acq_idx) or a channel number (acq_ch).
    '''
    if ax is None:
        ax = plt.gca()
    if acq_idx is not None and acq_ch is not None:
        raise ValueError("Please specify only one of acq_idx or acq_ch.")
    if acq_idx is not None:
        acq_ch = np.array(acq_idx)+1 # Change from index to ch numbers
        print("Annotating acquisition indices")
    else:
        print("Annotating acquisition channel numbers")

    # Get channel map (overwrite acq_ch if it was supplied to get the correct shape acq_ch)
    elec_pos, acq_ch, elecs = load_chmap(drive_type, acq_ch, theta)

    # Annotate each channel
    if isinstance(color, str) or len(color) < len(elec_pos):
        color = np.repeat(np.array(color), len(elec_pos))
    for pos, ch, color in zip(elec_pos, acq_ch, color):
        if acq_idx is not None:
            ch = ch - 1 # change back from channel numbers to indices
        annotate_spatial_map(pos, ch, color, fontsize, ax, **kwargs)

def plot_image_by_time(time, image_values, ylabel='trial', cmap='bwr', ax=None):
    '''
    Makes an nt x ntrial image colored by the timeseries values. 

    Example:
        ::

            time = np.array([-2, -1, 0, 1, 2, 3])
            data = np.array([[0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0]]).T
            plot_image_by_time(time, data)
            filename = 'image_by_time.png'

        .. image:: _images/image_by_time.png

    Args:
        time (nt): time vector to plot along the x axis
        image_values (nt, [nch or ntr]): time-by-trial or time-by-channel data
        ylabel (str, optional): description of the second axis of image_values. Defaults to 'trial'.
        cmap (str, optional): colormap with which to display the image. Defaults to 'bwr'.
        ax (pyplot.Axes, optional): Axes object on which to plot. Defaults to None.

    Returns:
        pyplot.AxesImage: the image object returned by pyplot
    '''
    
    image_values = np.array(image_values)
    extent = [np.min(time), np.max(time), 0, image_values.shape[1]]

    # Plot
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(image_values.T, cmap=cmap, origin='lower', extent=extent, aspect='auto', \
        resample=False, interpolation='none', filternorm=False)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    
    return im


def plot_raster(data, cue_bin=None, ax=None):
    '''
    Create a raster plot for binary input data and show the relative timing of an event with a vertical red line

    .. image:: _images/raster_plot_example.png

    Args:
        data (ntime, ncolumns): 2D array of data. Typically a time series of spiking events across channels or trials (not spike count- must contain only 0 or 1).
        cue_bin (float): time bin at which an event occurs. Leave as 'None' to only plot data. For example: Use this to indicate 'Go Cue' or 'Leave center' timing.
        ax (plt.Axis): axis to plot raster plot
        
    Returns:
        None: raster plot plotted in appropriate axis
    '''
    if ax is None:
        ax = plt.gca()

    ax.eventplot(data.T, color='black')
    
    if cue_bin is not None:
        ax.axvline(x=cue_bin, linewidth=2.5, color='r')

def set_bounds(bounds, ax=None):
    '''
    Sets the x, y, and z limits according to the given bounds

    Args:
        bounds (tuple): 6-element tuple describing (-x, x, -y, y, -z, z) cursor bounds
        ax (plt.Axis, optional): axis to plot the targets on
    '''
    if ax is None:
        ax = plt.gca()

    try:
        ax.set(xlim=(1.1 * bounds[0], 1.1 * bounds[1]),
               ylim=(1.1 * bounds[2], 1.1 * bounds[3]),
               zlim=(1.1 * bounds[4], 1.1 * bounds[5]))
    except:
        ax.set(xlim=(1.1 * bounds[0], 1.1 * bounds[1]),
               ylim=(1.1 * bounds[2], 1.1 * bounds[3]))


def plot_targets(target_positions, target_radius, bounds=None, alpha=0.5, 
                 origin=(0, 0, 0), ax=None, unique_only=True):    
    '''
    Add targets to an axis. If any targets are at the origin, they will appear 
    in a different color (magenta). Works for 2D and 3D axes

    Example:
        Plot four peripheral and one central target.
        ::
        
            target_position = np.array([
                [0, 0, 0],
                [1, 1, 0],
                [-1, 1, 0],
                [1, -1, 0],
                [-1, -1, 0]
            ])
            target_radius = 0.1
            plot_targets(target_position, target_radius, (-2, 2, -2, 2))

        .. image:: _images/targets.png

    Args:
        target_positions (ntarg, 3): array of target (x, y, z) locations
        target_radius (float): radius of each target
        bounds (tuple, optional): 6-element tuple describing (-x, x, -y, y, -z, z) cursor bounds
        origin (tuple, optional): (x, y, z) position of the origin
        ax (plt.Axis, optional): axis to plot the targets on
        unique_only (bool, optional): If True, function will only plot targets with unique positions (default: True)
    '''

    if unique_only:
        target_positions = np.unique(target_positions,axis=0)

    if isinstance(alpha,float):
        alpha = alpha * np.ones(len(target_positions))
    else:
        assert len(alpha) == len(target_positions), "list of alpha values must be equal in length to the list of targets."

    if ax is None:
        ax = plt.gca()

    if unique_only:
        target_positions = np.unique(target_positions,axis=0)

    for i in range(len(target_positions)):

        # Pad the vector to make sure it is length 3
        pos = np.zeros((3,))
        pos[:len(target_positions[i])] = target_positions[i]

        # Color according to its position
        if (pos == origin).all():
            target_color = 'm'
        else:
            target_color = 'b'

        plot_circles([pos], target_radius, target_color, bounds, alpha[i], ax, unique_only=False)

def plot_circles(circle_positions, circle_radius, circle_color='b', bounds=None, alpha=0.5, ax=None, unique_only=True):    
    '''
    Add circles to an axis. Works for 2D and 3D axes

    Args:
        circle_positions (ntarg, 3): array of target (x, y, z) locations
        circle_radius (float): radius of each target
        circle_color (str): color to draw circle - default is blue
        bounds (tuple, optional): 6-element tuple describing (-x, x, -y, y, -z, z) cursor bounds
        origin (tuple, optional): (x, y, z) position of the origin
        ax (plt.Axis, optional): axis to plot the targets on
        unique_only (bool, optional): If True, function will only plot targets with unique positions (default: True)
    '''

    if unique_only:
        circle_positions = np.unique(circle_positions,axis=0)

    if isinstance(alpha,float):
        alpha = alpha * np.ones(len(circle_positions))
    else:
        assert len(alpha) == len(circle_positions), "list of alpha values must be equal in length to the list of cricles."

    if ax is None:
        ax = plt.gca()

    for i in range(0, len(circle_positions)):

        # Pad the vector to make sure it is length 3
        pos = np.zeros((3,))
        pos[:len(circle_positions[i])] = circle_positions[i]

        # Plot in 3D or 2D
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        try:
            ax.set_zlabel('z')
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = pos[0] + circle_radius * np.outer(np.cos(u), np.sin(v))
            y = pos[1] + circle_radius * np.outer(np.sin(u), np.sin(v))
            z = pos[2] + circle_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, alpha=alpha[i], color=circle_color)
            ax.set_box_aspect((1, 1, 1))
        except:
            target = plt.Circle((pos[0], pos[1]),
                                radius=circle_radius, alpha=alpha[i], color=circle_color)
            ax.add_artist(target)
            ax.set_aspect('equal', adjustable='box')
    if bounds is not None: set_bounds(bounds, ax)


def plot_trajectories(trajectories, bounds=None, ax=None, **kwargs):
    '''
    Draws the given trajectories, one at a time in different colors. Works for 2D and 3D axes

    Example:
        Two random trajectories.
        ::

            trajectories =[
                np.array([
                    [0, 0, 0],
                    [1, 1, 0],
                    [2, 2, 0],
                    [3, 3, 0],
                    [4, 2, 0]
                ]),
                np.array([
                    [-1, 1, 0],
                    [-2, 2, 0],
                    [-3, 3, 0],
                    [-3, 4, 0]
                ])
            ]
            bounds = (-5., 5., -5., 5., 0., 0.)
            plot_trajectories(trajectories, bounds)

        .. image:: _images/trajectories.png

    Args:
        trajectories (list): list of (n, 2) or (n, 3) trajectories where n can vary across each trajectory
        bounds (tuple, optional): 6-element tuple describing (-x, x, -y, y, -z, z) cursor bounds
        ax (plt.Axis, optional): axis to plot the targets on
        kwargs (dict): keyword arguments to pass to the plt.plot function
   '''
    if ax is None:
        ax = plt.gca()

    # Plot in 3D, fall back to 2D
    try:
        ax.set_zlabel('z')
        for path in trajectories:
            ax.plot(*path.T, **kwargs)
        ax.set_box_aspect((1, 1, 1))
    except:
        for path in trajectories:
            ax.plot(path[:, 0], path[:, 1], **kwargs)
        ax.set_aspect('equal', adjustable='box')

    if bounds is not None: set_bounds(bounds, ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def color_trajectories(trajectories, labels, colors, ax=None, **kwargs):
    '''
    Draws the given trajectories but with the color of each trajectory corresponding to its given label.
    Works for 2D and 3D axes

    Example:

        .. code-block:: python

            trajectories =[
                    np.array([
                        [0, 0, 0],
                        [1, 1, 0],
                        [2, 2, 0],
                        [3, 3, 0],
                        [4, 2, 0]
                    ]),
                    np.array([
                        [-1, 1, 0],
                        [-2, 2, 0],
                        [-3, 3, 0],
                        [-3, 4, 0]
                    ])
                ]
            labels = [0, 0, 1, 0]
            colors = ['r', 'b']
            color_trajectories(trajectories, labels, colors)

            .. image:: _images/color_trajectories_simple.png

            labels_list = [[0, 0, 1, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]]
            fig = plt.figure()
            color_trajectories(trajectories, labels_list, colors)

            .. image:: _images/color_trajectories_segmented.png

    Args:
        trajectories (ntrials): list of (n, 2) or (n, 3) trajectories where n can 
            vary across each trajectory
        labels (ntrials): integer array of labels for each trajectory. Basically an 
            index for each trajectory
        colors (ncolors): list of colors. A list of arrays containing the label for 
            each corresponding trajectory, or a list of lists where each sublist 
            corresponds to the label for each timepoint in the corresponding trajectory.

        ax (plt.Axis, optional): axis to plot the targets on
        **kwargs (dict): other arguments for plot_trajectories(), e.g. bounds
    '''

    # If the labels are in list of lists format, segment the trajectories accordingly
    if isinstance(labels[0], list) or isinstance(labels[0], np.ndarray):
        all_trajectories = []
        all_labels = []
        for t, l in zip(trajectories, labels):
            assert len(t) == len(l), "Input labels must be the same length as input trajectories"
            segmented_trajectories, segmented_labels = utils.segment_array(t, l, duplicate_endpoints=True)
            all_trajectories += segmented_trajectories
            all_labels += segmented_labels
        trajectories = all_trajectories
        labels = all_labels
    
    # Convert the labels to integers for indexing into the color list
    labels = np.array(labels).astype(int)

    # Initialize a cycler with the appropriate colors
    style = plt.cycler(color=[colors[i] for i in labels])
    if ax is None:
        ax = plt.gca()
    ax.set_prop_cycle(style)

    # Use the regular trajectory plotting function
    plot_trajectories(trajectories, ax=ax, **kwargs)

def gradient_trajectories(trajectories, n_colors=100, color_palette='viridis', bounds=None, ax=None, **kwargs):
    '''
    Draw trajectories with a gradient of color from start to end of each trajectory. 
    Works in 2D and 3D.

    Note: this function applies the gradient evenly across the timepoints of the trajectory. 
        It might be useful to use the sampling rate of the data instead of n_colors, so that
        the time axis is consistent across sampling rates. 

    Args:
        trajectories (ntrials): list of 2D or 3D trajectories, in x, y[, z] coordinates
        n_colors (int, optional): number of colors in the gradient. Default 100.
        color_palette (str, optional): colormap to use for the gradient. Default 'viridis'.
        bounds (tuple, optional): 6-element tuple describing (-x, x, -y, y, -z, z) axes bounds
        ax (plt.Axis, optional): axis to plot the targets on
        kwargs (dict): keyword arguments to pass to the LineCollection function (similar to plt.plot)

    Example:

        Cursor trajectories in 2D
        .. code-block:: python

            subject = 'beignet'
            te_id = 5974
            date = '2022-07-01'
            preproc_dir = data_dir
            traj, _ = aopy.data.get_kinematic_segments(preproc_dir, subject, te_id, date, [32], [81, 82, 83, 239])
            gradient_trajectories(traj[:3])
        
        .. image:: _images/gradient_trajectories.png

        Hand trajectories in 3D
        .. code-block:: python

            traj, _ = aopy.data.get_kinematic_segments(preproc_dir, subject, te_id, date, [32], [81, 82, 83, 239], datatype='hand')
            plt.figure()
            ax = plt.axes(projection='3d')
            gradient_trajectories(traj[:3], bounds=[-10,0,60,70,20,40], ax=ax)

        .. image:: _images/gradient_trajectories_3d.png

    Note:
        Automatic bounds aren't set in 3D plots. The best alternative is to first plot in 2D, then use
        those bounds to manually set the first 2 axes bounds for the 3D plot.
    '''

    if ax is None:
        ax = plt.gca()

    color_list = sns.color_palette(color_palette, n_colors)
    for traj in trajectories:
        n_pt = len(traj)

        if n_pt < n_colors:
            warnings.warn("Not enough datapoints to divide into n_colors!")

        # Assign labels to the trajectory according to color
        labels = np.zeros((n_pt,), dtype='int')
        size = (n_pt // n_colors) * n_colors # largest size we can evenly split into n_colors
        labels[:size] = np.repeat(range(n_colors), n_pt // n_colors)
        labels[size:] = n_colors - 1 # leftovers also get the last color
            
        # Split the labeled trajectories into segments with unique colors
        segments, labels = utils.segment_array(traj, labels, duplicate_endpoints=True)
        labels = np.array(labels).astype(int)
        colors = [color_list[i] for i in labels]

        # Plot as line collections in 3D, fall back to 2D
        try:
            ax.set_zlabel('z')
            segments = [np.vstack([s[:,0], s[:,1], s[:,2]]).T for s in segments]
            lc = Line3DCollection(segments, colors=colors, **kwargs)
            ax.add_collection(lc)
            ax.set_box_aspect((1, 1, 1))
        except:
            segments = [np.vstack([s[:,0], s[:,1]]).T for s in segments]
            lc = LineCollection(segments, colors=colors, **kwargs)
            ax.add_collection(lc)
                
    try:
        ax.set_zlabel('z')
        ax.set_box_aspect((1, 1, 1))
    except:
        ax.set_aspect('equal', adjustable='box')

    if bounds is not None: 
        set_bounds(bounds, ax)
    else:
        ax.margins(0.05) # The ax.add_collection() call doesn't automatically set margins

    ax.set_xlabel('x')
    ax.set_ylabel('y')        

def plot_sessions_by_date(trials, dates, *columns, method='sum', labels=None, ax=None):
    '''
    Plot session data organized by date and aggregated such that if there are multiple rows on 
    a given date they are combined into a single value using the given method. If the method
    is 'mean' then the values will be averaged for each day, for example for size of cursor. The 
    average is weighted by the number of trials in that session. If the  method is 'sum' then the 
    values will be added together on each day, for example for number of trials.
    
    Example:

        Plotting success rate averaged across days.

        .. code-block:: python

            from datetime import date, timedelta
            date = [date.today() - timedelta(days=1), date.today() - timedelta(days=1), date.today()]
            success = [70, 65, 65]
            trials = [10, 20, 10]

            fig, ax = plt.subplots(1,1)
            plot_sessions_by_date(trials, dates, success, method='mean', labels=['success rate'], ax=ax)
            ax.set_ylabel('success (%)')

        .. image:: _images/sessions_by_date.png
        
    Args:
        trials (nsessions):
        dates (nsessions): 
        *columns (nsessions): dataframe columns or numpy arrays to plot
        method (str, optional): how to combine data within a single date. Can be 'sum' or 'mean'.
        labels (list, optional): string label for each column to go into the legend
        ax (pyplot.Axes, optional): axis on which to plot
    '''
    dates = np.array(dates)
    first_day = np.min(dates)
    last_day = np.max(dates)
    plot_days = pd.date_range(start=first_day, end=last_day).to_list()
    n_columns = len(columns)
    n_days = len(plot_days)
    aggregate = np.zeros((n_columns, n_days))

    for idx_day in range(n_days):
        day = plot_days[idx_day]
        for idx_column in range(n_columns):
            values = np.array(columns[idx_column])[dates == day]
            
            try:
                if method == 'sum':
                    aggregate[idx_column, idx_day] = np.sum(values)
                elif method == 'mean':
                    day_trials = np.array(trials)[dates == day]
                    aggregate[idx_column, idx_day] = np.average(values, weights=day_trials)
                else:
                    raise ValueError("Unknown method for combining data")
            except:
                aggregate[idx_column, idx_day] = np.nan

    if ax == None:
        ax = plt.gca()
    for idx_column in range(n_columns):
        if hasattr(columns[idx_column], 'name'):
            ax.plot(plot_days, aggregate[idx_column,:], '.-', label=columns[idx_column].name)
        else:
            ax.plot(plot_days, aggregate[idx_column,:], '.-')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=80)
    
    if labels:
        ax.legend(labels)
    else:
        ax.legend()

def plot_sessions_by_trial(trials, *columns, labels=None, ax=None):
    '''
    Plot session data by absolute number of trials completed
    
    Example:

        Plotting success rate over three sessions.

        .. code-block:: python

            success = [70, 65, 60]
            trials = [10, 20, 10]

            fig, ax = plt.subplots(1,1)
            plot_sessions_by_trial(trials, success, labels=['success rate'], ax=ax)
            ax.set_ylabel('success (%)')

        .. image:: _images/sessions_by_trial.png
        
    Args:
        trials (nsessions): number of trials in each session
        *columns (nsessions): dataframe columns or numpy arrays to plot
        labels (list, optional): string label for each column to go into the legend
        ax (pyplot.Axes, optional): axis on which to plot
    '''
    if ax == None:
        ax = plt.gca()
    for idx_column in range(len(columns)):
        values = columns[idx_column]
        trial_values = []
        
        # Accumulate individual trials with the values given for each session
        for v, t in zip(values, trials):
            trial_values = np.concatenate((trial_values, np.tile(v, t)))
        
        if hasattr(columns[idx_column], 'name'):
            ax.plot(trial_values,  '.-', label=values.name)
        else:
            ax.plot(trial_values,  '.-')

    ax.set_xlabel('trials')
    if labels:
        ax.legend(labels)
    else:
        ax.legend()

def plot_events_time(events, event_timestamps, labels, ax=None, colors=['tab:blue','tab:orange','tab:green']):
    '''
    This function plots multiple different events on the same plot. The first event (item in the list)
    will be displayed on the bottom of the plot.
    
    .. image:: _images/events_time.png
    
    Args:
        events (list (nevents) of 1D arrays (ntime)): List of Logical arrays that denote when an event(for example, a reward) occurred during an experimental session. Each item in the list corresponds to a different event to plot. 
        event_timestamps (list (nevents) of 1D arrays ntime): List of 1D arrays of timestamps corresponding to the events list. 
        labels (list (nevents) of str) : Event names for each list item.
        ax (axes handle): Axes to plot
        colors (list of str): Color to use for each list item
    '''

    if ax is None:
        ax = plt.gca()

    n_events = len(events)
    for i in range(n_events):
        this_events = events[i]
        this_timestamps = event_timestamps[i]
        n_timebins = np.shape(this_events)[0]

        if n_events <= len(colors):
            this_color = colors[i]
            ax.step(this_timestamps, 0.9*(this_events)+i+0.1, where='post', c=this_color)
        else:
            ax.step(this_timestamps, 0.9*(this_events)+i+0.1, where='post')
    ax.set_yticks(np.arange(n_events)+0.5)
    ax.set_yticklabels(labels)

    ax.set_xlabel('Time (s)') 

def plot_waveforms(waveforms, samplerate, plot_mean=True, ax=None):
    '''
    This function plots the input waveforms on the same figure and can overlay the mean if requested

    .. image:: _images/waveform_plot_example.png

    Args:
        waveforms (nt, nwfs): Array of waveforms to plot
        samplerate (float): Sampling rate of waveforms to calculate time axis. [Hz]
        plot_mean (bool): Indicate if the mean waveform should be plotted. Defaults to plot mean.
        ax (axes handle): Axes to plot
    '''

    if ax is None:
        ax = plt.gca()
    
    time_axis = (1e6)*np.arange(waveforms.shape[0])/samplerate

    if plot_mean:
        ax.plot(time_axis, waveforms, color='black', alpha=0.5)
        mean_waveform = np.nanmean(waveforms, axis=1)
        ax.plot(time_axis, mean_waveform, color='red')
    else:
        ax.plot(time_axis, waveforms)

    ax.set_xlabel(r'Time ($\mu$s)')

def plot_tuning_curves(fit_params, mean_fr, targets, n_subplot_cols=5, ax=None):
    '''
    This function plots the tuning curves output from analysis.run_tuningcurve_fit overlaying the actual firing rate data.
    The dashed line is the model fit and the solid line is the actual data. 

    .. image:: _images/tuning_curves_plot.png

    Args:
        fit_params (nunits, 3): Model fit coefficients. Output from analysis.run_tuningcurve_fit or analysis.curve_fitting_func
        mean_fr (nunits, ntargets): The average firing rate for each unit for each target.
        target_theta (ntargets): Orientation of each target in a center out task [degrees]. Corresponds to order of targets in 'mean_fr'
        n_subplot_cols (int): Number of columns to plot in subplot. This function will automatically calculate the number of rows. Defaults to 5
        ax (axes handle): Axes to plot

    '''
    nunits = mean_fr.shape[0]
    n_subplot_rows = ((nunits-1)//n_subplot_cols)+1
    axinput = True

    if ax is None:
        fig, ax = plt.subplots(n_subplot_rows, n_subplot_cols)
        axinput = False
        
    nplots = n_subplot_rows*n_subplot_cols
    for iunit in range(nplots):
        if nunits > n_subplot_cols and n_subplot_cols!=1:
          nrow = iunit//n_subplot_cols
          ncol = iunit - (nrow*n_subplot_cols)
          # Remove axis that aren't used
          if iunit >= nunits:
            ax[nrow, ncol].remove()
          else:
            ax[nrow, ncol].plot(targets, mean_fr[iunit,:], 'b-', label='data')
            ax[nrow, ncol].plot(targets, analysis.curve_fitting_func(targets, fit_params[iunit, 0], fit_params[iunit, 1], fit_params[iunit,2]), 'b--', label='fit')
            ax[nrow, ncol].set_title('Unit ' +str(iunit))

        else:
          # Remove axis that aren't used
          if iunit >= nunits:
            ax[iunit].remove()
          else:
            ax[iunit].plot(targets, mean_fr[iunit,:], 'b-', label='data')
            ax[iunit].plot(targets, analysis.curve_fitting_func(targets, fit_params[iunit, 0], fit_params[iunit, 1], fit_params[iunit,2]), 'b--', label='fit')
            ax[iunit].set_title('Unit ' +str(iunit))

    if not axinput:
        fig.tight_layout()
        
def plot_boxplots(data, plt_xaxis, trendline=True, facecolor='gray', linecolor='k', box_width=0.5, ax=None):
    '''
    This function creates a boxplot for each column of input data. If the input data has NaNs, they are ignored.

    .. image:: _images/boxplot_example.png

    Args:
        data (ncol): Data to plot. A different boxplot is created for each entry of the list.
        plt_xaxis (ncol): X-axis locations or labels to plot the boxplot of each column
        trendline (bool): If a line should be used to connect boxplots
        facecolor (color): Color of the box faces. Can be any input that pyplot interprets as a color.
        linecolor (color): Color of the connecting lines.
        ax (axes handle): Axes to plot
    '''
    if ax is None:
        ax = plt.gca()
        
    if np.ndim(data) > 1:
        data = [data[:,i] for i in range(data.shape[1])]

    if trendline:
        ax.plot(plt_xaxis, np.nanmedian(data, axis=1), color=facecolor)
    
    for featidx, ifeat in enumerate(plt_xaxis):
        temp_data = data[featidx]
        try:
            int(ifeat)
        except:
            ifeat = featidx
        ax.boxplot(temp_data[~np.isnan(temp_data)], 
            positions=np.array([ifeat]), patch_artist=True, widths=box_width, 
            boxprops=dict(facecolor=facecolor, color=linecolor), capprops=dict(color=linecolor),
            whiskerprops=dict(color=linecolor), flierprops=dict(color=facecolor, markeredgecolor=facecolor),
            medianprops=dict(color=linecolor))
    ax.set_xticklabels(plt_xaxis)

def advance_plot_color(ax, n):
    '''
    Utility to skip colors for the given axis.
    
    Args:
        ax (pyplot.Axes): specify which axis to advance the color
        n (int): how many colors to skip in the cycle
    '''
    for _ in range(n):
        next(ax._get_lines.prop_cycler)

def profile_data_channels(data, samplerate, figuredir, **kwargs):
    """
    Runs `plot_channel_summary` and `combine_channel_figures` on all channels in a data array

    Args:
        data (nt, nch): numpy array of neural data
        samplerate (int): sampling rate of data
        figuredir (str): string indicating file path to desired save directory
        kwargs (**dict): keyword arguments to pass to plot_channel_summary()

    .. image:: _images/channel_profile_example.png
    
    """
    
    if not os.path.exists(figuredir):
        os.makedirs(figuredir)
    _, nch = data.shape
    
    for chidx in tqdm(range(nch)):
        chname = f'ch. {chidx+1}'
        fig = plot_channel_summary(data[:,chidx], samplerate, title=chname, **kwargs)
        fig.savefig(os.path.join(figuredir,f'ch_{chidx}.png'))
        
    combine_channel_figures(figuredir, nch=nch, figsize=kwargs.pop('figsize', (6,5)), dpi=kwargs.pop('dpi', 150))

    
def combine_channel_figures(figuredir, nch=256, figsize=(6,5), dpi=150):
    """
    Combines all channel figures in directory generated from plot_channel_summary

    Args:
        figuredir (str): path to directory of channel profile images
        nch (int, optional): number of channels from data array. Determines combined image layout. Defaults to 256.
        figsize (tuple, optional): (width, height) to pass to pyplot. Default (6, 5)
        dpi (int, optional): resolution to pass to pyplot. Default 150
    """
    
    assert os.path.exists(figuredir), f"Directory not found: {figuredir}"
    
    ncol = int(np.ceil(np.sqrt(nch))) # make things as square as possible
    nrow = int(np.ceil(nch/ncol))
    imgw = figsize[0] * dpi # I should get these from the individual files...
    imgh = figsize[1] * dpi
    
    grid = Image.new(mode='RGB', size=(ncol*imgw, nrow*imgh))
    
    print(f'profiling all {nch} channels...')
    for chidx in tqdm(range(nch)):
        figurefile = os.path.join(figuredir,f'ch_{chidx}.png')
        rowidx = chidx // ncol
        colidx = chidx % ncol
        if not os.path.exists(figurefile):
            continue
        else:
            with Image.open(figurefile) as img:
                grid.paste(img,box=(colidx*imgw, rowidx*imgh))
    
    grid.save(os.path.join(figuredir,'all_ch.png'),'png')


def plot_channel_summary(chdata, samplerate, nperseg=None, noverlap=None, trange=None, title=None, figsize=(6, 5), dpi=150, frange=(0, 80), cmap_lim=(0, 40)):
    """
    Plot time domain trace, spectrogram and normalized (z-scored) spectrogram. Computes spectrogram.
    
    ::

        ---------------
        | time series |
        |-------------|
        | spectrogram |
        |-------------|
        | norm sgram  |
        ---------------
    
    Args:
        chdata (nt,1): neural recording data from a given channel (lfp, ecog, broadband)
        samplerate (int): data sampling rate
        nperseg (int): length of each spectrogram window (in samples)
        noverlap (int): number of samples shared between neighboring spectrogram windows (in samples)
        trange (tuple, optional): (min, max) time range to display. Default show the entire time series
        title (str, optional): print a title above the timeseries data. Default None
        figsize (tuple, optional): (width, height) to pass to pyplot. Default (6, 5)
        dpi (int, optional): resolution to pass to pyplot. Default 150
        frange (tuple, optional): range of frequencies to display in spectrogram. Default (0, 80)
        cmap_lim (tuple, optional): clim to display in the spectrogram. Default (0, 40)

    Outputs:
        fig (Figure): Figure object
    """
    
    assert len(chdata.shape) < 2, "Input data array must be 1d"
    
    time = np.arange(len(chdata))/samplerate
    if trange is None:
        trange = (time[0], time[-1])
                                   
    if nperseg is None:
        nperseg = int(2*samplerate)
                                
    if noverlap is None:
        noverlap = int(1.5*samplerate)
    
    f_sg, t_sg, sgram = signal.spectrogram(
        chdata,
        fs=samplerate,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='linear'
    )
    log_sgram = np.log10(sgram)
    
    fig, ax = plt.subplots(3,1,figsize=figsize,dpi=dpi,constrained_layout=True,sharex=True)
    ax[0].plot(time, chdata)
    sg_pcm = ax[1].pcolormesh(t_sg,f_sg,10*log_sgram,vmin=cmap_lim[0],vmax=cmap_lim[1],shading='auto')
    ax[1].set_ylim(*frange)
    sg_cb = plt.colorbar(sg_pcm,ax=ax[1])
    sg_cb.ax.set_ylabel('dB$\mu$')
    sgn_pcm = ax[2].pcolormesh(t_sg,f_sg,zscore(log_sgram,axis=-1),vmin=-3,vmax=3,shading='auto',cmap='bwr')
    ax[2].set_ylim(*frange)
    sgn_cb = plt.colorbar(sgn_pcm,ax=ax[2])
    sgn_cb.ax.set_ylabel('z-scored dB$\mu$')
    ax[0].set_xlim(*trange)
    ax[0].set_ylabel('amp. ($\mu V$)')
    ax[1].set_ylabel('freq. (Hz)')
    ax[2].set_ylabel('freq. (Hz)')
    ax[2].set_xlabel('time (s)')
    ax[0].set_title(title)
    
    return fig

def plot_corr_over_elec_distance(acq_data, acq_ch, elec_pos, ax=None, **kwargs):
    '''
    Makes a plot of correlation vs electrode distance for the given data.
    
    Args:
        acq_data (nt, nch): acquisition data indexed by acq_ch
        acq_ch (nelec): 1-indexed list of acquisition channels that are connected to electrodes
        elec_pos (nelec, 2): x, y position of each electrode in cm
        ax (pyplot.Axes, optional): axis on which to plot
        kwargs (dict, optional): other arguments to supply to :func:`aopy.analysis.calc_corr_over_elec_distance`

    Example:
        Using the multichannel test data generator in utils, we get a phase-shifted sine wave in each channel. 
        Assigning each channel i to an electrode with position (i, 0), the correlation across distances looks like this:

        .. code-block:: python

            duration = 0.5
            samplerate = 1000
            n_channels = 30
            frequency = 100
            amplitude = 0.5
            acq_data = aopy.utils.generate_multichannel_test_signal(duration, samplerate, n_channels, frequency, amplitude)
            acq_ch = (np.arange(n_channels)+1).astype(int)
            elec_pos = np.stack((range(n_channels), np.zeros((n_channels,))), axis=-1)
            
            plt.figure()
            plot_corr_over_elec_distance(acq_data, acq_ch, elec_pos)

        .. image:: _images/corr_over_dist.png

    '''
    if ax is None:
        ax = plt.gca()
    label = kwargs.pop('label', None)
    dist, corr = analysis.calc_corr_over_elec_distance(acq_data, acq_ch, elec_pos, **kwargs)
    ax.plot(dist, corr, label=label)
    ax.set_xlabel('binned electrode distance (cm)')
    ax.set_ylabel('correlation')
    ax.set_ylim(0,1)


def plot_tfr(values, times, freqs, cmap='plasma', logscale=False, ax=None, **kwargs):
    '''
    Plot a time-frequency representation of a signal.

    Args:
        values ((nt, nfreq) array): 
        times ((nt,) array): 
        freqs ((nfreq,) array): 
        cmap (str, optional): colormap to use for plotting
        logscale (bool, optional): apply a log scale to the color axis. Default False.
        ax (pyplot.Axes, optional): axes on which to plot. Default current axis.
        kwargs (dict, optional): other keyword arguments to pass to pyplot
        
    Returns:
        pyplot.Image: image object returned from pyplot.pcolormesh. Useful for adding colorbars, etc.
        
    Examples:
        
        .. code-block:: python

            fig, ax = plt.subplots(3,1,figsize=(4,6))

            samplerate = 1000
            data_200_hz = aopy.utils.generate_multichannel_test_signal(2, samplerate, 8, 200, 2)
            nt = data_200_hz.shape[0]
            data_200_hz[:int(nt/3),:] /= 3
            data_200_hz[int(2*nt/3):,:] *= 2

            data_50_hz = aopy.utils.generate_multichannel_test_signal(2, samplerate, 8, 50, 2)
            data_50_hz[:int(nt/2),:] /= 2

            data = data_50_hz + data_200_hz
            print(data.shape)
            aopy.visualization.plot_timeseries(data, samplerate, ax=ax[0])
            aopy.visualization.plot_freq_domain_amplitude(data, samplerate, ax=ax[1])

            freqs = np.linspace(1,250,100)
            coef = aopy.analysis.calc_cwt_tfr(data, freqs, samplerate, fb=10, f0_norm=1, verbose=True)
            t = np.arange(nt)/samplerate
            
            print(data.shape)
            print(coef.shape)
            print(t.shape)
            print(freqs.shape)
            pcm = aopy.visualization.plot_tfr(abs(coef[:,:,0]), t, freqs, 'plasma', ax=ax[2])

            fig.colorbar(pcm, label='Power', orientation = 'horizontal', ax=ax[2])
            
        .. image:: _images/tfr_cwt_50_200.png


    See Also:
        :func:`~aopy.analysis.calc_cwt_tfr`
    '''
    
    if ax == None:
        ax = plt.gca()
        
    if logscale:
        pcm = ax.pcolormesh(times, freqs, np.log10(values), cmap=cmap, **kwargs)
    else:
        pcm = ax.pcolormesh(times, freqs, values, cmap=cmap, **kwargs)
    pcm.set_edgecolor('face')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    return pcm

def get_color_gradient_RGB(npts, end_color, start_color=[1,1,1]):
    '''
    This function outputs an ordered list of RGB colors that are linearly spaced between white and the input color. See also sns.color_palette for a gradient of RGB values within a Seaborn color palette.

    Examples:
        
        .. code-block:: python

                npts = 200
                x = np.linspace(0, 2*np.pi, npts)
                y = np.sin(x)
                fig, ax = plt.subplots()
                ax.scatter(x, y, c=get_color_gradient(npts, 'g', [1,0,0]))
    
        .. image:: _images/color_gradient_example.png

    Args:
        npts (int): How many different colors are part of the gradient
        end_color (str or list): Color that ends the gradient. Can be any matplotlib color or specific RGB values.
        start_color (str or list): Color that ends the gradient. Can be any matplotlib color or specific RGB values. Defaults to white.

    Returns:
        (npts, 3): An array with linearly spaced colors from the start to end
    '''
    rgb_end = matplotlib.colors.to_rgb(end_color)
    rgb_start = matplotlib.colors.to_rgb(start_color)
    ct = np.zeros((npts, 3))
    ct[:,0] = np.flip(np.linspace(rgb_end[0], rgb_start[0], npts))
    ct[:,1] = np.flip(np.linspace(rgb_end[1], rgb_start[1], npts))
    ct[:,2] = np.flip(np.linspace(rgb_end[2], rgb_start[2], npts))
    return ct