# visualization.py
# Code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)
import string
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
from datetime import timedelta

from .. import precondition
from .. import analysis
from .. import data as aodata
from .. import utils
from .. import preproc

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

def subplots_with_labels(n_rows, n_cols, return_labeled_axes=False, 
                         rel_label_x=-0.25, rel_label_y=1.1,
                         label_font_size=11, constrained_layout=False, 
                         **kwargs):
    '''
    Create a figure with subplots labeled with letters. Augments plt.subplots().

    Examples:

        Generate a figure with 2 rows and 2 columns of subplots, labeled A, B, C, D

        .. code-block:: python
            
            fig, axes = subplots_with_labels(2, 2, constrained_layout=True)

        .. image:: _images/labeled_subplots.png

    Args:
        n_rows (int): Number of rows of subplots.
        n_cols (int): Number of columns of subplots.
        return_labeled_axes (bool, optional): Whether to return the labeled axes. Default False.
        rel_label_x (float, optional): The relative x position of the subplot label. Default -0.25.
        rel_label_y (float, optional): The relative y position of the subplot label. Default 1.1
        label_font_size (int, optional): The font size of the subplot label. Default 11.
        constrained_layout (bool, optional): Whether to use constrained layout. Default is False.
        **kwargs: Additional keyword arguments to pass to plt.subplot_mosaic.

    Returns:
        fig (Figure): The created figure.
        axes (np.ndarray): The created axes.
        labels_axes (dict, optional): The labeled axes if return_labeled_axes is True.
    '''
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

def place_subplots(fig, positions, width, height, **kwargs):
    '''
    Plotting utility to create subplots in arbitrary positions on a figure. Positions 
    are in inches from the bottom left corner of the figure.
    
    Args:
        fig (pyplot.Figure): figure to place the subplots on
        positions (npos, 2): list of (x, y) coordinates (in inches) where to center the subplots 
        width (float): width (in inches) of each subplot
        height (float): height (in inches) of each subplot
        kwargs (dict, optional): other keyword arguments to pass to fig.add_axes
        
    Returns:
        list: pyplot.Axes handles for each position

    Examples:

        .. code-block:: python

            fig = plt.figure(figsize=(4,6))
            positions = [[1, 2], [3, 4]]
            width = 1
            height = 1
            ax = place_subplots(fig, positions, width, height)
            ax[0].annotate('1', (0.5,0.5), fontsize=40)
            ax[1].annotate('2', (0.5,0.5), fontsize=40)

        .. image:: _images/place_subplots_1.png

        .. code-block:: python
   
            fig = plt.figure(figsize=(4,6))
            positions = [[1, 1.5], [3, 4.5]]
            width = 2
            height = 3
            ax = place_subplots(fig, positions, width, height)
            ax[0].annotate('1', (0.5,0.5), fontsize=40)
            ax[1].annotate('2', (0.5,0.5), fontsize=40)

        .. image:: _images/place_subplots_2.png
            
    '''
    # Normalize the positions to fit into the size of the figure
    fig_width, fig_height = fig.get_size_inches()
    positions = np.array(positions, dtype='float')
    positions[:,0] = positions[:,0] / fig_width
    positions[:,1] = positions[:,1] / fig_height
    width /= fig_width
    height /= fig_height

    # Place subplots
    ax = []
    for cx, cy in positions:
        left = cx - width/2
        bottom = cy - height/2
        ax.append(fig.add_axes([left, bottom, width, height], **kwargs))
    return ax

def place_Opto32_subplots(fig_size=5, subplot_size=0.75, offset=(0.,-0.25), theta=0, **kwargs):
    '''
    Wrapper around place_subplots() for the Opto32 stimulation sites.

    Args:
        fig_size (float): width and height (in inches) of the figure
        subplot_size (float): width and height (in inches) of each subplot
        offset (tuple): x and y offset (in inches) from the bottom left corner of the figure
        theta (float): rotation (in degrees) to apply to positions.
        kwargs (dict, optional): other keyword arguments to pass to fig.add_axes

    Returns:
        tuple: tuple containing:
        | **fig (pyplot.Figure):** figure where the subplots were placed
        | **ax (list):** pyplot.Axes handles for each stimulation site

    Examples:

        .. image:: _images/place_Opto32_subplots.png
    '''
    stim_pos, _, _ = aodata.load_chmap('Opto32', theta=theta)

    # Normalize the positions to the width and height of the figure
    stim_pos = (stim_pos - np.mean(stim_pos, axis=0)) / (np.max(stim_pos) - np.min(stim_pos)) * fig_size + fig_size/2

    # Place subplots
    fig = plt.figure(figsize=(fig_size,fig_size), **kwargs)
    ax = place_subplots(fig, stim_pos + np.array(offset), subplot_size, subplot_size)

    # Remove the axis labels
    for ax_ in ax:
        ax_.tick_params(
            which='both',
            bottom=False,
            left=False,  
            labelbottom=False,
            labelleft=False
        )
    return fig, ax

def plot_timeseries(data, samplerate, t0=0., ax=None, **kwargs):
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
        t0 (float, optional): time (in seconds) of the first sample. Default 0.
        ax (pyplot axis, optional): where to plot
        kwargs (dict, optional): optional keyword arguments to pass to plt.plot
    '''
    if np.ndim(data) < 2:
        data = np.expand_dims(data, 1)
    if ax is None:
        ax = plt.gca()

    time = np.arange(np.shape(data)[0]) / samplerate + t0
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
    x_pos = np.round(x_pos, 9) # avoid floating point errors
    y_pos = np.round(y_pos, 9)

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


def calc_data_map(data, x_pos, y_pos, grid_size, interp_method='nearest', threshold_dist=None, extent=None):
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
        extent (list): [xmin, xmax, ymin, ymax] to define the extent of the interpolated grid. Default None,
            which will use the min and max of the x and y positions.

    Returns:
        tuple: tuple containing:
            | *data_map (grid_size array, e.g. (16,16)):* map of the data on the given grid
            | *xy (grid_size array, e.g. (16,16)):* new grid positions to use with this map

    '''
    if extent is None:
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
    cmap = copy.copy(plt.get_cmap(cmap))
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

def plot_spatial_drive_map(data, bad_elec=[], interp=True, drive_type='ECoG244', cmap='bwr', 
                           theta=0, ax=None, **kwargs):
    '''
    Plot a 2D spatial map of data from a spatial electrode array.

    Args:
        data ((nch,) array): values from the spatial drive to plot in 2D
        bad_elec (list, optional): channels to remove from the plot. Defaults to [].
        interp (bool, optional): flag to include 2D interpolation of the result. Defaults to True.
        drive_type (str, optional): type of drive. Defaults to 'ECoG244'.
        cmap (str, optional): matplotlib colormap to use in image. Defaults to 'bwr'.
        theta (float): rotation (in degrees) to apply to positions. rotations are applied clockwise, 
            e.g., theta = 90 rotates the map clockwise by 90 degrees, -90 rotates the map anti-clockwise 
            by 90 degrees. Default 0.
        ax (pyplot.Axes, optional): axis on which to plot. Defaults to None.
        kwargs (dict): dictionary of additional keyword argument pairs to send to calc_data_map and plot_spatial_map.

    Returns:
        pyplot.Image: image returned by pyplot.imshow. Use to add colorbar, etc.
    '''
    
    if ax is None:
        ax = plt.gca()
    
    # Load the signal path files
    elec_pos, acq_ch, elecs = aodata.load_chmap(drive_type=drive_type, theta=theta)

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


def plot_ECoG244_data_map(data, bad_elec=[], interp=True, cmap='bwr', 
                          theta=0, ax=None, **kwargs):
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
    return plot_spatial_drive_map(data, bad_elec=bad_elec, interp=interp, drive_type='ECoG244', 
                                  cmap=cmap, theta=theta, ax=ax, **kwargs)

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
    elec_pos, acq_ch, elecs = aodata.load_chmap(drive_type, acq_ch, theta)

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

def color_targets(target_locations, target_idx, colors, target_radius, bounds=None, ax=None, **kwargs):
    '''
    Color targets according to their index. Useful for visualizing unique targets when trajectories
    aren't obviously aligned to specific targets.

    Args:
        target_locations ((ntargets, 2) or (ntargets, 3) array): array of target (x, y[, z]) locations
        target_idx ((ntargets,) array): array of indices for each target, used to determine color
        colors (list): list of colors corresponding to each unique index in target_idx
        target_radius (float): radius of the targets in cm
        bounds (tuple, optional): 4- or 6-element tuple describing (-x, x, -y, y[, -z, z]) cursor bounds
        ax (plt.Axis, optional): axis to plot the targets on (2D or 3D)
        **kwargs: additional keyword arguments to pass to plot_circles()

    Examples:
        Create and plot eight targets for a center-out task.
        
        .. code-block:: python

            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            radius = 6.5
            target_locations = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
            target_locations = np.vstack(([0, 0], target_locations))

        Specify the colors per target index in case they are out of order.
            
        .. code-block:: python

            target_idx = [0] + np.arange(1, 9).tolist()  # Center is index 0, peripheral are index 1 through 9
            colors = ['black'] + sns.color_palette("husl", 8)
            target_radius = 0.5
            bounds = (-8, 8, -8, 8)

        Use :func:`~aopy.visualization.color_targets` to plot the targets
        
        .. code-block:: python

            fig, ax = plt.subplots(figsize=(8, 8))
            color_targets(target_locations, target_idx, colors, target_radius, bounds, ax)
            ax.set_aspect('equal')
            filename = 'color_targets.png'

        .. image:: _images/color_targets.png
    '''
    
    assert len(target_locations) == len(target_idx), "Locations must be the same length as indices"
    target_locations = np.array(np.array(target_locations).tolist())
    target_idx = np.array(np.array(target_idx).tolist())
    loc_idx = np.concatenate((np.expand_dims(target_idx, 1), target_locations), axis=1)
    loc_idx = np.unique(loc_idx, axis=0)
    assert len(colors) >= len(np.unique(target_idx)), "Not enough colors for unique indices"
    for row in loc_idx:
        idx = row[0].astype(int)
        loc = row[1:]
        plot_circles([loc], target_radius, colors[idx], bounds=bounds, ax=ax, **kwargs)
        
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
            values = np.array(columns[idx_column])[dates == day.date()]
            
            try:
                if method == 'sum':
                    if len(values) > 0:
                        aggregate[idx_column, idx_day] = np.sum(values)
                    else:
                        aggregate[idx_column, idx_day] = np.nan
                elif method == 'mean':
                    day_trials = np.array(trials)[dates == day.date()]
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
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=(mdates.MO, mdates.TU, mdates.WE, 
                                                                mdates.TH, mdates.FR, mdates.SA, mdates.SU)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=80)
    
    if labels:
        ax.legend(labels)
    else:
        ax.legend()

def plot_sessions_by_trial(trials, *columns, dates=None, smoothing_window=None, labels=None, ax=None, **kwargs):
    '''
    Plot session data by absolute number of trials completed. Optionally split up the sessions by date and
    apply smoothing to each day's data.
    
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
        dates (nsessions, optional): dataframe columns or numpy arrays of the date of each session
        smoothing_window (int, optional): number of trials to smooth. Default no smoothing.
        labels (list, optional): string label for each column to go into the legend
        ax (pyplot.Axes, optional): axis on which to plot
    '''
    if ax == None:
        ax = plt.gca()  

    date_chg = []
    if dates is not None:
        trial_dates = np.repeat(np.array(dates), trials)
        date_chg = np.insert(np.where(np.diff(trial_dates) > timedelta(0))[0] + 1, 0, 0)

    for idx_column in range(len(columns)):
        
        # Accumulate individual trials with the values given for each session
        values = np.array(columns[idx_column])
        trial_values = np.repeat(values, trials)
        
        # Apply smoothing
        if smoothing_window is not None and dates is not None:
            split = np.split(trial_values, date_chg[1:])
            split = [analysis.calc_rolling_average(s, window_size=smoothing_window, mode='nan') for s in split]
            trial_values = np.concatenate(split)
        elif smoothing_window is not None:
            trial_values = analysis.calc_rolling_average(trial_values, window_size=smoothing_window)
        
        # Plot with additional kwargs
        if hasattr(columns[idx_column], 'name'):
            ax.plot(trial_values, label=columns[idx_column].name, **kwargs)
        else:
            ax.plot(trial_values, **kwargs)

    # Add date labels
    for i in date_chg:
        date = trial_dates[i]
        ax.axvline(i, ymin=0, ymax=1, color='gray', alpha=0.5, linestyle='dashed')
        ax.text(i, 1, str(date), color='gray', rotation=90, ha='left', va='top', 
                transform=ax.get_xaxis_transform())

    ax.set_xlabel('trials')
    if labels is not None:
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

    Args:
        data (ncol list or (m, ncol) array): Data to plot. A different boxplot is created for each entry of the list.
        plt_xaxis (ncol): X-axis locations or labels to plot the boxplot of each column
        trendline (bool): If a line should be used to connect boxplots
        facecolor (color): Color of the box faces. Can be any input that pyplot interprets as a color.
        linecolor (color): Color of the connecting lines.
        ax (axes handle): Axes to plot

    Examples:

        Using a rectangular array and numeric x-axis points.

        .. code-block:: python
            
            data = np.random.normal(0, 2, size=(20, 5))
            xaxis_pts = np.array([2,3,4,4.75,5.5])
            fig, ax = plt.subplots(1,1)
            plot_boxplots(data, xaxis_pts, ax=ax)

        .. image:: _images/boxplot_example.png

        Using a list of nonrectangular arrays with categorical x-axis points.

        .. code-block:: python
        
            data = [np.random.normal(0, 2, size=(10)), np.random.normal(0, 1, size=(20))]
            xaxis_pts = ['foo', 'bar']
            fig, ax = plt.subplots(1,1)
            plot_boxplots(data, xaxis_pts, ax=ax)

        .. image:: _images/boxplot_example_nonrectangular.png

    '''
    if ax is None:
        ax = plt.gca()

    # If data is 2D, turn the columns into lists
    if hasattr(data, 'ndim') and data.ndim == 2:
        data = [data[:,i] for i in range(data.shape[1])]

    # If data is a single column, make it a list
    try:
        int(data[0])
        data = [data]
    except:
        pass
        
    if trendline:
        ax.plot(plt_xaxis, [np.nanmedian(data[i]) for i in range(len(data))], color=facecolor)
    
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

    Examples:

        Using advance_plot_color to skip the first color in the cycle.

        .. code-block:: python

            plt.subplots()
            aopy.visualization.advance_plot_color(plt.gca(), 1)
            plt.plot(np.arange(10), np.arange(10))

        .. image:: _images/advance_plot_color.png
        
    '''
    for _ in range(n):
        next(ax._get_lines.prop_cycler)

def reset_plot_color(ax):
    '''
    Utility to reset the color cycle on a given axis to the default.

    Args:
        ax (pyplot.Axes): specify which axis to reset the color

    Examples:

        Using reset_plot_color to reset the color cycle between calls to `plt.plot()`.
        
        .. code-block:: python

            plt.subplots()
            plt.plot(np.arange(10), np.ones(10))
            aopy.visualization.reset_plot_color(plt.gca())
            plt.plot(np.arange(10), 1 + np.ones(10))

        .. image:: _images/reset_plot_color.png
    '''
    ax.set_prop_cycle(None)

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


def plot_corr_over_elec_distance(elec_data, elec_pos, ax=None, **kwargs):
    '''
    Makes a plot of correlation vs electrode distance for the given data.
    
    Args:
        elec_data (nt, nelec): electrode data with nch corresponding to elec_pos
        elec_pos (nelec, 2): x, y position of each electrode
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

    Updated:
        2024-03-13 (LRS): Changed input from acq_data and acq_ch to elec_data.
        2024-07-01 (LRS): Fixed default x-axis label units to mm.
    '''
    if ax is None:
        ax = plt.gca()
    label = kwargs.pop('label', None)
    dist, corr = analysis.calc_corr_over_elec_distance(elec_data, elec_pos, **kwargs)
    ax.plot(dist, corr, label=label)
    ax.set_xlabel('binned electrode distance (mm)')
    ax.set_ylabel('correlation')
    ax.set_ylim(0,1)

def plot_corr_across_entries(preproc_dir, subjects, ids, dates, band=(70,200), taper_len=0.1, num_seconds=60, 
                             cmap='viridis', ax=None, remove_bad_ch=True, **bad_ch_kwargs):
    '''
    Plot the correlation vs electrode distance for each entry in the given list of subjects, ids, and dates.
    
    Args:
        preproc_dir (str): path to the preprocessed data directory
        subjects (list): list of subject names
        ids (list): list of te_ids
        dates (list): list of dates
        band (tuple, optional): frequency band to filter the data. Default (70, 200)
        taper_len (float, optional): length of taper to use in the filter. Default 0.1
        num_seconds (int, optional): number of seconds to use in the correlation calculation. Default 60
        cmap (str, optional): colormap to use for plotting. Default 'viridis'
        ax (pyplot.Axes, optional): axis on which to plot. Default current axis
        remove_bad_ch (bool, optional): whether to remove bad channels from the data. Default True
        bad_ch_kwargs (dict, optional): keyword arguments to pass to :func:`a
    
    Example:
        Plotting the correlation vs electrode distance for a few entries in the preprocessed data directory.

        .. image:: _images/corr_over_entries.png
    '''
    assert len(subjects) == len(ids) == len(dates), "Subjects, ids, and dates must be equal length"
    
    if ax is None:
        ax = plt.gca()    
    ax.set_prop_cycle('color', sns.color_palette(cmap, len(subjects)))
        
    for subject, te_id, date in zip(subjects, ids, dates):

        try:
            lfp_data, lfp_metadata = aodata.load_preproc_lfp_data(preproc_dir, subject, te_id, date)
            exp_data, exp_metadata = aodata.load_preproc_exp_data(preproc_dir, subject, te_id, date)
        except:
            print(f"Could not find data for entry {te_id} ({subject} on {date})")
            continue
        try:
            elec_pos, acq_ch, _ = aodata.load_chmap(exp_metadata['drmap_drive_type'])
        except:
            elec_pos, acq_ch, _ = aodata.load_chmap('ECoG244')

        samplerate = lfp_metadata['samplerate']
        short_data = lfp_data[:num_seconds*samplerate,acq_ch-1]
        filt_data = precondition.mt_bandpass_filter(short_data, band, taper_len, 
                                                            samplerate, verbose=False)
        
        if remove_bad_ch:
            bad_ch = preproc.quality.detect_bad_ch_outliers(filt_data, **bad_ch_kwargs)
            filt_data = filt_data[:,~bad_ch]
            elec_pos = elec_pos[~bad_ch]
        
        plot_corr_over_elec_distance(filt_data, elec_pos, label=date, ax=ax)

    leg = ax.legend(bbox_to_anchor = (1,1))
    for obj in leg.legend_handles:
        obj.set_linewidth(4.0)

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

def plot_laser_sensor_alignment(sensor_volts, samplerate, stim_times, ax=None):
    '''
    Plot laser sensor data aligned to the stimulus times. Useful to debug laser timing issues to
    make sure the laser is actually on when you think it is.

    Args:
        sensor_volts ((nstim,) float array): laser sensor data
        samplerate (float): sampling rate of the sensor data
        stim_times ((nstim,) array): times at which the laser was turned on
        ax (pyplot.Axes, optional): axes on which to plot. Default current axis.
        kwargs (dict, optional): other keyword arguments to pass to pyplot
    
    Returns:
        pyplot.Image: image object returned from pyplot.pcolormesh. Useful for adding colorbars, etc.

    Examples:
        .. image:: _images/laser_sensor_alignment.png
    '''
    if ax is None:
        ax = plt.gca()
    time_before = 0.1 # seconds
    time_after = 0.1 # seconds
    analog_erp = analysis.calc_erp(sensor_volts, stim_times, time_before, time_after, samplerate)
    t = 1000*(np.arange(analog_erp.shape[0])/samplerate - time_before) # milliseconds
    im = plot_image_by_time(t, analog_erp[:,0,:], ylabel='trials')
    plt.xlabel('time (ms)')
    plt.title('laser sensor aligned')
    return im

def plot_circular_hist(data, bins=16, density=False, offset=0, proportional_area=False, gaps=False, normalize=False, ax=None, **kwargs):
    '''
    Plot a circular histogram of angles on a given ax. Adapted from: 
        https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python. 

    Args:            
        data (arr): angles to plot, in radians.
        bins (int, optional): defines the number of equal-width bins in the range. Default is 16.
        density (bool, optional): whether to return the probability density function at each bin, instead of the number of samples 
            (passed to np.histogram). Default is False.
        offset (float, optional): the offset for the location of the 0 direction, in radians. 
            Default is 0.
        proportional_area (bool, optional): If True, plots bars proportional to area. If False, plots bars
            proportional to radius. Default is False.
        gaps (bool, optional): whether to allow gaps between bins. If True, the bins will only span the values
            of the data. If False, the bins are forced to partition the entire [-pi, pi] range. Default is False.
        normalize (bool, optional): whether to normalize the bin values such that the max value is 1. Default is False.
        ax (pyplot.Axes, optional): axes on which to plot. Should be an axis instance created with 
            subplot_kw=dict(projection='polar'). Default current axis.
        kwargs (dict, optional): other keyword arguments to pass to ax.bar

    Returns:
        n (arr or list of arr): the number of values in each bin
        bins (arr): the edges of the bins
        patches (`.BarContainer` or list of a single `.Polygon`): container of individual artists used to create the histogram
            or list of such containers if there are multiple input datasets

    Examples:
        .. image:: _images/circular_histograms.png
    '''
    if ax is None:
        ax = plt.gca()

    # Wrap angles to [-pi, pi)
    data = (data+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(data, bins=bins, density=density)

    # Compute width of each bin
    widths = np.diff(bins)

    # If indicated, plot frequency proportional to area
    if proportional_area:
        # Area to assign each bin
        area = n / data.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
         # Remove ylabels for area plots (they are mostly obstructive)
        ax.set_yticks([])

    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # If indicated, normalize the bar values so that the max is 1
    if normalize:
        radius = radius/np.max(radius)

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, width=widths, align='edge', **kwargs)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)  

    return n, bins, patches