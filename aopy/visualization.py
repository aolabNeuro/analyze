# visualization.py
# Code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

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


from . import postproc
from . import analysis

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

    Example:
        Plot 50 and 100 Hz sine wave.

        ::
            data = np.reshape(np.sin(np.pi*np.arange(1000)/10) + np.sin(2*np.pi*np.arange(1000)/10), (1000))
            samplerate = 1000
            plot_timeseries(data, samplerate)

        .. image:: _images/timeseries.png

    Args:
        data (nt, nch): timeseries data in volts, can also be a single channel vector
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

def plot_freq_domain_amplitude(data, samplerate, ax=None, rms=False):
    '''
    Plots a amplitude spectrum of each channel on the given axis. Just need to input time series
    data and this will calculate and plot the amplitude spectrum. 

    Example:
        Plot 50 and 100 Hz sine wave amplitude spectrum. 

        ::
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
        ax.semilogx(non_negative_freq, data_ampl[:,ch]*1e6) # convert to microvolts
    ax.set_xlabel('Frequency (Hz)')
    if rms:
        ax.set_ylabel('RMS amplitude (uV)')
    else:
        ax.set_ylabel('Peak amplitude (uV)')

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

    Example:
        Make a plot of a 10 x 10 grid of increasing values with some missing data.
        
        ::
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
        | *data_map (2,n array):* map of the data on the given grid
        | *xy (2,n array):* new grid positions to use with this map

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


def plot_spatial_map(data_map, x, y, alpha_map=None, ax=None, cmap='bwr'):
    '''
    Wrapper around plt.imshow for spatial data

    Example:
        Make a plot of a 10 x 10 grid of increasing values with some missing data.
        
        ::
            data = np.linspace(-1, 1, 100)
            x_pos, y_pos = np.meshgrid(np.arange(0.5,10.5),np.arange(0.5, 10.5))
            missing = [0, 5, 25]
            data_missing = np.delete(data, missing)
            x_missing = np.reshape(np.delete(x_pos, missing),-1)
            y_missing = np.reshape(np.delete(y_pos, missing),-1)

            data_map = get_data_map(data_missing, x_missing, y_missing)
            plot_spatial_map(data_map, x_missing, y_missing)

        .. image:: _images/posmap.png

    Args:
        data_map (2,n array): map of x,y data
        x (list): list of x positions
        y (list): list of y positions
        alpha_map (2,n array): map of alpha values (optional, default alpha=1 everywhere)
        ax (int, optional): axis on which to plot, default gca
        cmap (str, optional): matplotlib colormap to use in image

    Returns:
        mappable: image object which you can use to add colorbar, etc.
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
    cmap.set_bad(color='black')
    
    # Make an alpha map scaled between 0 and 1
    if alpha_map is None:
        alpha_map = 1
    else:
        alpha_range = np.nanmax(alpha_map) - np.nanmin(alpha_map)
        alpha_map = (alpha_map - np.nanmin(alpha_map)) / alpha_range
        alpha_map[np.isnan(alpha_map)] = 0

    # Plot
    if ax is None:
        ax = plt.gca()
    image = ax.imshow(data_map, alpha=alpha_map, cmap=cmap, origin='lower', extent=extent)
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')

    return image

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

def saveanim(animation, base_dir, filename, dpi=72, **savefig_kwargs):
    '''
    Save an animation using ffmpeg

    Args:
        animation (pyplot.Animation): animation to save
        base_dir (str): directory to write
        filename (str): should end in '.mp4'
        dpi (float): resolution of the video file
        savefig_kwargs (kwargs, optional): arguments to pass to savefig
    '''
    filepath = os.path.join(base_dir, filename)
    animation.save(filepath, dpi=dpi, savefig_kwargs=savefig_kwargs)


def showanim(animation):
    '''
    Display an animation in a python notebook

    Args:
        animation (pyplot.Animation): animation to display
    '''
    from IPython.display import HTML  # not a required package
    HTML(animation.to_html5_video())


def animate_events(events, times, fps, xy=(0.3, 0.3), fontsize=30, color='g'):
    '''
    Silly function to plot events as text, frame by frame in an animation

    Args:
        events (list): list of event names or numbers
        times (list): timestamps of each event
        fps (float): sampling rate to animate
        xy (tuple, optional): (x, y) coorindates of the left bottom corner of each event label, from 0 to 1.
        fontsize (float, optional): size to draw the event labels

    Returns:
        matplotlib.animation.FuncAnimation: animation object
    '''
    frame_events, event_names = postproc.sample_events(events, times, fps)

    def display_text(num, events, names, note):
        display = names[events[num, :] == 1]
        if len(display) > 0:
            note.set_text(display[0])  # note if simultaneous events occur, we just print the first

    fig, ax = plt.subplots(1, 1)
    note = ax.annotate("", xy, fontsize=fontsize, color=color)
    plt.axis('off')
    return FuncAnimation(fig, display_text, frames=frame_events.shape[0],
                         interval=round(1000 / fps),
                         fargs=(frame_events, event_names, note))


def animate_trajectory_3d(trajectory, samplerate, history=1000, color='b',
                          axis_labels=['x', 'y', 'z']):
    '''
    Draws a trajectory moving through 3D space at the given sampling rate and with a
    fixed maximum number of points visible at a time.

    Args:
        trajectory (n, 3): matrix of n points
        samplerate (float): sampling rate of the trajectory data
        history (int, optional): maximum number of points visible at once
    '''

    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')

    line, = ax.plot(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color=color)

    ax.set_xlim((np.nanmin(trajectory[:, 0]), np.nanmax(trajectory[:, 0])))
    ax.set_xlabel(axis_labels[0])

    ax.set_ylim((np.nanmin(trajectory[:, 1]), np.nanmax(trajectory[:, 1])))
    ax.set_ylabel(axis_labels[1])

    ax.set_zlim((np.nanmin(trajectory[:, 2]), np.nanmax(trajectory[:, 2])))
    ax.set_zlabel(axis_labels[2])

    def draw(num):
        length = min(num, history)
        start = num - length
        line.set_data(trajectory[start:num, 0], trajectory[start:num, 1])
        line.set_3d_properties(trajectory[start:num, 2])
        return line,

    return FuncAnimation(fig, draw, frames=trajectory.shape[0],
                         init_func=lambda: None, interval=1000. / samplerate)

def animate_spatial_map(data_map, x, y, samplerate, cmap='bwr'):
    '''
    Animates a 2d heatmap. Use :func:`aopy.visualization.get_data_map` to get a 2d array
    for each timepoint you want to animate, then put them into a list and feed them to this
    function. See also :func:`aopy.visualization.show_anim` and :func:`aopy.visualization.save_anim`

    Example:
        ::
        
            samplerate = 20
            duration = 5
            x_pos, y_pos = np.meshgrid(np.arange(0.5,10.5),np.arange(0.5, 10.5))
            data_map = []
            for frame in range(duration*samplerate):
                t = np.linspace(-1, 1, 100) + float(frame)/samplerate
                c = np.sin(t)
                data_map.append(get_data_map(c, x_pos.reshape(-1), y_pos.reshape(-1)))

            filename = 'spatial_map_animation.mp4'
            ani = animate_spatial_map(data_map, x_pos, y_pos, samplerate, cmap='bwr')
            saveanim(ani, write_dir, filename)

        .. raw:: html

            <video controls src="_static/spatial_map_animation.mp4"></video>

    Args:
        data_map (nt): array of 2d maps
        x (list): list of x positions
        y (list): list of y positions
        samplerate (float): rate of the data_map samples
        cmap (str, optional): name of the colormap to use. Defaults to 'bwr'.
    '''

    # Plotting subroutine
    def plotdata(i):
        im.set_data(data_map[i])
        return im

    # Initial plot
    fig, ax = plt.subplots()
    im = plot_spatial_map(data_map[0], x, y, ax=ax, cmap=cmap)

    # Change the color limits
    min_c = np.min(np.array(data_map))
    max_c = np.max(np.array(data_map))
    im.set_clim(min_c, max_c)
        
    # Create animation
    ani = FuncAnimation(fig, plotdata, frames=len(data_map),
                            interval=1000./samplerate)

    return ani

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


def plot_targets(target_positions, target_radius, bounds=None, alpha=0.5, origin=(0, 0, 0), ax=None, unique_only=True):    
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
        alpha = alpha * np.ones(target_positions.shape[0])
    else:
        assert len(alpha) == target_positions.shape[0], "list of alpha values must be equal in length to the list of targets."

    if ax is None:
        ax = plt.gca()

    if unique_only:
        target_positions = np.unique(target_positions,axis=0)

    for i in range(0, target_positions.shape[0]):

        # Pad the vector to make sure it is length 3
        pos = np.zeros((3,))
        pos[:len(target_positions[i])] = target_positions[i]

        # Color according to its position
        if (pos == origin).all():
            target_color = 'm'
        else:
            target_color = 'b'

        # Plot in 3D or 2D
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        try:
            ax.set_zlabel('z')
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = pos[0] + target_radius * np.outer(np.cos(u), np.sin(v))
            y = pos[1] + target_radius * np.outer(np.sin(u), np.sin(v))
            z = pos[2] + target_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, alpha=alpha[i], color=target_color)
            ax.set_box_aspect((1, 1, 1))
        except:
            target = plt.Circle((pos[0], pos[1]),
                                radius=target_radius, alpha=alpha[i], color=target_color)
            ax.add_artist(target)
            ax.set_aspect('equal', adjustable='box')
    if bounds is not None: set_bounds(bounds, ax)

def plot_circles(circle_positions, circle_radius, circle_color = 'b', bounds=None, alpha=0.5, ax=None, unique_only=True):    
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
        alpha = alpha * np.ones(circle_positions.shape[0])
    else:
        assert len(alpha) == circle_positions.shape[0], "list of alpha values must be equal in length to the list of targets."

    if ax is None:
        ax = plt.gca()

    for i in range(0, circle_positions.shape[0]):

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


def plot_trajectories(trajectories, bounds=None, ax=None):
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
   '''
    if ax is None:
        ax = plt.gca()

    # Plot in 3D, fall back to 2D
    try:
        ax.set_zlabel('z')
        for path in trajectories:
            ax.plot(*path.T)
        ax.set_box_aspect((1, 1, 1))
    except:
        for path in trajectories:
            ax.plot(path[:, 0], path[:, 1])
        ax.set_aspect('equal', adjustable='box')

    if bounds is not None: set_bounds(bounds, ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def color_trajectories(trajectories, labels, colors, ax=None, **kwargs):
    '''
    Draws the given trajectories but with the color of each trajectory corresponding to its given label.
    Works for 2D and 3D axes

    Example:

        ::
            >>> trajectories =[
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
            >>> labels = [0, 0, 1, 0]
            >>> colors = ['r', 'b']
            >>> color_trajectories(trajectories, labels, colors)

        .. image:: _images/color_trajectories.png

    Args:
        trajectories (ntrials): list of (n, 2) or (n, 3) trajectories where n can vary across each trajectory
        labels (ntrials): integer array of labels for each trajectory. Basically an index for each trajectory
        colors (ncolors): list of colors. The number of colors should be the same as the number of unique labels.
        ax (plt.Axis, optional): axis to plot the targets on
        **kwargs (dict): other arguments for plot_trajectories(), e.g. bounds
    '''

    # Initialize a cycler with the appropriate colors
    style = plt.cycler(color=[colors[i] for i in labels])
    if ax is None:
        ax = plt.gca()
    ax.set_prop_cycle(style)

    # Use the regular trajectory plotting function
    plot_trajectories(trajectories, **kwargs)

def plot_sessions_by_date(trials, dates, *columns, method='sum', labels=None, ax=None):
    '''
    Plot session data organized by date and aggregated such that if there are multiple rows on 
    a given date they are combined into a single value using the given method. If the method
    is 'mean' then the values will be averaged for each day, for example for size of cursor. The 
    average is weighted by the number of trials in that session. If the  method is 'sum' then the 
    values will be added together on each day, for example for number of trials.
    
    Example:
        Plotting success rate averaged across days.

        ::
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

        ::
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
        
def plot_boxplots(data, plt_xaxis, trendline=True, facecolor=[0.5, 0.5, 0.5], linecolor=[0,0,0], box_width = 0.5, ax=None):
    '''
    This function creates a boxplot for each column of input data. If the input data has NaNs, they are ignored.

    .. image:: _images/boxplot_example.png

    Args:
        data (n1, n2): Data to plot. A different boxplot is created for each column of this variable.
        plt_xaxis (n2): X-axis locations to plot the boxplot of each column
        trendline (bool): If a line should be used to connect boxplots
        facecolor (list or word):
        linecolor (list or word):
        ax (axes handle): Axes to plot
    '''
    if ax is None:
        ax = plt.gca()

    if trendline:
        ax.plot(plt_xaxis, np.nanmedian(data, axis=0), color=facecolor)
    
    for featidx, ifeat in enumerate(plt_xaxis):
        temp_data = data[:,featidx]
        ax.boxplot(temp_data[~np.isnan(temp_data)], 
            positions=np.array([ifeat]), patch_artist=True, widths=box_width, 
            boxprops=dict(facecolor=facecolor, color=linecolor), capprops=dict(color=linecolor),
            whiskerprops=dict(color=linecolor), flierprops=dict(color=facecolor, markeredgecolor=facecolor),
            medianprops=dict(color=linecolor))

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
    """profile_data_channels

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
    """combine_channel_figures

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
    """plot_channel_summary
    
    Plot time domain trace, spectrogram and normalized (z-scored) spectrogram. Computes spectrogram.
    
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

def plot_random_segments(distances, sac_start_times, sac_end_times = np.array([]), sac_times = np.array([]), 
                         sac_durs=np.array([]), sac_vel_amps=np.array([]), sac_acc_amps=np.array([]), 
                         samplerate=1, num_plots=3, print_info=False):
    '''
    Plots distances from target and saccade start and end times for randomly selected segments.
    Useful as feedback when tuning saccade or hand reach initiation detection.
    Also can print saccade properties (duration, velocity, acceleration). 
    
    Args:
        distances (ntrial): list of time series of distance of eye from target for each segment
        sac_times (ntrial): list of lists containing indices where saccade is occurring for each segment
        sac_start_times (ntrial): list of lists containing indices of saccade start times for each segment
        sac_end_times (ntrial): list of lists, containing indices of saccade end times for each segment
        sac_durs (ntrial): list of lists containing saccade durations for each segment
        sac_vel_amps (ntrial): list of lists containing saccade velocity amplitudes for each segment
        sac_acc_amps (ntrial): list of lists containing saccade acceleration amplitudes for each segment
        samplerate (float): sampling rate of distance time series
        num_plots (int): number of plots
        print_info (bool): whether to print saccade metrics (durations, velocity, acceleration)
        
    Returns:
        None: plots distance to target with saccade starts/ends indicated, and optionally prints saccade metrics
    
    '''
    idx_segments = np.random.randint(0, len(distances), size=num_plots)
    fig, ax = plt.subplots(num_plots, 1, figsize=(14,10))
    for p in range(len(idx_segments)):
        idx_segment = idx_segments[p]
        
        # plot distance to target:
        ax[p].plot(distances[idx_segment])
        ax[p].vlines(sac_start_times[idx_segment], ymin=np.min(distances[idx_segment]), 
                     ymax=np.max(distances[idx_segment]), linestyles='solid', linewidths = 0.5, color='k')
        if len(sac_end_times) > 0:
            ax[p].vlines(sac_end_times[idx_segment], ymin=np.min(distances[idx_segment]), 
                         ymax=np.max(distances[idx_segment]), linestyles='solid', linewidths = 0.5, color='b')
        ax[p].set_title(f"Trial {idx_segment}/{len(distances)}")
        
        if print_info:
            print(f"Num saccades: {len(sac_start_times[idx_segment])}")
            print(f"Saccade times (samples): {sac_times[idx_segment]}")
            print(f"Saccade durations (ms): {sac_durs[idx_segment]/samplerate*1e3}")
            print(f"Saccade velocity amplitudes (cm/s): {sac_vel_amps[idx_segment]}")
            print(f"Saccade acceleration amplitudes (cm/s^2): {sac_acc_amps[idx_segment]}")
            print()
        
    ax[-1].set_xlabel('Time (sample)')
    ax[num_plots//2].set_ylabel('Distance to target (cm)')
    plt.show()

def plot_boxplot_by_label(data, segment_nums, which_saccade, ax):
    '''
    Arrays subset of trial data for boxplotting in groups, and plots the boxplot
    
    Args:
        data (dict): keys is label, value is list of lists of entries
        segment_nums (list): list of indices of segments whose data should be plotted
        ax (plt.Axis): axis to plot boxplot
        
    Returns:
        None: boxplot plotted in appropriate axis
    '''
    
    groups = list(data.keys())
    boxplot_array = np.zeros((len(segment_nums), 3))
    for q in range(len(groups)):
        for i, idx_segment in enumerate(segment_nums):
            boxplot_array[i, q] = data[groups[q]][idx_segment][which_saccade]
    ax.boxplot(boxplot_array, labels = groups)


def plot_cursor_kinematics_around_saccade(cursor_dists, cursor_vels, dist_slope, dist_avg, 
                                          vel_slope, vel_avg, result_segments, selected_result, which_saccade,
                                          num_saccades, segment_conditions, selected_condition, condition_name):
    '''
    Plots cursor position/velocity before/during/after the first saccade of each segment specified
    (Does not plot cursor kinematics from segments without saccades)
    
    Args:
        cursor_dists (dict): value is list of timeseries (one per saccade) for distance of cursor from target
        cursor_vels (dict): value is list of timeseries (one per saccade) for velocity of cursor
        dist_slope (dict): value is change in distance of cursor from target in period, normalized by length of period
        dist_avg (dict): value is average distance of cursor from target in period
        vel_slope (dict): value is change in velocity of cursor in period, normalized by length of period
        vel_avg (dict): value is average velocity of cursor in period
        result_segments (dict): key is trial result, value is segment indices which have the corresponding result
        selected_result (str): trial result by which to select segments for plotting
        which_saccade (int): which saccade out of all saccades in segment to plot cursor kinematics around (0 selects first saccade)
        num_saccades (ntrials): list of number of saccades per segment
        segment_conditions (ntrials): list with indices corresponding to segment condition
        selected_condition (list): list containing conditions which segment must satisfy to be plotted
        condition_name (str): name of condition used to select segments (for title)
        
    Returns:
        None: plots cursor kinematics and boxplots in a new figure
    '''
    
    periods = ['before', 'during', 'after']
    
    # will only plot segments which have specified "selected_condition" stored in "segment_conditions",
    # and have specified "result", and have at least one saccade
    segment_nums = [i for i, x in enumerate(segment_conditions) 
                    if (x in selected_condition) 
                    and (i in result_segments[selected_result])
                    and num_saccades[i] > which_saccade]
    fig, ax = plt.subplots(2,5,figsize=(16,10))
    for idx_segment in segment_nums:
        num_sac = num_saccades[idx_segment]
        for q in range(len(periods)):
            ax[0,q].plot(cursor_dists[periods[q]][idx_segment][which_saccade], '-o', markersize=3)
            ax[1,q].plot(cursor_vels[periods[q]][idx_segment][which_saccade], '-o', markersize=3)
    
    plot_boxplot_by_label(dist_avg, segment_nums, which_saccade, ax[0,3])
    plot_boxplot_by_label(dist_slope, segment_nums, which_saccade, ax[0,4])
    plot_boxplot_by_label(vel_avg, segment_nums, which_saccade, ax[1,3])
    plot_boxplot_by_label(vel_slope, segment_nums, which_saccade, ax[1,4])
    
    ymin = min([ax[0,i].get_ylim()[0] for i in range(3)]); ymax = max([ax[0,i].get_ylim()[1] for i in range(3)])
    for i in range(3):
        ax[0,i].set_ylim(ymin,ymax)
    ax[1,0].set_title('Before saccades')
    ax[1,1].set_title('During saccades')
    ax[0,1].set_xlim(0, 10)
    ax[0,1].set_title(f'{condition_name} {selected_condition}')
    ax[1,2].set_title('After saccades')
    ax[0,0].set_ylabel('Cursor distance to target (cm)')
    ax[0,3].set_title('Avg distance to target (cm)')
    ax[0,3].set_ylim(ymin,ymax)
    ax[0,4].set_title('Avg slope (cm/sample)')
    
    ymin = min([ax[1,i].get_ylim()[0] for i in range(3)]); ymax = max([ax[1,i].get_ylim()[1] for i in range(3)])
    for i in range(3):
        ax[1,i].set_ylim(ymin,ymax)
        ax[1,i].set_xlabel('Time (sample)')
    ax[1,0].set_ylabel('Cursor velocity (cm/s)')
    ax[1,1].set_xlim(0, num_samples_before)
    ax[1,3].set_title('Avg velocity (cm/s)')
    ax[1,3].set_ylim(ymin,ymax)
    ax[1,4].set_title('Avg slope (cm/s/sample)')

def plot_eye_vs_cursor_scatter(eye_data, cursor_data, eye_samplerate=1e3, cursor_samplerate=1e3, ax=None, xlabel='', ylabel='', title=''):
    '''
    Plot scatterplot of eye versus cursor data
    
    Args:
        eye_data (1D array): array of data values for eye
        cursor_data (1D array): array of data values for cursor
        eye_samplerate (float): sampling rate for eye data, in Hz (default 1e3, if no unit conversion desired)
        cursor_samplerate (float): sampling rate for cursor data, in Hz (default 1e3, if no unit conversion desired)
        ax (plt.Axis): axis to plot scatterplot
        xlabel (str): label for scatterplot x-axis
        ylabel (str): label for scatterplot y-axis
        title (str): label for plot title
        
    Returns:
        sc (matplotlib.cm.ScalarMappable): mappable for adding colorbar to plot
    '''
    t = np.reshape(np.arange(len(eye_data)), eye_data.shape)
    sc = ax.scatter(eye_data/eye_samplerate*1e3, cursor_data/cursor_samplerate*1e3, c=t, cmap='Greys') # color points by trial index (darker is later)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(eye_data) >= 2:
        fit_cursor_data, fit_score, pcc, pcc_pvalue, reg_fit = analysis.linear_fit_analysis2D(eye_data, cursor_data)
        ax.plot(eye_data/eye_samplerate*1e3, fit_cursor_data/cursor_samplerate*1e3)
        ax.annotate('CC = %.3f'%pcc, (min(eye_data/cursor_samplerate*1e3),max(cursor_data/cursor_samplerate*1e3)))
    return sc

def plot_event_lines(cur_init_time, sac_start_times, sac_end_times, event_times, event_codes, ymin, ymax, ax):
    '''
    Plots vertical lines at event timepoints
    
    Args:
        cur_init_time (float): timepoint of hand reach initiation, in seconds
        sac_start_times (1D Array): timepoints of saccade starts, in seconds
        sac_end_times (1D Array): timepoints of saccade ends, in seconds
        event_times (1D Array): timepoints of trial events, in seconds
        event_codes (1D Array): codes corresponding to trial events
        ymin (float): lower bound of y axis
        ymax (float): upper bound of y axis
        ax (plt.Axis): axis for vertical lines
        
    Returns:
        None: plots vertical lines on appropriate axis
    '''
    
    ax.vlines(cur_init_time, ymin=ymin, ymax=ymax, linestyles='dashed', linewidths = 0.5, color='k')
    ax.vlines(sac_start_times, ymin=ymin, ymax=ymax, linestyles='solid', linewidths = 0.5, color='r')
    ax.vlines(sac_end_times, ymin=ymin, ymax=ymax, linestyles='solid', linewidths = 0.5, color='b')
    for i in range(len(event_codes)):
        event_code = event_codes[i]
        ax.vlines(event_times[i], ymin=ymin, ymax=ymax, linestyles='solid', 
                     color='k', linewidths = 3, label = f'{event_code}')

def plot_error_angle(cursor_angle, eye_angle, cursor_init_time, saccade_start_times, saccade_end_times, 
                     event_times, event_codes, cursor_samplerate, eye_samplerate, ax):
    '''
    Plots directional error angle of eye and cursor from target for one segment
    
    Args:
        cursor_angle (nt-1): time series of cursor directional error angle
        eye_angle (nt-1): time series of eye directional error angle
        cursor_init_time (int): timepoint of hand reach initiation, in samples
        saccade_start_times (list): timepoints of saccade start times, in samples
        saccade_end_times (list): timepoints of saccade end times, in samples
        event_timestamps (list): timepoints of trial events, in samples
        event_codes (1D Array): codes corresponding to trial events
        cursor_samplerate (float): sampling rate of cursor data, in Hz
        eye_samplerate (float): sampling rate of eye data, in Hz
        ax (plt.Axis): axis for plot
        
    Returns:
        None: plots error angle on appropriate axis
    '''
    
    time_cursor = np.arange(len(cursor_angle)+1)/cursor_samplerate*1e3
    time_eye = np.arange(len(eye_angle)+1)/eye_samplerate*1e3
    
    ax.plot(time_cursor[1:], cursor_angle, label='cursor')
    ax.plot(time_eye[1:], eye_angle, label='eye', linewidth = 0.5)
    
    ymin = min([np.min(cursor_angle), np.min(eye_angle)]); ymax = max([np.max(cursor_angle), np.max(eye_angle)])
    plot_event_lines(cursor_init_time, saccade_start_times, saccade_end_times, event_times, event_codes, ymin, ymax, ax)
    
    ax.set_xlim((time_cursor[0], time_cursor[-1]))
    # ax.set_title(f'Trial result: {get_segment_result(idx_segment)}')
    ax.legend(loc = 'best')
    ax.set_ylabel('Angle bet vel and direction of target (deg)')    

def create_zone_line(zone, samplerate, colors, height, ax):
    '''
    Creates line collection object colored according to zone
    
    Args:
        zone (nt): time series of zone location
        samplerate (float): sampling rate of data, in Hz
        colors (list): list of colors corresponding to zone location
        ax (plt.Axis): axis to plot zone line
        
    Returns:
        tuple: tuple containing:
            |**time (nt):** timepoints of time series
            |**line (matplotlib.collections.Collection):** line collection object (mappable for colorbar)
    '''
    
    time = np.arange(len(zone))/samplerate*1e3
    
    points = np.array([time, height*np.ones(len(zone))]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(range(len(colors)+1)), cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(zone)
    lc.set_linewidth(15)
    lc.set_alpha(0.4)
    line = ax.add_collection(lc)
    
    return time, line    

def plot_zone_lines(cursor_zone, eye_zone, colors, cursor_samplerate, eye_samplerate, ax):
    '''
    Plots cursor and eye lines colored according to zone
    
    Args:
        cursor_zone (nt): time series of zone location of cursor
        eye_zone (nt): time series of zone location of eye
        colors (list): list of colors, equal in length to number of zones
        cursor_samplerate (float): sampling rate of cursor data, in Hz
        eye_samplerate (float): sampling rate of eye data, in Hz
        ax (plt.Axis): axis for zone lines
    
    Returns:
        line (matplotlib.collections.Collection): line collection object (mappable for colorbar)
    '''

    time, line = create_zone_line(cursor_zone, cursor_samplerate, colors, 0, ax)
    _ = create_zone_line(eye_zone, eye_samplerate, colors, 1, ax)
    ax.annotate('Cursor Zone', (0, 0))
    ax.annotate('Eye Zone', (0, 1))
    ax.set_xlim((time[0], time[-1]))
    ax.set_ylim((-0.5, 1.5))
    ax.set_yticks([])
    
    return line

def plot_distance_from_target(cursor_dist, eye_dist, cursor_init_time, saccade_start_times, saccade_end_times, 
                              event_times, event_codes, cursor_samplerate, eye_samplerate, ax):
    '''
    Plots distance of eye and cursor from target for one segment
    
    Args:
        cursor_dist (nt): time series of cursor distance from target
        eye_dist (nt): time series of eye distance from target
        cursor_init_times (int): timepoint of hand reach initiation, in samples
        saccade_start_times (list): timepoints of saccade start times, in samples
        saccade_end_times (list): timepoints of saccade end times, in samples
        event_timestamps (list): timepoints of trial events, in samples
        event_codes (1D Array): codes corresponding to trial events
        cursor_samplerate (float): sampling rate of cursor data, in Hz
        eye_samplerate (float): sampling rate of eye data, in Hz
        ax (plt.Axis): axis for plot
        
    Returns:
        None: plots error angle on appropriate axis
    '''
    
    time_cursor = np.arange(len(cursor_dist))/cursor_samplerate*1e3
    time_eye = np.arange(len(eye_dist))/eye_samplerate*1e3
    
    ax.plot(time_cursor, cursor_dist, label='cursor')
    ax.plot(time_eye, eye_dist, label='eye')
    
    ymin = min([np.min(cursor_dist), np.min(eye_dist)]); ymax = max([np.max(cursor_dist), np.max(eye_dist)])
    plot_event_lines(cursor_init_time, saccade_start_times, saccade_end_times, event_times, event_codes, ymin, ymax, ax)
    
    ax.set_xlim((time_cursor[0], time_cursor[-1]))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Dist to target (cm)')
    ax.legend(loc = 'best')

def plot_window_around_event(cursor_angle, eye_angle, event_times, which_event, num_samples_before, num_samples_after, 
                             segment_conditions, selected_condition, condition_name, result_segments, selected_result):
    '''
    Plot cursor and eye directional error in window around a specified event, averaged across selected segments
    
    Args:
        cursor_angle (nt-1): time series of cursor directional error angle
        eye_angle (nt-1): time series of eye directional error angle
        event_times (ntrials): list of indices of event occurrence, for each segment
        which_event (int): index of event in segment (e.g., 1 selects the second event in a segment)
        num_samples_before (int): number of samples to plot before event
        num_samples_after (int): number of samples to plot after event
        segment_conditions (ntrials): list with indices corresponding to segment condition
        selected_condition (list): list containing conditions which segment must satisfy to be plotted
        condition_name (str): name of condition used to select segments (for title)
        result_segments (dict): key is trial result, value is segment indices which have the corresponding result
        selected_result (str): trial result by which to select segments for plotting
    
    Returns:
        None: plots average cursor and eye error in a new figure
    '''
    segment_nums = [i for i, x in enumerate(segment_conditions) 
                    if (x in selected_condition) and (i in result_segments[selected_result])]

    # inspect average directional error for all segments with at least one event where window is within bounds
    segment_nums = [i for i, x in enumerate(event_times) if i in segment_nums and len(x) > 0 
                          and x[which_event] + num_samples_after < len(cursor_angle[i]) 
                          and x[which_event] >= num_samples_before]
    
    # extract interval around saccade start
    window_length = num_samples_before + num_samples_after + 1
    cursor_errors = np.zeros((len(segment_nums), window_length))
    eye_errors = np.zeros((len(segment_nums), window_length))
    for j, idx_segment in enumerate(segment_nums):
        event_time = event_times[idx_segment][which_event]
        cursor_errors[j] = cursor_angle[idx_segment][(event_time-num_samples_before):(event_time+num_samples_after+1)]
        eye_errors[j] = eye_angle[idx_segment][(event_time-num_samples_before):(event_time+num_samples_after+1)]

    mean_cursor_error = np.mean(cursor_errors, axis=0)
    mean_eye_error = np.mean(eye_errors, axis=0)
    sd_cursor_error = np.std(cursor_errors, axis=0)
    sd_eye_error = np.std(eye_errors, axis=0)

    window = np.arange(-num_samples_before, num_samples_after+1) 
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.plot(window, mean_cursor_error, label='cursor')
    ax.fill_between(window, mean_cursor_error-sd_cursor_error, mean_cursor_error+sd_cursor_error, alpha=0.1)
    ax.plot(window, mean_eye_error, label='eye')
    ax.fill_between(window, mean_eye_error-sd_eye_error, mean_eye_error+sd_eye_error, alpha=0.1)
    ax.vlines(0, ymin=0, ymax=180, linestyles='dashed')
    ax.set_xlabel('Relative time (samples)')
    ax.set_ylabel('Directional error (deg)')
    ax.set_title(f'{condition_name} {selected_condition}, Trial result: {selected_result}, Number of trials: {len(segment_nums)}')
    ax.legend()
    plt.show()

def plot_zone_heatmap(data, event_times, which_event, segment_conditions, selected_condition, condition_name, 
                      result_segments, selected_result, samplerate, order='', segment_order=[], 
                      plot_cbar=True, with_title=True, fig=None, ax=None):
    '''
    Plots heatmap of colored zone locations for all segments stacked
    
    Args:
        data (nseg): list of time series (one per segment) of zone locations
        event_times (ntrials): list of indices of event occurrence, for each segment
        which_event (int): index of event in segment (e.g., 1 selects the second event in a segment)
        segment_conditions (ntrials): list with indices corresponding to segment condition
        selected_condition (list): list containing conditions which segment must satisfy to be plotted
        condition_name (str): name of condition used to select segments (for title)
        result_segments (dict): key is trial result, value is segment indices which have the corresponding result
        selected_result (str): trial result by which to select segments for plotting
        samplerate (float): sampling rate of data, in Hz
        order (str): method by which to sort or cluster zone data across segments (options: 'duration', 'foveation')
        segment_order (list): order of segment indices (from previous sorting/clustering) by which to stack segments
        plot_cbar (bool): whether to plot with a colorbar
        with_title (bool): whether to plot with a title
        fig (plt.figure): figure for title
        ax (plt.Axis): axis for heatmap
        
    Returns:
        segment_nums (list): order of segment indices by which segments were stacked
    '''
    
    if len(segment_order) == 0:
        segment_nums = np.array([i for i, x in enumerate(segment_conditions) 
                        if (x in selected_condition) 
                        and (i in result_segments[selected_result])
                        and len(event_times[i]) > which_event])
        if order == 'duration': # order segments by duration
            segment_durations = [len(x) for i, x in enumerate(data) if i in segment_nums]
            segment_nums = segment_nums[np.argsort(segment_durations)]
        if order == 'foveation': # cluster segments by foveation or no foveation
            segments_without_foveation = np.array([i for i in segment_nums if 2 not in data[i]])
            segments_with_foveation = np.array([i for i in segment_nums if 2 in data[i]])
            segment_nums = np.concatenate((segments_without_foveation, segments_with_foveation))
    else:
        segment_nums = segment_order
    
    max_time_before_event = max([x[which_event] for i, x in enumerate(event_times) if i in segment_nums])
    max_time_after_event = max([len(data[i]) - x[which_event] for i, x in enumerate(event_times) if i in segment_nums])
    max_length = max([len(x) for i, x in enumerate(data) if i in segment_nums])
    heatmap_array = np.empty((len(segment_nums), max_time_before_event+max_time_after_event))
    heatmap_array[:] = np.nan
    for j, idx_segment in enumerate(segment_nums):
        event_time = event_times[idx_segment][which_event]
        data_segment = data[idx_segment]
        data_start = max_time_before_event - event_time
        data_end = len(data_segment) + data_start
        heatmap_array[j, data_start:data_end] = data_segment
        
    xgrid = (np.arange(heatmap_array.shape[1]) - max_time_before_event)/samplerate*1e3
    ygrid = np.arange(len(segment_nums))
    cmap = ListedColormap(color_list)
    cmap.set_bad(color='black')
    norm = BoundaryNorm(list(range(len(zone_str)+1)), cmap.N)
    pcm = ax.pcolormesh(xgrid, ygrid, heatmap_array, cmap=cmap, norm=norm)
    if plot_cbar:
        cbar = plt.colorbar(pcm, orientation = 'vertical', label='Zone')
        cbar.set_ticks(np.arange(len(zone_str))+0.5)
        cbar.set_ticklabels(zone_str)
    ax.vlines(0, ymin=0, ymax=len(segment_nums)-1, linestyles='dashed')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial')
    if with_title:
        fig.suptitle(f'{condition_name} {selected_condition}, Trial selected_result: {selected_result}, Number of trials: {len(segment_nums)}', y=0.93)
    
    return segment_nums