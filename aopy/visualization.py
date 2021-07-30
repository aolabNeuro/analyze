# visualization.py
# code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates

from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
import numpy as np
import os
import copy

import pandas as pd

from . import postproc


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


def plot_freq_domain_power(data, samplerate, ax=None):
    '''
    Plots a power spectrum of each channel on the given axis
    Args:
        data (nt, nch): timeseries data, can also be a single channel vector
    '''
    time = np.arange(np.shape(data)[0])/samplerate
    for ch in range(np.shape(data)[1]):
        ax.plot(time, data[:,ch]*1e6) # convert to microvolts
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (uV)')

def plot_freq_domain_amplitude(data, samplerate, ax=None, rms=False):
    '''
    Plots a amplitude spectrum of each channel on the given axis

    Args:
        data (nt, nch): timeseries data in volts, can also be a single channel vector
        samplerate (float): sampling rate of the data
        ax (pyplot axis, optional): where to plot
        rms (bool, optional): compute root-mean square amplitude instead of peak amplitude
    '''
    if np.ndim(data) < 2:
        data = np.expand_dims(data, 1)
    if ax is None:
        ax = plt.gca()
    freq_data = np.fft.fft(data, axis=0)
    length = np.shape(freq_data)[0]
    freq = np.fft.fftfreq(length, d=1./samplerate)
    data_ampl = abs(freq_data[freq>=0,:])*2/length # compute the one-sided amplitude
    if rms:
        data_ampl[1:] = data_ampl[1:]/np.sqrt(2)
    non_negative_freq = freq[freq>=0]
    for ch in range(np.shape(freq_data)[1]):
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

    x_spacing = (extent[1]-extent[0])/(grid_size[0]-1)
    y_spacing = (extent[3]-extent[2])/(grid_size[1]-1)
    xy = np.vstack((x_pos, y_pos)).T
    xq, yq = np.meshgrid(np.arange(extent[0],x_spacing*grid_size[0],x_spacing), np.arange(extent[2],y_spacing*grid_size[1],y_spacing))
    X = griddata(xy, data, (np.reshape(xq,-1), np.reshape(yq,-1)), method=interp_method, rescale=False)

    # Construct kd-tree, functionality copied from scipy.interpolate
    tree = cKDTree(xy)
    xi = _ndim_coords_from_arrays((np.reshape(xq,-1), np.reshape(yq,-1)))

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

    x_spacing = (extent[1]-extent[0])/(data_map.shape[0]-1)
    y_spacing = (extent[3]-extent[2])/(data_map.shape[1]-1)
    extent = np.add(extent, [-x_spacing/2, x_spacing/2, -y_spacing/2, y_spacing/2])

    # Set the 'bad' color to something different
    cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
    cmap.set_bad(color='black')

    # Plot
    if ax is None:
        ax = plt.gca()
    ax.imshow(data_map, cmap=cmap, origin='lower', extent=extent)
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')


def plot_raster(data, plot_cue, cue_bin, ax):
    '''
       Create a raster plot of neural data

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

    color_palette = sns.set_palette("Accent", n_neurons) #TODO: make this into a separate function
    for n in range(n_neurons):  # set this to 1 if we need rastor plot for only one neuron
        for tr in range(n_trial):
            for t in range(n_bins):
                if data[n, tr, t] == 1:
                    x1 = [tr, tr + 1]
                    x2 = [t, t]
                    ax.plot(x2, x1, color=color_palette(n))
    if plot_cue:
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
    from matplotlib.animation import FFMpegFileWriter # requires ffmpeg
    filepath = os.path.join(base_dir, filename)
    writer = FFMpegFileWriter()
    animation.save(filepath, dpi=dpi, writer=writer, savefig_kwargs=savefig_kwargs)

def showanim(animation):
    '''
    Display an animation in a python notebook

    Args:
        animation (pyplot.Animation): animation to display
    '''
    from IPython.display import HTML # not a required package
    HTML(animation.to_html5_video())

def animate_events(events, times, fps, xy=(0.3,0.3), fontsize=30, color='g'):
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
        display = names[events[num,:] == 1]
        if len(display) > 0:
            note.set_text(display[0]) # note if simultaneous events occur, we just print the first

    fig, ax = plt.subplots(1,1)
    note = ax.annotate("", xy, fontsize=fontsize, color=color)
    plt.axis('off')
    return FuncAnimation(fig, display_text, frames=frame_events.shape[0], 
			             interval=round(1000/fps), 
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
    
    line, = ax.plot(trajectory[0,0], trajectory[0,1], trajectory[0,2], color=color)
    
    ax.set_xlim((np.nanmin(trajectory[:,0]), np.nanmax(trajectory[:,0])))
    ax.set_xlabel(axis_labels[0])

    ax.set_ylim((np.nanmin(trajectory[:,1]), np.nanmax(trajectory[:,1])))
    ax.set_ylabel(axis_labels[1])
    
    ax.set_zlim((np.nanmin(trajectory[:,2]), np.nanmax(trajectory[:,2])))
    ax.set_zlabel(axis_labels[2])
    
    def draw(num):
        length = min(num, history)
        start = num-length
        line.set_data(trajectory[start:num,0], trajectory[start:num,1])
        line.set_3d_properties(trajectory[start:num,2])
        return line,
        
    return FuncAnimation(fig, draw, frames=trajectory.shape[0],
                         init_func=lambda : None, interval=1000./samplerate)

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
        ax.set(xlim=(1.1*bounds[0], 1.1*bounds[1]), 
               ylim=(1.1*bounds[2], 1.1*bounds[3]),
               zlim=(1.1*bounds[4], 1.1*bounds[5]))
    except:
        ax.set(xlim=(1.1*bounds[0], 1.1*bounds[1]), 
               ylim=(1.1*bounds[2], 1.1*bounds[3]))

def plot_targets(target_positions, target_radius, bounds=None, origin=(0,0,0), ax=None):
    '''
    Add targets to an axis. If any targets are at the origin, they will appear 
    in a different color (magenta). Works for 2D and 3D axes

    Args:
        target_positions (ntarg, 3): array of target (x, y, z) locations
        target_radius (float): radius of each target
        bounds (tuple, optional): 6-element tuple describing (-x, x, -y, y, -z, z) cursor bounds
        origin (tuple, optional): (x, y, z) position of the origin
        ax (plt.Axis, optional): axis to plot the targets on
    '''
    if ax is None:
        ax = plt.gca()

    for i in range(0,target_positions.shape[0]):

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
            ax.plot_surface(x, y, z, alpha=0.5, color=target_color)
            ax.set_box_aspect((1,1,1))
        except:
            target = plt.Circle((pos[0], pos[1]), 
                            radius=target_radius, alpha=0.5, color=target_color)
            ax.add_artist(target)
            ax.set_aspect('equal', adjustable='box')
    if bounds is not None: set_bounds(bounds, ax)
    

def plot_trajectories(trajectories, bounds=None, ax=None):
    '''
    Draws the given trajectories, one at a time in different colors. Works for 2D and 3D axes

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
        ax.set_box_aspect((1,1,1))
    except:
        for path in trajectories:
            ax.plot(path[:,0], path[:,1])
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

        .. image:: _images/colored_trajectories.png

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