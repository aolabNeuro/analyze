# visualization.py
# code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
import numpy as np
import os
import copy

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
        ax.plot(time, data[:,ch]*1e6)
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
    freq = np.fft.fftfreq(length, d=1./samplerate)
    data_ampl = abs(freq_data[freq>1,:])*2/length
    non_negative_freq = freq[freq>1]
    for ch in range(np.shape(freq_data)[1]):
        ax.semilogx(non_negative_freq, data_ampl[:,ch]*1e4)
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

def sample_events(events, times, samplerate):
    '''
    Converts a list of events and timestamps to a matrix of events where
    each column is a different event and each row is a sample in time

    Args:
        events (list): list of event names or numbers
        times (list): list of timestamps for each event
        samplerate (float): rate at which you want to sample the events

    Returns:
        tuple: tuple containing:
            frame_events (nt, n_events): boolean matrix of when each event occurred
            event_names (n_events): list of event column names

    '''
    n_samples = round(times[-1]*samplerate) + 1
    unique_events = np.unique(events)
    frame_events = np.zeros((n_samples, len(unique_events)), dtype='bool')
    for idx_event in range(len(events)):
        unique_idx = unique_events == events[idx_event]
        event_time = times[idx_event]
        event_frame = round(event_time * samplerate)
        frame_events[event_frame,unique_idx] = True
        
    return frame_events, unique_events

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
    frame_events, event_names = sample_events(events, times, fps)

    def display_text(num, events, names, note):
        display = names[events[num,:] == 1]
        if len(display) > 0:
            note.set_text(display[0]) # note if simultaneous events occur, we just print the first

    fig, ax = plt.subplots(1,1)
    note = ax.annotate("", xy, fontsize=fontsize, color=color)
    plt.axis('off')
    return FuncAnimation(fig, display_text, frames=frame_events.shape[0],
                         init_func=lambda : None,
                         fargs=(frame_events, event_names, note))

def animate_trajectory_3d(trajectory, samplerate, history=1000, color='b'):
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
    ax.set_xlabel('x = Right')

    ax.set_ylim((np.nanmin(trajectory[:,1]), np.nanmax(trajectory[:,1])))
    ax.set_ylabel('y = Forwards')
    
    ax.set_zlim((np.nanmin(trajectory[:,2]), np.nanmax(trajectory[:,2])))
    ax.set_zlabel('z = Up')
    
    def draw(num):
        length = min(num, history)
        start = num-length
        line.set_data(trajectory[start:num,0], trajectory[start:num,1])
        line.set_3d_properties(trajectory[start:num,2])
        return line,
        
    return FuncAnimation(fig, draw, frames=trajectory.shape[0],
                         init_func=lambda : None, interval=1000./samplerate)
