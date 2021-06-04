# visualization.py
# code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)

import matplotlib
import matplotlib.pyplot as plt
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
        data (nt, nch): timeseries data in volts, can also be a single channel vector
        samplerate (float): sampling rate of the data
        ax (pyplot axis, optional): where to plot
    '''
    if np.ndim(data) < 2:
        data = np.expand_dims(data, 1)
    if ax is None:
        ax = plt.gca()
    time = np.arange(np.shape(data)[0])/samplerate
    for ch in range(np.shape(data)[1]):
        ax.plot(time, data[:,ch]*1e6) # convert to microvolts
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (uV)')

def plot_freq_domain_power(data, samplerate, ax=None):
    '''
    Plots a power spectrum of each channel on the given axis

    Args:
        data (nt, nch): timeseries data in volts, can also be a single channel vector
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
    data_ampl = abs(freq_data[freq>=0,:])*2/length # compute the one-sided amplitude
    non_negative_freq = freq[freq>=0]
    for ch in range(np.shape(freq_data)[1]):
        ax.semilogx(non_negative_freq, data_ampl[:,ch]*1e6) # convert to microvolts
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (uV)')

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
