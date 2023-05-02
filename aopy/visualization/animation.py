from matplotlib.animation import FuncAnimation
import os
import matplotlib.pyplot as plt
import numpy as np
from .base import plot_spatial_map, color_trajectories, plot_circles, plot_targets

from .. import postproc


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


def showanim(animation, closeanim=True):
    '''
    Display an animation in a python notebook

    Args:
        animation (pyplot.Animation): animation to display
        closeanim (bool, optional): also close the animation figure to avoid showing a static plot
    '''
    from IPython import display # not a required package
    html = display.HTML(animation.to_html5_video())
    display.display(html)
    if closeanim:
        plt.close()

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

def animate_cursor_eye(cursor_trajectory, eye_trajectory, samplerate, target_positions, target_radius, 
                       bounds, cursor_radius=0.5, eye_radius=0.25,
                       cursor_color='blue', eye_color='purple'):
    '''
    Draws an animation of two trajectories with static targets. The colors and endpoint radii of the
    two trajectories can be specified along with the position and radius of the targets. Targets are
    colored automatically according to :func:`~aopy.visualization.plot_targets`.

    Example:

        .. raw:: html

            <video controls src="_static/test_anim.mp4"></video>

    Args:
        cursor_trajectory ((nt, ndim) array): Cursor positions over time for 2D or 3D trajectories.
        eye_trajectory ((nt, ndim) array): Eye positions over time for 2D or 3D trajectories.
        samplerate (float): The sampling rate of the trajectories in Hz.
        target_positions ((ntargets, ndim) array): Array of target positions for 2D or 3D targets.
        target_radius (float): Radius of the targets.
        bounds (tuple): Boundaries of the plot area. See :func:`~aopy.visualization.plot_targets`.
        cursor_radius (float, optional): Radius of the cursor endpoint. Default is 0.5.
        eye_radius (float, optional): Radius of the eye endpoint. Default is 0.25.
        cursor_color (plt.color, optional): Color of the cursor trajectory. Default is 'blue'.
        eye_color (plt.color, optional): Color of the eye trajectory. Default is 'purple'.

    Returns:
        None
    '''
    assert len(cursor_trajectory) == len(eye_trajectory), "Cursor and Eye trajectories must have the same length"

    def plotdata(i):
        cur.center = cursor_trajectory[i]
        eye.center = eye_trajectory[i]
        cur_line.set_data(*cursor_trajectory[:i+1].T)
        eye_line.set_data(*eye_trajectory[:i+1].T)
        return ax
            
    # # Initial plot
    fig, ax = plt.subplots(1, 1)
    plot_targets(target_positions, target_radius, bounds=bounds, ax=ax)
    cur = plt.Circle(cursor_trajectory[0], radius=cursor_radius, alpha=0.5, color=cursor_color)
    eye = plt.Circle(eye_trajectory[0], radius=eye_radius, alpha=0.5, color=eye_color)
    ax.add_artist(cur)
    ax.add_artist(eye)
    cur_line, = plt.plot(*cursor_trajectory[:1].T, color=cursor_color)
    eye_line, = plt.plot(*eye_trajectory[:1].T, color=eye_color)

    # Create animation
    ani = FuncAnimation(fig, plotdata, 
                        frames=len(cursor_trajectory), interval=1000./samplerate)  
    return ani
