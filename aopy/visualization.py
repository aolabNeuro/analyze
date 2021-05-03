# visualization.py
# code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)
import seaborn as sns
import numpy as np

def plot_rastor(data,plot_cue, cue_bin, ax):
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
    for n in range(n_neurons):  # set this to 1 if we need rastor plot for only one neurons
        for tr in range(n_trial):
            for t in range(n_bins):
                if data[n, tr, t] == 1:
                    x1 = [tr, tr + 1]
                    x2 = [t, t]
                    ax.plot(x2, x1, color=color_palette[n])
    if plot_cue:
        ax.axvline(x=cue_bin, linewidth=2.5, color='r')

def plot_psth(data, cue_bin, ax):
    '''
       Create a peristimulus histogram for neural data

       Args:
           data (n_trials, n_neurons, n_timebins): neural spiking data (not spike count- must contain only 0 or 1) in the form of a three dimensional matrix
           plot_cue : If plot_cue is true, a vertical line showing when this event happens is plotted in the rastor plot
           cue_bin : time bin at which an event occurs. For example: Go Cue or Leave center
            ax: axis to plot rastor plot
       Returns:
           rastor plot in appropriate axis
    '''