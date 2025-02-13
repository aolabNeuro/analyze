# bmi3d.py
#
# visuzalition specific to BMI3D 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import base
from .. import data as aodata

def plot_decoder_weight_matrix(decoder, ax=None):
    """
    Plot the decoder weight matrix. Compatible with Decoder objects with 
    KFDecoder and lindecoder filters.

    Args:
        decoder (riglib.bmi.Decoder): The decoder object from BMI3D.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if hasattr(decoder.filt, 'unit_to_state'):
        matrix = np.array(decoder.filt.unit_to_state)
    elif hasattr(decoder.filt, 'C'):
        matrix = np.array(decoder.filt.C.T)
    else:
        raise ValueError("Decoder does not have a recognizable weight matrix attribute.")

    max_weight = np.max(abs(matrix))
    im = ax.pcolor(matrix, cmap='bwr')
    ax.set_xticks(np.arange(decoder.n_units))
    ax.set_xticklabels([decoder.units[iu][0] for iu in range(decoder.n_units)], rotation=45)
    ax.set_yticks(np.arange(decoder.n_states) + .5)
    ax.set_yticklabels(decoder.states)
    ax.set(xlabel='Readout unit', ylabel='State')
    plt.colorbar(im, ax=ax, label='Weight')
    im.set_clim(-max_weight, max_weight)
    
def plot_readout_map(decoder, readouts, drive_type='ECoG244', cmap='YlGnBu', ax=None):
    """
    Plot the spatial location of readouts.

    Args:
        decoder (riglib.bmi.Decoder): The decoder object from BMI3D.
        readouts (list): The readout channels.
        drive_type (str, optional): The type of drive. See :func:`~aopy.data.load_chmap` for options. Defaults to 'ECoG244'.
        cmap (str, optional): The colormap to use. Defaults to 'YlGnBu'.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    elec_pos, acq_ch, elecs = aodata.load_chmap(drive_type=drive_type)
    channels = np.nan * np.zeros(np.max(acq_ch) + 1)
    channels[decoder.channels - 1] = np.arange(len(decoder.channels))
    base.plot_spatial_drive_map(channels, interp=False, drive_type=drive_type, cmap=cmap, ax=ax, nan_color='#FF000000')
    base.annotate_spatial_map_channels(drive_type=drive_type, color='k', ax=ax)
    base.annotate_spatial_map_channels(acq_ch=readouts, drive_type=drive_type, color='w', ax=ax)

def plot_decoder_weight_vectors(decoder, x_idx, y_idx, colors, ax=None):
    """
    Plot decoder weight vectors.

    Args:
        decoder (riglib.bmi.Decoder): The decoder object from BMI3D.
        x_idx (int): The index for the x state in the decoder's weight matrix
        y_idx (int): The index for the y state in the decoder's weight matrix
        colors (list): List of colors for the vectors.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if hasattr(decoder.filt, 'unit_to_state'):
        matrix = np.array(decoder.filt.unit_to_state)
    elif hasattr(decoder.filt, 'C'):
        matrix = np.array(decoder.filt.C.T)
    else:
        raise ValueError("Decoder does not have a recognizable weight matrix attribute.")
    
    max_len = np.max(np.linalg.norm(matrix[[x_idx,y_idx],:], axis=0))
    if max_len == 0:
        max_len = 1.
    nch = matrix.shape[1]
    ax.quiver(np.zeros(nch), np.zeros(nch), matrix[x_idx, :], matrix[y_idx, :], width=.01, 
                 color=colors, alpha=0.5, angles='xy', scale_units='xy', scale=1.)
    ax.set(xlabel='Px weight', ylabel='Py weight', 
           xlim=(-max_len,max_len), ylim=(-max_len,max_len))

def plot_decoder_summary(decoder, drive_type='ECoG244', cmap='YlGnBu'):
    """
    Plot a summary of the decoder weight matrix, readout map, and weight vectors.

    Example:

        A KF decoder with 7 states and 16 readout channels.

        .. image:: _images/decoder_weights.png

    Args:
        decoder (riglib.bmi.Decoder): The decoder object from BMI3D.
        drive_type (str, optional): The type of drive. See :func:`~aopy.data.load_chmap` for options. Defaults to 'ECoG244'.
        cmap (str, optional): The colormap to use. Defaults to 'YlGnBu'.
    """
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.flatten()

    plot_decoder_weight_matrix(decoder, ax=ax[0])
    ax[0].set_title('Weight matrix')
    plot_readout_map(decoder, decoder.channels, drive_type=drive_type, 
                     cmap=cmap, ax=ax[1])
    ax[1].set_title('Readout location')

    colors = sns.color_palette(cmap, n_colors=len(decoder.channels))
    plot_decoder_weight_vectors(decoder, 0, 2, colors, ax=ax[2])
    ax[2].set_title('Pos weight vectors')
    plot_decoder_weight_vectors(decoder, 3, 5, colors, ax=ax[3])
    ax[3].set_title('Vel weight vectors')
    fig.tight_layout()