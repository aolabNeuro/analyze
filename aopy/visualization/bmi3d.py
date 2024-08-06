import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import aopy.visualization

def plot_decoder_weight_matrix(decoder, ax=None):
    """
    Plot the decoder weight matrix.

    Args:
        matrix (np.ndarray): The weight matrix to plot.
        decoder (object): The decoder object containing unit and state information.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if hasattr(decoder.filt, 'unit_to_state'):
        matrix = decoder.filt.unit_to_state
    elif hasattr(decoder.filt, 'C'):
        matrix = decoder.filt.C.T
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
        decoder (object): The decoder object containing channel information.
        readouts (list): The readout channels.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    channels = np.nan * np.zeros(256)
    channels[decoder.channels - 1] = np.arange(len(decoder.channels))
    aopy.visualization.plot_ECoG244_data_map(channels, interp=False, drive_type=drive_type, cmap=cmap, ax=ax, nan_color='#FF000000')
    aopy.visualization.annotate_spatial_map_channels(color='k', ax=ax)
    aopy.visualization.annotate_spatial_map_channels(acq_ch=readouts, color='w', ax=ax)

def plot_decoder_weight_vectors(decoder, x_idx, y_idx, colors, ax=None):
    """
    Plot decoder weight vectors.

    Args:
        decoder (object): The decoder object containing unit information.
        matrix (np.ndarray): The weight matrix.
        x_idx (int): The index for the x dimension of the vector.
        y_idx (int): The index for the y dimension of the vector.
        colors (list): List of colors for the vectors.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if hasattr(decoder.filt, 'unit_to_state'):
        matrix = decoder.filt.unit_to_state
    elif hasattr(decoder.filt, 'C'):
        matrix = decoder.filt.C.T
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

def plot_decoder_weights(decoder):
    """
    Plot the decoder weights and spatial locations.

    Args:
        decoder (object): The decoder object containing weight and channel information.
    """
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.flatten()

    plot_decoder_weight_matrix(decoder, ax=ax[0])
    ax[0].set_title('Weight matrix')
    plot_readout_map(decoder, decoder.channels, ax=ax[1])
    ax[1].set_title('Readout location')

    colors = sns.color_palette('YlGnBu', n_colors=len(decoder.channels))
    plot_decoder_weight_vectors(decoder, 0, 2, colors, ax=ax[2])
    ax[2].set_title('Pos weight vectors')
    plot_decoder_weight_vectors(decoder, 3, 5, colors, ax=ax[3])
    ax[3].set_title('Vel weight vectors')
    fig.tight_layout()