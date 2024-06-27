
def plot_decoder_weights(decoder):
    '''
    Plot unit weight map

    '''
    fig, ax = plt.subplots(1,3,figsize=(14,3.5), width_ratios=(2,1,1))
    cmap = ax[0].pcolor(decoder.filt.unit_to_state, cmap='bwr')
    ax[0].set_xticks(np.arange(decoder.n_units), [decoder.units[iu][0] for iu in range(decoder.n_units)], rotation=30)
    ax[0].set_yticks(np.arange(decoder.n_states)+.5, decoder.states)
    ax[0].set(xlabel='Readout unit', ylabel='State', title=f"{df['date'][0]} - {np.array(df['date'])[-1]}")
    cb = plt.colorbar(cmap, label='Weight')

    for iunit, unit in enumerate(decoder.units):
        ax[1].arrow(0,0, decoder.filt.unit_to_state[3,iunit], decoder.filt.unit_to_state[5,iunit], width=.03, color='k')

    ax[1].set(xlabel='Vx weight', ylabel='Vy weight', xlim=(-1,1), ylim=(-1,1), title=f"{df['date'][0]} - {np.array(df['date'])[-1]}")

    # Plot Spatial location of readouts
    channels = np.zeros(256)
    channels[readouts-1] = 1 # subtracting 1 to convert acq_ch into python index

    romap = aopy.visualization.plot_ECoG244_data_map(channels, interp=False, cmap='Reds', ax=ax[2])
    aopy.visualization.annotate_spatial_map_channels(color='k', ax=ax[2])
    aopy.visualization.annotate_spatial_map_channels(acq_ch=readouts, color='w', ax=ax[2])
    ax[2].set_title('Readout Locations')

    fig.tight_layout()
    plt.show()