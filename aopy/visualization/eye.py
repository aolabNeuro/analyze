# eye.py
#
# visuzalition specific to eye tracking data

import numpy as np
import matplotlib.pyplot as plt

from .. import postproc

def plot_eye_calibration_result(eye_calibration_data, cursor_calibration_data, coefficients, correlation_coeff,
                                eye_labels=['LE x', 'LE y', 'RE x', 'RE y'], bounds=[-10, 10, -10, 10]):
    '''
    Plot the results of eye calibration.

    Args:
        eye_calibration_data (nt, nch): The eye calibration data.
        cursor_calibration_data (nt, 2): The cursor calibration data.
        coefficients (nch): The coefficients for eye calibration.
        correlation_coeff (float): The correlation coefficient.
        eye_labels (list, optional): The labels for each eye. Defaults to ['LE x', 'LE y', 'RE x', 'RE y'].
        bounds (list, optional): The bounds for the plot. Defaults to [-10, 10, -10, 10].

    Examples:

        Beignet center-out task (5974) from 2022-07-01:
        .. image:: _images/eye_calibration.png
    '''
    
    estimated_calibration_pos = postproc.get_calibrated_eye_data(eye_calibration_data, coefficients)

    nch = eye_calibration_data.shape[1]
    fig = plt.figure(figsize=(nch*3,6))
    gs = fig.add_gridspec(2,nch)
    for i in range(nch):
        ax = fig.add_subplot(gs[0,i])
        eye_label = eye_labels[i]
        cursor_label = ['cursor x', 'cursor y'][i % 2]
        
        ax.set_title(f'{eye_label} vs. {cursor_label}')
        ax.scatter(cursor_calibration_data[:,i], eye_calibration_data[:,i], color='b')
        ax.plot(estimated_calibration_pos[:,i], eye_calibration_data[:,i], 'k')
        ax.text(0.6, 0.9, f'$R^2: {correlation_coeff[i]**2:.4f}$', transform=ax.transAxes)
        ax.text(0.05, 0.05, f' \n $y = {coefficients[i,0]:.2f} * x - {coefficients[i,1]:.2f}$ ', transform=ax.transAxes)
        ax.set_ylabel("Eye raw pos.")
        ax.set_xlabel('Cursor pos. (cm)')

    ax = fig.add_subplot(gs[1,:nch//2])
    ax.scatter(cursor_calibration_data[:,0], cursor_calibration_data[:,1], color='g', label='cursor')
    ax.scatter(estimated_calibration_pos[:,0], estimated_calibration_pos[:,1], color='m', label='eye')
    ax.set_title('LE calibration')
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')
    handles, labels = ax.get_legend_handles_labels()
    unique = [h for i, h in enumerate(handles) if h not in handles[:i]]
    plt.legend(unique, ['cursor', 'eye'], loc='upper left', bbox_to_anchor=(1,1))
    
    ax = fig.add_subplot(gs[1,nch//2:])
    ax.scatter(cursor_calibration_data[:,0], cursor_calibration_data[:,1], color='g', label='cursor')
    ax.scatter(estimated_calibration_pos[:,2], estimated_calibration_pos[:,3], color='m', label='eye')
    ax.set_title('RE calibration')
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')
    handles, labels = ax.get_legend_handles_labels()
    unique = [h for i, h in enumerate(handles) if h not in handles[:i]]
    plt.legend(unique, ['cursor', 'eye'], loc='upper left', bbox_to_anchor=(1,1))

    plt.tight_layout()