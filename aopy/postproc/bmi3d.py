# bmi3d.py 
# postprocessing specific to BMI3D data

import numpy as np
from scipy.spatial.transform import Rotation as R

def covert_input_to_world_coords(manual_input, rotation, offset, scale=1):
    '''
    Transforms manual input to centered world coordinates for BMI3D tasks. For example, for 
    optitrack input, raw coordinates are in the form (x: forward/backward, y: up/down, z: right/left).
    This function applies the BMI3D offset and scale to the coordinates, then transforms the coordinates 
    to world coordinates (x: right/left, y: forward/backward, z: up/down). For joystick input, the
    coordinates are in the form (x: left/right, y: backward/forward, z: nothing) and the scale is -1. 
    Thus the output is always in the form (x: right/left, y: forward/backward, z: up/down) if the
    inputs are copied directly from exp_metadata.

    Args:
        manual_input (nt, 3): manual input coordinates from bmi3d, e.g. exp_data['task']['manual_input']
        rotation (str): rotation metadata from exp_metadata['rotation']
        offset (3-tuple): x,y,z offset for the manual input from exp_metadata['offset']
        scale (float, optional): scaling factor for cursor movement from exp_metadata['scale']. Default 1.

    Returns:
        (nt, 3): manual input in world coordinates

    Examples:
        Load manual input data from an experiment and transform it to world coordinates using the 
        metadata from the experiment. Then segment the data into trials and plot a comparison of the
        raw input and transformed input.

        .. code-block:: python

            subject = 'beignet'
            id = 5974
            date = datetime.date(2022, 7, 1)
            exp_data, exp_metadata = aopy.data.load_preproc_exp_data(data_dir, subject, id, date)

            original = exp_data['task']['manual_input']
            input = aopy.postproc.bmi3d.covert_input_to_world_coords(original, exp_metadata['rotation'], exp_metadata['offset'])
            
            go_cue = 32
            trial_end = 239
            print(exp_data['bmi3d_events']['code'], exp_data['bmi3d_events']['time'])
            segments, times = aopy.preproc.get_trial_segments(exp_data['bmi3d_events']['code'], exp_data['bmi3d_events']['time'], 
                                                    [go_cue], [trial_end])
            segments_original = aopy.preproc.get_data_segments(original, times, 1)
            segments_input = aopy.preproc.get_data_segments(input, times, 1)

            plt.figure()
            plt.subplot(2,2,1)
            aopy.visualization.plot_trajectories(segments_original)
            plt.title('Raw input')
            plt.subplot(2,2,2)
            aopy.visualization.plot_trajectories(segments_input, bounds=[-10,10,-10,10])
            plt.title('Transformed')
            plt.subplot(2,2,3, projection='3d')
            aopy.visualization.plot_trajectories(segments_original)
            plt.subplot(2,2,4, projection='3d')
            aopy.visualization.plot_trajectories(segments_input, bounds=[-10,10,-10,10,-10,10])

        .. image:: _images/test_get_bmi3d_mc_input.png
    '''
    from built_in_tasks.manualcontrolmultitasks import rotations
    bmi3d_space_input = np.dot((manual_input + offset), scale * rotations[rotation][:3,:3])

    return bmi3d_space_input[:,[0,2,1]] # return (right-handed) world coordinates

def get_world_to_screen_mapping(exp_rotation='none', x_rot=0, y_rot=0, z_rot=0, exp_scale=1):
    '''
    Returns the mapping $M$ that transforms centered user input from world to screen coordinates.
    World coordinates (x: right/left, y: up/down, z: forward/backward) and screen coordinates
    (x: right/left, y: up/down, z: into/out of the screen) differ only in that the screen may be
    placed arbitrarily in the world. 
    The mapping $M$ is related to the `exp_rotation` mapping $M_{q}$ used by bmi3d, but with 
    axes swapped through multiplication with $T_{q\rightarrow w} = T_{w\rightarrow q}$,
    the transformation that converts bmi3d coordinates to world coordinates
    (it happens to be its own inverse). The full mapping $M$ returned by this function is:
    $M = T_{w\rightarrow q} M_q T_{q\rightarrow w}$

    Args:
        exp_rotation (str, optional): desired experimental rotation from exp_metadata['rotation']. Default 'none'.
        x_rot (float, optional): rotation about x-axis in degrees from exp_metadata['x_perturbation_rotation']. Default 0.
        y_rot (float, optional): rotation about y-axis in degrees from exp_metadata['pertubation_rotation']. Default 0.
        z_rot (float, optional): rotation about z-axis in degrees from exp_metadata['z_perturbation_rotation']. Default 0.
        exp_scale (float, optional): gain scaling factor of the mapping from exp_metadata['exp_scale']. Default 1.

    Returns:
        (3, 3): mapping from centered world coordinates to screen coordinates    

    Examples:
        To reproduce how the mapping was used online, multiply position data in world coordinates ($r_w$)
        by the mapping $M$ to get coordinates in screen space ($r_s$; x is up on the screen, y is right 
        on the screen, z is into the screen). 

        $$
        r_{s} = r_{w} M
        $$

        .. code-block:: python

        .. image:: _images/bmi3d_mc_mapping.png

        Alternatively, to transform screen space coordinates to world coordinates, multiply by the 
        inverse mapping $M^{-1}$:

        $$
        r_{w} = r_{s} M^{-1}
        $$

        This can be useful for example to find the plane on which the user data should have been moving if the
        mapping was correctly learned.

    '''
    from built_in_tasks.manualcontrolmultitasks import exp_rotations
    perturbation_rotation = R.from_euler('xyz', [x_rot, y_rot, z_rot], degrees=True).as_matrix()
    bmi3d_mapping = exp_scale * np.dot(exp_rotations[exp_rotation][:3,:3], perturbation_rotation)
    
    return bmi3d_mapping[[0, 2, 1], :][:, [0, 2, 1]] # return mapping in right-handed coordinates

def get_incremental_world_to_screen_mappings(start, stop, step, bmi3d_axis='y', exp_rotation='none', exp_scale=1):
    '''
    Get the mappings from *centered* user input (in world coordinates) to screen space coordinates
    for an incremental rotation experiment. 

    Args:
        start (float): starting angle in degrees
        stop (float): ending angle in degrees
        step (float): step size in degrees
        bmi3d_axis (str, optional): axis about which to rotate the hand. Default 'y'.
        exp_rotation (str, optional): desired experimental rotation from exp_metadata['rotation']. Default 'none'.
        exp_scale (float, optional): gain scaling factor of the mapping from exp_metadata['exp_scale']. Default 1.

    Returns:
        list: list of mapping matrices from centered world coordinates to screen coordinates
    '''
    from built_in_tasks.manualcontrolmultitasks import exp_rotations
    mappings = []
    for perturbation in np.arange(start, stop+step, step):
        perturbation_rotation = R.from_euler(bmi3d_axis, perturbation, degrees=True).as_matrix()
        mapping = exp_scale * np.dot(exp_rotations[exp_rotation][:3,:3], perturbation_rotation)
        mappings.append(mapping[[0, 2, 1], :][:, [0, 2, 1]]) # return mappings in right-handed coordinates

    return mappings