# postproc.py
# code for post-processing neural data
import numpy as np
import warnings

def translate_spatial_data(input_spatial_data, new_origin):
    '''
    Shifts 2D or 3D spatial data to a new location.

    Required packages: 
        import numpy as np

    Inputs:
        input_spatial_data [nt, ndim]: Spatial data in 2D or 3D
        new_origin [ndim]: Location of point that will become the origin

    Output:
        new_spatial_data [nt, ndim]: new reach trajectory translated to the new origin

    Note:
      - should deal with 2D and 3D data / origin
    '''
    new_spatial_data = np.subtract(input_spatial_data, new_origin)

    return new_spatial_data


def rotate_spatial_data(input_spatial_data, input_desired_vector, input_starting_vector):
    '''
    Rotates data about the origin into a new coordinate system based on the relationship between
    'desired_vector' and 'starting_vector'. If 'input_starting_vector' and 'input_desired_vector' point in 
    the same direction, the code will return 'input_spatial_data' with a warning that the vectors point in
    the same direction.
    
    If the starting and desired vector are exactly 180deg apart. The first component of the starting vector
    gets nudged by 1e-7.

    This function was written to rotate spatial data but can be applied to other data of similar form.

    Required packages: 
        import numpy as np
        import warnings

    Inputs:
        input_spatial_data [nt, ndim]: Spatial data in 2D or 3D
        input_desired_vector [ndim]: vector pointing along the desired orientation of the data
        input_starting_vector [ndim]: vector pointing along the current orientation of the dat
    Output:
        output_spatial_data [nt, ndim]: new reach trajectory rotated to the new axis
    '''

    # Check if input data is a single point and enfore that it is a row vector
    if len(input_spatial_data.shape) == 1:
      input_spatial_data.shape = (1,len(input_spatial_data))    

    # Initialize output array
    output_spatial_data = np.empty((input_spatial_data.shape[0], 3))

    # Check for a 2D or 3D trajectory
    if input_spatial_data.shape[1] == 2:
      spatial_data = np.concatenate((input_spatial_data, np.zeros((input_spatial_data.shape[0],1))), axis = 1)
      desired_vector = np.concatenate((input_desired_vector, np.array([0])))
      starting_vector = np.concatenate((input_starting_vector, np.array([0])))
    elif input_spatial_data.shape[1] == 3:
      spatial_data = input_spatial_data
      desired_vector = input_desired_vector
      starting_vector = input_starting_vector

    # Calcualte angle between 'desired_vector' and target trajectory via dot product
    angle = np.arccos(np.dot(desired_vector, starting_vector)/(np.linalg.norm(desired_vector)*np.linalg.norm(starting_vector)))

    # If angle is 0, return the original data and warn
    if np.isclose(angle, 0, atol = 1e-8):
      warnings.warn("Starting and desired vector are the same. No rotation applied")
      output_spatial_data = input_spatial_data
      return output_spatial_data

    # If the angle is exactly 180 degrees, slightly nudge the starting vector
    elif np.isclose(angle, np.pi, atol = 1e-8):
      starting_vector = starting_vector.astype('float64')
      starting_vector[0] += 1e-7

    # Calculate unit vector axis to rotate about via cross product
    rotation_axis = np.cross(starting_vector, desired_vector)
    unit_rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)

    # Calculate quaternion
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternions_as_rotations
    qr = np.cos(angle/2)
    qi = np.sin(angle/2)*unit_rotation_axis[0]
    qj = np.sin(angle/2)*unit_rotation_axis[1]
    qk = np.sin(angle/2)*unit_rotation_axis[2]


    # Convert quaternion to rotation matrix
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    rotation_matrix = np.array([[1-2*(qj**2 + qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr)],
                                [2*(qi*qj + qk*qr), 1-2*(qi**2 + qk**2), 2*(qj*qk - qi*qr)],
                                [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1-2*(qi**2 + qj**2)]])

    # Apply rotation matrix to each point in the trajectory
    for point_idx in range(spatial_data.shape[0]):
      output_spatial_data[point_idx,:] = rotation_matrix @ spatial_data[point_idx,:]

    # Return trajectories the the same dimensions as the input
    if input_spatial_data.shape[1] == 2:
      return output_spatial_data[:,:2]
    elif input_spatial_data.shape[1] == 3:
      return output_spatial_data