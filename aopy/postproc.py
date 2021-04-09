# postproc.py
# code for post-processing neural data
import numpy as np
import warnings

def translate_spatial_data(spatial_data, new_origin):
    '''
    Shifts 2D or 3D spatial data to a new location.

    Required packages: 
        import numpy as np

    Inputs:
        spatial_data [nt, ndim]: Spatial data in 2D or 3D
        new_origin [ndim]: Location of point that will become the origin in cartesian coordinates

    Output:
        new_spatial_data [nt, ndim]: new reach trajectory translated to the new origin
    '''
    new_spatial_data = np.subtract(spatial_data, new_origin)

    return new_spatial_data


def rotate_spatial_data(spatial_data, new_axis, current_axis):
    '''
    Rotates data about the origin into a new coordinate system based on the relationship between
    'new_axis' and 'current_axis'. If 'input_current_axis3d' and 'new_axis' point in 
    the same direction, the code will return 'spatial_data' with a warning that the vectors point in
    the same direction.
    
    This function was written to rotate spatial data but can be applied to other data of similar form.

    Required packages: 
        import numpy as np
        import warnings

    Inputs:
        spatial_data [nt, ndim]: Array of spatial data in 2D or 3D
        new_axis [ndim]: vector pointing along the desired orientation of the data
        current_axis [ndim]: vector pointing along the current orientation of the dat
    Output:
        output_spatial_data [nt, ndim]: new reach trajectory rotated to the new axis
    '''

    # Check if input data is a single point and enfore that it is a row vector
    if len(spatial_data.shape) == 1:
      spatial_data.shape = (1,len(spatial_data))    

    # Initialize output array
    output_spatial_data = np.empty((spatial_data.shape[0], 3))

    # Check for a 2D or 3D trajectory and convert to 3D points
    if spatial_data.shape[1] == 2:
      spatial_data3d = np.concatenate((spatial_data, np.zeros((spatial_data.shape[0],1))), axis = 1)
      new_axis3d = np.concatenate((new_axis, np.array([0])))
      current_axis3d = np.concatenate((current_axis, np.array([0])))
    elif spatial_data.shape[1] == 3:
      spatial_data3d = spatial_data
      new_axis3d = new_axis
      current_axis3d = current_axis

    # Calcualte angle between 'new_axis3d' and target trajectory via dot product
    angle = np.arccos(np.dot(new_axis3d, current_axis3d)/(np.linalg.norm(new_axis3d)*np.linalg.norm(current_axis3d)))

    # If angle is 0, return the original data and warn
    if np.isclose(angle, 0, atol = 1e-8):
      warnings.warn("Starting and desired vector are the same. No rotation applied")
      output_spatial_data = spatial_data3d
      return output_spatial_data

    # If the angle is exactly 180 degrees, slightly nudge the starting vector by 1e-7
    elif np.isclose(angle, np.pi, atol = 1e-8):
      current_axis3d = current_axis3d.astype('float64')
      current_axis3d[0] += 1e-7

    # Calculate unit vector axis to rotate about via cross product
    rotation_axis = np.cross(current_axis3d, new_axis3d)
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
    for point_idx in range(spatial_data3d.shape[0]):
      output_spatial_data[point_idx,:] = rotation_matrix @ spatial_data3d[point_idx,:]

    # Return trajectories the the same dimensions as the input
    if spatial_data.shape[1] == 2:
      return output_spatial_data[:,:2]
    elif spatial_data.shape[1] == 3:
      return output_spatial_data