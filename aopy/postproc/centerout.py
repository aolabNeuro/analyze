import numpy as np
from scipy.spatial.transform import Rotation as R

rotations = dict(
    yzx = np.array(    # names come from rows (optitrack), but screen coords come from columns:
        [[0, 1, 0, 0], # x goes into second column (y-coordinate, coming out of screen)
        [0, 0, 1, 0],  # y goes into third column (z-coordinate, up)
        [1, 0, 0, 0],  # z goes into first column (x-coordinate, right)
        [0, 0, 0, 1]]
    ),
    zyx = np.array(
        [[0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]]
    ),
    xzy = np.array(
        [[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]]
    ),
    xyz = np.identity(4),
)

exp_rotations = dict(
    none = np.identity(4),
    about_x_90 = np.array(
        [[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]]
    ),
    about_x_minus_90 = np.array(
        [[1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]]
    ),
    oop_xy_45 = np.array(
        [[ 0.707,  0.5  ,  0.5  , 0.],
         [ 0.   ,  0.707, -0.707, 0.],
         [-0.707,  0.5  ,  0.5  , 0.],
         [ 0.,     0.   ,  0.,    1.]]
    ),
    oop_xy_minus_45 = np.array(
        [[ 0.707,  0.5  , -0.5  , 0.],
         [ 0.   ,  0.707,  0.707, 0.],
         [ 0.707, -0.5  ,  0.5  , 0.],
         [ 0.,     0.   ,  0.,    1.]]
    ),
    oop_xy_20 = np.array(
        [[ 0.94 ,  0.117,  0.321, 0.],
         [-0.   ,  0.94 , -0.342, 0.],
         [-0.342,  0.321,  0.883, 0.],
         [ 0.,     0.   ,  0.,    1.]]
    ),
    oop_xy_minus_20 = np.array(
        [[ 0.94 ,  0.117, -0.321, 0.],
         [-0.   ,  0.94 ,  0.342, 0.],
         [ 0.342, -0.321,  0.883, 0.],
         [ 0.,     0.   ,  0.,    1.]]
    ))

def _get_mapping(exp_metadata):
    '''
    Returns transformed coordinates based on rotation, offset, and scale traits
    Args:
        exp_metadata: exp_metadata from load_preproc_exp_data(preproc_dir, subject, te_id, date)

    Returns:
        mapping (3 x 3 numpy array): mapping of hand to cursor data used
    '''
    offset = exp_metadata['offset']
    offset_arr = np.array(
        [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [offset[0], offset[1], offset[2], 1]]
    )
    scale =  exp_metadata['scale']
    scale_arr = np.array(
        [[scale, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0, scale, 0],
        [0, 0, 0, 1]]
    )
    rotation = exp_metadata['rotation'] # optitrack -> screen space
#     print(rotation)
    exp_rotation = exp_metadata['exp_rotation'] # out of plane perturbations applied in screen space - x - right, y - towards monkey, z - up
    perturbation = exp_metadata['pertubation_rotation'] # in plane perturbations applied in screen space about y axis so rotation is inplane
    new_plane = np.linalg.multi_dot((offset_arr, scale_arr, rotations[rotation], exp_rotations[exp_rotation])) # multi dot product of 4 x 4 matrices
    pertubation_rot = R.from_euler('y', perturbation, degrees=True)
    mapping = np.matmul(pertubation_rot.as_matrix(), new_plane[0:3])
    return mapping

def _transform_coords(hand_data, exp_metadata):
  '''
    Transforms hand data into mapping used in center out experiment
  Args:
      hand_traj (2D numpy array): 3D array of hand trajectory data (n_timepoints x 3)
      exp_metadata: exp_metadata from load_preproc_exp_data(preproc_dir, subject, te_id, date)

  Returns:

  '''
    offset = exp_metadata['offset']
    offset_arr = np.array(
        [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [offset[0], offset[1], offset[2], 1]]
    )
    scale =  exp_metadata['scale']
    scale_arr = np.array(
        [[scale, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0, scale, 0],
        [0, 0, 0, 1]]
    )
    rotation = exp_metadata['rotation'] # optitrack -> screen space
#     print(rotation)
    exp_rotation = exp_metadata['exp_rotation'] # out of plane perturbations applied in screen space - x - right, y - towards monkey, z - up
    perturbation = exp_metadata['pertubation_rotation'] # in plane perturbations applied in screen space about y axis so rotation is inplane
    old = np.concatenate((np.reshape(hand_data, -1), [1]))
#     print( coords.shape,  old.shape, offset_arr.shape, scale_arr.shape, rotations[rotation].shape, exp_rotations[exp_rotation].shape )
    new = np.linalg.multi_dot((old, offset_arr, scale_arr, rotations[rotation], exp_rotations[exp_rotation]))
    pertubation_rot = R.from_euler('y', perturbation, degrees=True)
    hand_transformed = np.matmul(pertubation_rot.as_matrix(), new[0:3])
#     print(coords, new_coords)
    return hand_transformed

def get_taskspace_and_nullspace(mapping, task='2DCenterOut'):
    '''
    Decomposes hand movement into components of movement in the plane of the screen and out of the plane of the screen
    Args:
        mapping (3 x 3 numpy array): Mapping matrix used in experiment. See :func:`get_mapping` for more details

    Returns:
        t_a (3 x 3 numpy array): Matrix that projects hand movement into the task potent cursor space
        n_a (3 x 3 numpy array): Matrix that projects hand movement into the null space
    '''
    if task == '2DCenterOut':
        a = mapping[[0,2], :] # rows are cursor and columns are hand
    elif task == '1DTracking':
        a = mapping[2,:]

    a_plus = a.T @ np.linalg.pinv(a @ a.T)
    t_a = a_plus @ a
    n_a = np.eye(3) - t_a

    return t_a, n_a

def decompose_hand_movements(hand_data, mapping):
    '''
    Decomposes hand movement into components of movement in the plane of the screen and out of the plane of the screen
    Args:
        hand_data (2D numpy array): 3D array of hand trajectory data (n_timepoints x 3)
        mapping (3 x 3 numpy array): Mapping matrix used in experiment. See :func:`get_mapping` for more details . Rows must be cursor and columns must be hand.

    Returns:

    '''
    t_a, n_a = get_taskspace_and_nullspace(mapping)
    hand_data = np.array(hand_data)
    hT = np.dot(t_a, hand_data.T)  # 3x2400
    hN = np.dot(n_a, hand_data.T)

    hT_norm = np.mean(np.linalg.norm(hT, axis=0))
    hN_norm = np.mean(np.linalg.norm(hN, axis=0))

    return  hT_norm, hN_norm