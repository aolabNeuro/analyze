import aopy
import numpy as np
import warnings
import unittest

class move_trajectory_tests(unittest.TestCase):

    def test_translate_spatial_data(self):
        # Test 2D input array and single point
        spatial_data2d = np.array([[0,1], [0,2]])
        spatial_point2d = np.array([0,1])
        new_origin2d = np.array([0,1])
        goal_shifted_data2d = np.array([[0,0], [0,1]])
        goal_shifted_point2d = np.array([0,0])
        shited_data2d = aopy.postproc.translate_spatial_data(spatial_data2d, new_origin2d)
        shited_point2d = aopy.postproc.translate_spatial_data(spatial_point2d, new_origin2d)

        np.testing.assert_almost_equal(shited_data2d, goal_shifted_data2d)
        np.testing.assert_almost_equal(shited_point2d, goal_shifted_point2d)

        # Test 3D input array and single point      
        spatial_data3d = np.array([[1,1, 1], [2,2,2]])
        spatial_point3d = np.array([[1,1, 1]])
        new_origin3d = np.array([1,1,1])
        goal_shifted_data3d = np.array([[0,0,0], [1,1,1]])
        goal_shifted_point3d = np.array([[0,0,0]])
        shited_data3d = aopy.postproc.translate_spatial_data(spatial_data3d, new_origin3d)
        shited_point3d = aopy.postproc.translate_spatial_data(spatial_point3d, new_origin3d)

        np.testing.assert_almost_equal(shited_data3d, goal_shifted_data3d)
        np.testing.assert_almost_equal(shited_point3d, goal_shifted_point3d)

    def test_rotate_spatial_data(self):
        # Test 2D input array and single point 
        spatial_data2d = np.array([[0,1], [0,2]])
        spatial_point2d = np.array([0,1])
        current_axis2d = np.array([0,1])
        
        # 90 deg rotation
        new_axis2d = np.array([1,0])
        rotated_data2d = aopy.postproc.rotate_spatial_data(spatial_data2d, new_axis2d, current_axis2d)
        rotated_point2d = aopy.postproc.rotate_spatial_data(spatial_point2d, new_axis2d, current_axis2d)

        goal_rotated_data2d = np.array([[1,0], [2,0]])
        goal_rotated_point2d = np.array([[1,0]])

        np.testing.assert_almost_equal(rotated_data2d, goal_rotated_data2d)
        np.testing.assert_almost_equal(rotated_point2d, goal_rotated_point2d)

        # 180 deg rotation
        new_axis2d = np.array([0,-1])
        rotated_data2d = aopy.postproc.rotate_spatial_data(spatial_data2d, new_axis2d, current_axis2d)
        rotated_point2d = aopy.postproc.rotate_spatial_data(spatial_point2d, new_axis2d, current_axis2d)

        goal_rotated_data2d = np.array([[0,-1], [0,-2]])
        goal_rotated_point2d = np.array([[0,-1]])

        np.testing.assert_almost_equal(rotated_data2d, goal_rotated_data2d)
        np.testing.assert_almost_equal(rotated_point2d, goal_rotated_point2d)

        # -90 deg rotation
        new_axis2d = np.array([-1,0])

        rotated_data2d = aopy.postproc.rotate_spatial_data(spatial_data2d, new_axis2d, current_axis2d)
        rotated_point2d = aopy.postproc.rotate_spatial_data(spatial_point2d, new_axis2d, current_axis2d)

        goal_rotated_data2d = np.array([[-1,0], [-2,0]])
        goal_rotated_point2d = np.array([[-1,0]])

        np.testing.assert_almost_equal(rotated_data2d, goal_rotated_data2d)
        np.testing.assert_almost_equal(rotated_point2d, goal_rotated_point2d)

        # Test 3D input array and single point 
        spatial_data3d = np.array([[0,0,1], [0,0,2]])
        spatial_point3d = np.array([0,0,1])
        current_axis3d = np.array([0,0,1])
        
        # Rotation 1
        new_axis3d = np.array([1,0, 0])
        rotated_data3d = aopy.postproc.rotate_spatial_data(spatial_data3d, new_axis3d, current_axis3d)
        rotated_point3d = aopy.postproc.rotate_spatial_data(spatial_point3d, new_axis3d, current_axis3d)

        goal_rotated_data3d = np.array([[1,0, 0], [2,0,0]])
        goal_rotated_point3d = np.array([[1,0,0]])

        np.testing.assert_almost_equal(rotated_data3d, goal_rotated_data3d)
        np.testing.assert_almost_equal(rotated_point3d, goal_rotated_point3d)

        # Rotation 2
        new_axis3d = np.array([-1,0, 0])
        rotated_data3d = aopy.postproc.rotate_spatial_data(spatial_data3d, new_axis3d, current_axis3d)
        rotated_point3d = aopy.postproc.rotate_spatial_data(spatial_point3d, new_axis3d, current_axis3d)

        goal_rotated_data3d = np.array([[-1,0,0], [-2,0,0]])
        goal_rotated_point3d = np.array([[-1,0,0]])

        np.testing.assert_almost_equal(rotated_data3d, goal_rotated_data3d)
        np.testing.assert_almost_equal(rotated_point3d, goal_rotated_point3d)

        # Rotation 3
        new_axis3d = np.array([0,1, 0])
        rotated_data3d = aopy.postproc.rotate_spatial_data(spatial_data3d, new_axis3d, current_axis3d)
        rotated_point3d = aopy.postproc.rotate_spatial_data(spatial_point3d, new_axis3d, current_axis3d)

        goal_rotated_data3d = np.array([[0,1,0], [0,2,0]])
        goal_rotated_point3d = np.array([[0,1,0]])

        np.testing.assert_almost_equal(rotated_data3d, goal_rotated_data3d)
        np.testing.assert_almost_equal(rotated_point3d, goal_rotated_point3d)



if __name__ == "__main__":
    unittest.main()
