from aopy.postproc import *
import aopy
import numpy as np
import warnings
import unittest
import os
import matplotlib.pyplot as plt

test_dir = os.path.dirname(__file__)
data_dir = os.path.join(test_dir, 'data')
write_dir = os.path.join(test_dir, 'tmp')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

class TestTrajectoryFuncs(unittest.TestCase):

    def test_translate_spatial_data(self):
        # Test 2D input array and single point
        spatial_data2d = np.array([[0,1], [0,2]])
        spatial_point2d = np.array([0,1])
        new_origin2d = np.array([0,1])
        goal_shifted_data2d = np.array([[0,0], [0,1]])
        goal_shifted_point2d = np.array([0,0])
        shited_data2d = translate_spatial_data(spatial_data2d, new_origin2d)
        shited_point2d = translate_spatial_data(spatial_point2d, new_origin2d)

        np.testing.assert_almost_equal(shited_data2d, goal_shifted_data2d)
        np.testing.assert_almost_equal(shited_point2d, goal_shifted_point2d)

        # Test 3D input array and single point      
        spatial_data3d = np.array([[1,1, 1], [2,2,2]])
        spatial_point3d = np.array([[1,1, 1]])
        new_origin3d = np.array([1,1,1])
        goal_shifted_data3d = np.array([[0,0,0], [1,1,1]])
        goal_shifted_point3d = np.array([[0,0,0]])
        shited_data3d = translate_spatial_data(spatial_data3d, new_origin3d)
        shited_point3d = translate_spatial_data(spatial_point3d, new_origin3d)

        np.testing.assert_almost_equal(shited_data3d, goal_shifted_data3d)
        np.testing.assert_almost_equal(shited_point3d, goal_shifted_point3d)

    def test_rotate_spatial_data(self):
        # Test 2D input array and single point 
        spatial_data2d = np.array([[0,1], [0,2]])
        spatial_point2d = np.array([0,1])
        current_axis2d = np.array([0,1])
        
        # 90 deg rotation
        new_axis2d = np.array([1,0])
        rotated_data2d = rotate_spatial_data(spatial_data2d, new_axis2d, current_axis2d)
        rotated_point2d = rotate_spatial_data(spatial_point2d, new_axis2d, current_axis2d)

        goal_rotated_data2d = np.array([[1,0], [2,0]])
        goal_rotated_point2d = np.array([[1,0]])

        np.testing.assert_almost_equal(rotated_data2d, goal_rotated_data2d)
        np.testing.assert_almost_equal(rotated_point2d, goal_rotated_point2d)

        # 180 deg rotation
        new_axis2d = np.array([0,-1])
        rotated_data2d = rotate_spatial_data(spatial_data2d, new_axis2d, current_axis2d)
        rotated_point2d = rotate_spatial_data(spatial_point2d, new_axis2d, current_axis2d)

        goal_rotated_data2d = np.array([[0,-1], [0,-2]])
        goal_rotated_point2d = np.array([[0,-1]])

        np.testing.assert_almost_equal(rotated_data2d, goal_rotated_data2d)
        np.testing.assert_almost_equal(rotated_point2d, goal_rotated_point2d)

        # another 180 deg rotation
        current_axis2d = np.array([1,0])
        new_axis2d = np.array([-1,0])
        traj = np.array([
            [-0.5, -1.0, -1.5, -2.0, -2.5],
            [-0.5, -0.8, -1.2, 0.0, 2.5]
        ]).T

        rotated_traj = rotate_spatial_data(traj, new_axis2d, current_axis2d)

        expected_rotated_traj = -traj

        print(traj, np.round(rotated_traj, 5), expected_rotated_traj)
        np.testing.assert_almost_equal(rotated_traj, expected_rotated_traj)

        # -90 deg rotation
        current_axis2d = np.array([0,1])
        new_axis2d = np.array([-1,0])

        rotated_data2d = rotate_spatial_data(spatial_data2d, new_axis2d, current_axis2d)
        rotated_point2d = rotate_spatial_data(spatial_point2d, new_axis2d, current_axis2d)

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
        rotated_data3d = rotate_spatial_data(spatial_data3d, new_axis3d, current_axis3d)
        rotated_point3d = rotate_spatial_data(spatial_point3d, new_axis3d, current_axis3d)

        goal_rotated_data3d = np.array([[1,0, 0], [2,0,0]])
        goal_rotated_point3d = np.array([[1,0,0]])

        np.testing.assert_almost_equal(rotated_data3d, goal_rotated_data3d)
        np.testing.assert_almost_equal(rotated_point3d, goal_rotated_point3d)

        # Rotation 2
        new_axis3d = np.array([-1,0, 0])
        rotated_data3d = rotate_spatial_data(spatial_data3d, new_axis3d, current_axis3d)
        rotated_point3d = rotate_spatial_data(spatial_point3d, new_axis3d, current_axis3d)

        goal_rotated_data3d = np.array([[-1,0,0], [-2,0,0]])
        goal_rotated_point3d = np.array([[-1,0,0]])

        np.testing.assert_almost_equal(rotated_data3d, goal_rotated_data3d)
        np.testing.assert_almost_equal(rotated_point3d, goal_rotated_point3d)

        # Rotation 3
        new_axis3d = np.array([0,1, 0])
        rotated_data3d = rotate_spatial_data(spatial_data3d, new_axis3d, current_axis3d)
        rotated_point3d = rotate_spatial_data(spatial_point3d, new_axis3d, current_axis3d)

        goal_rotated_data3d = np.array([[0,1,0], [0,2,0]])
        goal_rotated_point3d = np.array([[0,1,0]])

        np.testing.assert_almost_equal(rotated_data3d, goal_rotated_data3d)
        np.testing.assert_almost_equal(rotated_point3d, goal_rotated_point3d)

        # Test that points and arrays give the same result
        spatial_data = np.array([[2,1,3], [4,1,4]])
        spatial_point = np.array([4,1,4])
        current_axis = np.array([5,3,1])
        new_axis = np.array([1,0, 0])
        rotated_data = rotate_spatial_data(spatial_data, new_axis, current_axis)
        rotated_point = rotate_spatial_data(spatial_point, new_axis, current_axis)

        np.testing.assert_almost_equal(rotated_data[1,:], rotated_point[0,:])

    def test_get_relative_point_location(self):
        # Test with multiple points
        cursorpos = np.array(((1,1),(1,1),(1,1)))
        targetpos = np.array(((-1,-1),(-1,-1),(-1,-1)))
        relative_target_angle, relative_target_pos = get_relative_point_location(cursorpos, targetpos)
        np.testing.assert_almost_equal(relative_target_angle, np.deg2rad(np.array((225, 225, 225))))
        np.testing.assert_almost_equal(relative_target_pos, np.array(((-2, -2),(-2, -2),(-2, -2))))

        # Test with one point
        cursorpos = np.array((1,1))
        targetpos = np.array((-1,-1))
        relative_target_angle, relative_target_pos = get_relative_point_location(cursorpos, targetpos)
        np.testing.assert_almost_equal(relative_target_angle, np.deg2rad(225))
        np.testing.assert_almost_equal(relative_target_pos, np.array((-2, -2)))

    def test_get_inst_target_dir(self):
        cursorpos = np.zeros((3,3,2))
        cursorpos[:,:,0] =  np.array([[1,0,1], [0,-1,1], [-1,0,-1]])
        cursorpos[:,:,1] =  np.array([[1,0,1], [1,-1,-1], [0,1,0]])
        targetpos = np.array([[1,0], [1,1], [-1,-1]])
        insttargetdir = get_inst_target_dir(cursorpos, targetpos)
        expected_insttargetdir = np.array([[270,45,225], [315,45,180], [0,0,270]])
        np.testing.assert_almost_equal(insttargetdir, np.deg2rad(expected_insttargetdir))

    def test_mean_fr_inst_dir(self):
        ## Test
        cursorpos = np.zeros((20,2,2))
        cursorxpos = np.arange(-1, 1, 0.1) #[ncursor time bin (20pts)]
        cursorpos[:,:,0] = np.tile(cursorxpos.reshape(-1,1), (1,2))
        cursorypos = np.arange(-1, 1, 0.1) #[ncursor time bin (20pts)]
        cursorpos[:,:,1] = np.tile(cursorypos.reshape(-1,1), (1,2))
        data = np.ones((100, 10, 2)) #[ntime, nunit, ntrial]
        targetpos = np.array([[1,1], [-1,-1]]) #[ntrial x 2]
        data_binwidth = 1
        ntarget_directions = 8
        data_samplerate = 10
        cursor_samplerate = 2

        meanfr = mean_fr_inst_dir(data, cursorpos, targetpos, data_binwidth, ntarget_directions, data_samplerate, cursor_samplerate)
        exp_meanfr = np.zeros((10, int(360/45)))*np.nan
        exp_meanfr[:,1] = 10
        exp_meanfr[:,5] = 10
    
        self.assertEqual(meanfr.shape[1], 360/45) # Check correct num of direction bins
        np.testing.assert_almost_equal(meanfr, exp_meanfr)
    

class TestCalcFuncs(unittest.TestCase):

    def test_calc_reward_intervals(self):
        timestamps = np.array([1,2,5,6,9,10])
        values = np.array([1,0,1,0,1,0])
        intervals = calc_reward_intervals(timestamps, values)
        self.assertEqual(len(intervals), len(timestamps)/2)
        self.assertTrue(np.allclose(intervals, [(1,2),(5,6),(9,10)]))
        values = np.array([1,1,1,1,1,1,])
        self.assertRaises(ValueError, lambda: calc_reward_intervals(timestamps, values))

    def test_sample_events(self):
        events = ["reward", "reward", "penalty", "reward"]
        times = [0.3, 0.5, 0.7, 1.0]
        samplerate = 10
        frame_events, event_names = sample_events(events, times, samplerate)
        expected_result = [
            [False, False],
            [False, False],
            [False, False],
            [False, True ],
            [False, False],
            [False, True ],
            [False, False],
            [ True, False],
            [False, False],
            [False, False],
            [False, True ]
        ]
        
        np.testing.assert_array_equal(expected_result, frame_events)
        np.testing.assert_array_equal(["penalty", "reward"], event_names)

class TestGetFuncs(unittest.TestCase):

    def test_get_trial_targets(self):
        trials = [0, 1, 1, 2]
        targets = [[1,2,3],[2,3,4],[2,3,4],[5,6,7]]
        trial_targets = get_trial_targets(trials, targets)
        self.assertEqual(len(trial_targets), 3)
        self.assertEqual(len(trial_targets[1]), 2)

    def test_get_calibrated_eye_data(self):
        eye_data = np.array([[0, 1, 2], [1, 2, 3]]).T
        coefficients = np.array([[1, 1], [0, 0]]).T
        calibrated = get_calibrated_eye_data(eye_data, coefficients)
        np.testing.assert_array_equal(eye_data, calibrated)

if __name__ == "__main__":
    unittest.main()
