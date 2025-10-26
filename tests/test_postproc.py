from aopy.postproc import *
import aopy
import numpy as np
import warnings
import unittest
import os
import matplotlib.pyplot as plt
import datetime

test_dir = os.path.dirname(__file__)
data_dir = os.path.join(test_dir, 'data')
write_dir = os.path.join(test_dir, 'tmp')
docs_dir = os.path.join(test_dir, '../docs/source/_images')
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

        # Test if target positions in center out rotate correctly
        unique_targets = np.array([[4.5962, 4.5962],[0., -6.5 ],[-6.5, 0.],[-4.5962, -4.5962],[0., 6.5]])
        new_axis = np.array([6.5,0])

        for target in unique_targets:
            current_axis = target.copy()
            print(target, new_axis, target)
            rotated_target = rotate_spatial_data(target, new_axis, current_axis)
            np.allclose(rotated_target, new_axis, atol=1e-6)

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

    def test_smooth_timeseries_gaus(self):
        # Test basics
        npts = 1001
        data = np.ones(npts)
        samplerate = 1000
        sd = 50
        smoothed_data = smooth_timeseries_gaus(data, sd, samplerate)
        self.assertTrue(len(smoothed_data)==npts)

        smoothed_data = smooth_timeseries_gaus(data, sd, samplerate, conv_mode='valid')
        self.assertTrue(np.sum(np.diff(smoothed_data))==0)

        # Test that kernel is acting gaussian as expected
        data = np.zeros(npts)
        center_pt = 500
        data[center_pt] = 1
        nstd = 3
        output, kernel = smooth_timeseries_gaus(data, sd, samplerate,nstd=nstd, return_kernel=True)
        self.assertEqual(np.round(np.sum(output[int(center_pt-(sd)):int(center_pt+(sd))]), 4), 0.6827) # From 68/95/99.7 rule of standard deviations
        self.assertEqual(np.round(np.sum(output[int(center_pt-(2*sd)):int(center_pt+(2*sd))]), 4), 0.9545) # From 68/95/99.7 rule of standard deviations

        # Test return kenrnel
        self.assertEqual(len(kernel), (2*sd*nstd)+1) # Multiply by 2 to account for SD on both sites of the center, and +1 to account for center point.
                

        # Plot example
        np.random.seed(0)
        npts = 250
        sd = 20
        unsmoothed_data = np.random.rand(npts)
        smoothed_data = smooth_timeseries_gaus(unsmoothed_data, sd, samplerate)
        time_axis = (np.arange(npts)/samplerate) * 1000 # Convert from [s] to [ms]

        fig, ax = plt.subplots(1,1)
        ax.plot(time_axis, unsmoothed_data, label='Unsmoothed')
        ax.plot(time_axis, smoothed_data, label='Smoothed')
        ax.legend()
        aopy.visualization.savefig(docs_dir, 'gaus_smoothing_example.png')

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
        
    def test_get_conditioned_trials_per_target(self):
        target_idx_test = np.array([1,2,3,1,2,3,1,1,3,2,3,1,2,1,3])
        cond_mask_test = np.array([False,False,False,False,True,True,True,True,True,True,True,True,True,True,True])
        min_trial = get_minimum_trials_per_target(target_idx_test, cond_mask_test)
        trial_mask = get_conditioned_trials_per_target(target_idx_test, min_trial, cond_mask_test, replacement=False, seed=None)
        self.assertEqual(min_trial,3)
        self.assertTrue(np.all(target_idx_test[trial_mask] == [1,2,3,1,2,3,1,2,3]))
        self.assertTrue(sum(target_idx_test[trial_mask] == 1) == min_trial)
        self.assertTrue(sum(target_idx_test[trial_mask] == 2) == min_trial)
        self.assertTrue(sum(target_idx_test[trial_mask] == 3) == min_trial)

        cond_mask_test = target_idx_test != 1
        trial_mask = get_conditioned_trials_per_target(target_idx_test, min_trial, cond_mask_test, replacement=False, seed=None)
        self.assertTrue(sum(target_idx_test[trial_mask] == 1) == 0)
        self.assertTrue(sum(target_idx_test[trial_mask] == 2) == min_trial)
        self.assertTrue(sum(target_idx_test[trial_mask] == 3) == min_trial)

class TestEyeFuncs(unittest.TestCase):
    
    def test_get_saccade_target_index(self):
        eye_pos = np.array([[0,0,0,1,1,1,1,5,5],[0,0,0,0,0,0,0,0,0]]).T
        saccades_times = np.array([3,7])
        dur = np.array([1.0,1.0])
        fs = 1.
        targ_pos = np.array([[0,0],[1,0]])
        radius = 0.5
        times = np.array([2,6])
        event = np.array([20,30])
        
        onset_pos, offset_pos = get_saccade_pos(eye_pos, saccades_times, dur, fs)
        onset_targ,offset_targ = get_saccade_target_index(onset_pos, offset_pos, targ_pos, radius)
        onset_event, offset_event = get_saccade_event(saccades_times, dur, times, event)
        self.assertTrue(np.all(onset_pos.shape == (2,2)))
        self.assertTrue(np.all(onset_targ==[1,-1]))
        self.assertTrue(np.all(onset_event==[20,30]))
        self.assertTrue(np.all(offset_targ==[1,-1]))
        self.assertTrue(np.all(offset_event==[20,30]))

    def test_get_relevant_saccade_idx(self):
        onset_target = np.array([0,2,0,0])
        offset_target = np.array([0,1,1,1])
        saccade_distance = np.array([1.0,2.0,2.0,3.0])
        target_idx = 1
        relevant_saccade_idx = get_relevant_saccade_idx(onset_target, offset_target, saccade_distance, target_idx)
        self.assertTrue(relevant_saccade_idx == 3)        

class TestMappingFuncs(unittest.TestCase):

    def test_convert_raw_to_world_coords(self):
        # Test with fabricated data
        coords = np.array([[0,0,0],[0,1,1],[0,2,2],[0,3,3],[0,4,4]])
        offset = np.array([2,2,2])
        original = coords - offset
        rotation = 'yzx'
        input = bmi3d.convert_raw_to_world_coords(original, rotation, offset)

        expected = np.array([[0,0,0],[1,1,0],[2,2,0],[3,3,0],[4,4,0]])
        np.testing.assert_allclose(input, expected)

        # Test on some real optitrack data
        subject = 'beignet'
        id = 5974
        date = datetime.date(2022, 7, 1)
        exp_data, exp_metadata = aopy.data.load_preproc_exp_data(data_dir, subject, id, date)

        self.assertEqual(exp_metadata['rotation'], 'yzx') # optitrack
        np.testing.assert_allclose(exp_metadata['offset'], [0,-70,-36]) # rig 1 right arm offset

        original = exp_data['task']['manual_input']
        input = bmi3d.convert_raw_to_world_coords(original, exp_metadata['rotation'], exp_metadata['offset'])
        
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
        plt.title('Original')
        plt.subplot(2,2,2)
        aopy.visualization.plot_trajectories(segments_input, bounds=[-10,10,-10,10])
        plt.title('Transformed')
        plt.subplot(2,2,3, projection='3d')
        aopy.visualization.plot_trajectories(segments_original)
        plt.subplot(2,2,4, projection='3d')
        aopy.visualization.plot_trajectories(segments_input, bounds=[-10,10,-10,10,-10,10])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = 'test_get_bmi3d_mc_input.png'
        aopy.visualization.savefig(docs_dir, filename, transparent=False)

        self.assertTrue(np.all(input[:,[2,1,0]] == exp_data['clean_hand_position'] + exp_metadata['offset']))           

    def test_get_mapping(self):

        # Test with fabricated data
        coords = np.array([[0,0,0],[1,1,0],[2,1,0],[3,4,0],[5,4,0]])
        M = bmi3d.get_world_to_screen_mapping('about_x_90')
        mapped = np.dot(coords, M)
        expected = np.array([[0,0,0],[1,0,1],[2,0,1],[3,0,4],[5,0,4]])
        np.testing.assert_allclose(mapped, expected)

    def test_get_incremental_mappings(self):

        # Test with fabricated data
        coords = np.array([[0,0,0],[1,1,0],[2,1,0],[3,4,0],[5,4,0]])
        mappings = bmi3d.get_incremental_world_to_screen_mappings(0, 90, 90, 'x')
        self.assertEqual(len(mappings), 2)
        mapped_0 = np.dot(coords, mappings[0])
        np.testing.assert_allclose(mapped_0, coords) # should be unchanged
        mapped_1 = np.dot(coords, mappings[1])
        expected = np.array([[0,0,0],[1,0,1],[2,0,1],[3,0,4],[5,0,4]])
        np.testing.assert_allclose(np.round(mapped_1, 5), expected)
        
class NeuropixelFuncs(unittest.TestCase):

    def test_calc_presence_ratio(self):
        # Test no trials
        data = np.zeros((10, 5, 0))  # 10 time points, 0 trials, 5 units
        presence_ratio, present_units = aopy.postproc.neuropixel.calc_presence_ratio(data)
        self.assertEqual(presence_ratio.shape[0], 5)
        self.assertTrue(np.all(present_units == False))

        # Test all units active
        data = np.random.randint(1, 10, (10, 3, 5))  # 10 time points, 5 trials, 3 units
        presence_ratio, present_units = neuropixel.calc_presence_ratio(data)
        self.assertTrue(np.all(present_units))  # All units should be present
        self.assertTrue(np.all(presence_ratio == 1.0))  # Presence ratio should be 1 for all units

        # Test return_details argument
        data = np.zeros((10, 5, 2))  # 10 time points, 5 trials, 2 units
        data[1:5, [0, 1, 0, 1, 1], 0] = 1  # Unit 0 is active in trials 1, 3, and 4
        data[1:3, [0, 1, 1, 0, 0], 1] = 1  # Unit 1 is active in trials 1 and 2
        _, _, presence_details = neuropixel.calc_presence_ratio(data, return_details=True)
        self.assertTrue(np.array_equal(presence_details, np.sum(data, axis=0) > 0))  # Should match trials and units shape

        # Test all units active
        data = np.random.randint(1, 10, (10, 3, 5))  # 10 time points, 5 trials, 3 units
        presence_ratio, present_units = neuropixel.calc_presence_ratio(data)
        self.assertTrue(np.all(present_units))  # All units should be present
        self.assertTrue(np.all(presence_ratio == 1.0))

        # Test some units inactive
        data = np.zeros((10, 3, 5))  # 10 time points, 5 trials, 3 units
        data[1:3,0, :3] = 1  # Only unit 0 is active in trials 1, 2, and 3
        data[0:5,1, :3] = 1  # Only unit 1 is active in trials 1, 2, and 3
                             # Unit 2 is not active on any trials.
        presence_ratio, present_units = neuropixel.calc_presence_ratio(data, min_trial_prop=0.5)
        self.assertEqual(presence_ratio.shape[0], 3)
        self.assertTrue(np.all(present_units[[0, 1]]))  # Units 0 and 1 should be present
        self.assertFalse(present_units[2])  # Unit 2 should not be present

        # Test different min_trial_prop
        data = np.zeros((10, 3, 5))  # 10 time points, 5 trials, 3 units
        data[1:5,0, :4] = 1  # Unit 0 is active in trials 0,1, 2,3,
        data[1:3,1, :3] = 1  # Unit 1 is active in trials 0,1,2
        data[0:2,2,  1] = 1  # Unit 2 is active in trial 1

        # Test with a higher min_trial_prop
        presence_ratio, present_units = aopy.postproc.neuropixel.calc_presence_ratio(data, min_trial_prop=0.6)
        self.assertTrue(present_units[0])  # Unit 0 should be present
        self.assertTrue(present_units[1])  # Unit 1 should be present
        self.assertFalse(present_units[2])
        
    
    def test_get_units_without_refractory_violations(self):
        subject='affi'
        te_id = 18378
        date = datetime.date(2024,9,23)
        port = 1
        ref_period = 1
        ref_perc_thresh = 1
        filename_mc = aopy.data.get_preprocessed_filename(subject,te_id,date,'spike')
        spike_times = aopy.data.load_hdf_group(os.path.join(data_dir,subject), filename_mc, f'drive{port}/spikes')
        good_unit_labels, ref_violations = neuropixel.get_units_without_refractory_violations(spike_times, ref_perc_thresh=ref_perc_thresh, min_ref_period=ref_period, start_time=0, end_time=None)
  
        # Load all unit labels
        all_unit_labels = list(spike_times.keys())

        # Check internal consistency 
        self.assertEqual(len(all_unit_labels), len(ref_violations))
        bad_unit_labels = np.array(all_unit_labels)[ref_violations > ref_perc_thresh]
        for bad_unit_label in bad_unit_labels:
            self.assertFalse(bad_unit_label in good_unit_labels)
    
        # For the bad units double check by hand
        for iunit, unit_label in enumerate(all_unit_labels):
            if unit_label in bad_unit_labels:
                spike_delta = np.diff(spike_times[unit_label])
                bad_spike_perc = np.sum(spike_delta < (ref_period/1000))/len(spike_times[unit_label])
                self.assertAlmostEqual(ref_violations[iunit], bad_spike_perc*100)

        # Check a longer refractory period that will cause many failed units
        ref_period_long = 100
        good_unit_labels_long, ref_violations_long = neuropixel.get_units_without_refractory_violations(spike_times, ref_perc_thresh=ref_perc_thresh, min_ref_period=ref_period_long, start_time=0, end_time=None)
        self.assertFalse(len(good_unit_labels) == len(good_unit_labels_long))

        # Check a higher refractory period percent threshold that will cause all units to be acceptable.
        ref_perc_thresh_thresh = 100
        good_unit_labels_thresh, ref_violations_thresh = neuropixel.get_units_without_refractory_violations(spike_times, ref_perc_thresh=ref_perc_thresh_thresh, min_ref_period=ref_period_long, start_time=0, end_time=None)
        self.assertTrue(len(good_unit_labels_thresh) == len(all_unit_labels))

        # Check that start and end time arguments work
        start_time = 1
        end_time = 2
        good_unit_labels_short, ref_violations_short = neuropixel.get_units_without_refractory_violations(spike_times, ref_perc_thresh=ref_perc_thresh, min_ref_period=ref_period, start_time=0, end_time=None)
        self.assertTrue(len(good_unit_labels_short)==len(good_unit_labels))

    def test_get_high_amplitude_units(self):
        subject='affi'
        te_id = 'wftest'
        date = datetime.date(2024,9,23)
        port = 1
        good_unit_labels, amps, mean_wfs = neuropixel.get_high_amplitude_units(data_dir, subject, te_id, date, port, amp_thresh=50)
        self.assertTrue(len(good_unit_labels)==2) # two units should pass
        self.assertTrue(len(amps)==2)
        self.assertTrue((amps>50).all())
        self.assertTrue(mean_wfs.shape[0]==60)
        self.assertTrue(mean_wfs.shape[1]==2)

        # Test a different amplitude threshold - only one unit should pass
        good_unit_labels, amps, mean_wfs = neuropixel.get_high_amplitude_units(data_dir, subject, te_id, date, port, amp_thresh=100)
        self.assertTrue(len(good_unit_labels)==1)
        self.assertTrue(len(amps)==1)
        self.assertTrue((amps>100).all())
        self.assertTrue(mean_wfs.shape[0]==60)
        self.assertTrue(mean_wfs.shape[1]==1)

        # Check start and end time
        good_unit_labels, amps, mean_wfs = neuropixel.get_high_amplitude_units(data_dir, subject, te_id, date, port, amp_thresh=20, start_time=0, end_time=3.5)
        self.assertTrue('0' in good_unit_labels)
        self.assertTrue('1' not in good_unit_labels)
        self.assertTrue('2' not in good_unit_labels)

        good_unit_labels, amps, mean_wfs = neuropixel.get_high_amplitude_units(data_dir, subject, te_id, date, port, amp_thresh=20, start_time=7, end_time=10)
        self.assertTrue('0' not in good_unit_labels)
        self.assertTrue('1' in good_unit_labels)
        self.assertTrue('2' in good_unit_labels)

    def test_extract_ks_template_amplitudes(self):
        # Test basic functionality with known data
        template_amps = neuropixel.extract_ks_template_amplitudes(data_dir, 'affi', 'tftest', datetime.date(2024,9,23), 1, data_source='Neuropixel', start_time=0, end_time=None)
        for unit_lbl in list(template_amps.keys()):
            self.assertTrue(len(template_amps[unit_lbl])==2)
            self.assertTrue((template_amps[unit_lbl]==1).all())

        # Test new start time and end time
        template_amps = neuropixel.extract_ks_template_amplitudes(data_dir, 'affi', 'tftest', datetime.date(2024,9,23), 1, data_source='Neuropixel', start_time=3, end_time=12)
        for unit_lbl in list(template_amps.keys()):
            if unit_lbl == '0' or unit_lbl == '3':
                self.assertTrue(len(template_amps[unit_lbl])==0)
            else:
                self.assertTrue(len(template_amps[unit_lbl])==2)
                self.assertTrue((template_amps[unit_lbl]==1).all())

    def test_apply_noise_cutoff_thresh(self):
        np.random.seed(0)
        good_spike_amplitudes = np.random.normal(10, 2, size=(1000))
        template_amps = {}
        template_amps['0'] = good_spike_amplitudes
        template_amps['1'] = np.sort(good_spike_amplitudes)[600:]
        template_amps['2'] = np.sort(good_spike_amplitudes)[200:]

        low_bin_thresh = 0.1
        uhq_std_thresh = 5
        good_unit_labels, low_bin_perc, cutoff_metric = neuropixel.apply_noise_cutoff_thresh(template_amps, bin_width=0.2, low_bin_thresh=low_bin_thresh, uhq_std_thresh=uhq_std_thresh, min_spikes=10)
        self.assertTrue(len(good_unit_labels)==1)
        self.assertTrue(low_bin_perc[1] > low_bin_thresh or cutoff_metric[1] > uhq_std_thresh)
        self.assertTrue(low_bin_perc[2] > low_bin_thresh or cutoff_metric[2] > uhq_std_thresh)

        # Test by changing parameters so that every unit passes
        low_bin_thresh = 0.9
        uhq_std_thresh = 20
        good_unit_labels, low_bin_perc, cutoff_metric = neuropixel.apply_noise_cutoff_thresh(template_amps, bin_width=0.2, low_bin_thresh=low_bin_thresh, uhq_std_thresh=uhq_std_thresh, min_spikes=10)
        
        self.assertTrue(len(good_unit_labels)==3)
        self.assertTrue(low_bin_perc[1] < low_bin_thresh or cutoff_metric[1] < uhq_std_thresh)
        self.assertTrue(low_bin_perc[2] < low_bin_thresh or cutoff_metric[2] < uhq_std_thresh)

if __name__ == "__main__":
    unittest.main()
