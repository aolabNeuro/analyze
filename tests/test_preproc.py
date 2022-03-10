from aopy.preproc import *
from aopy.preproc.bmi3d import *
from aopy.data import *
import numpy as np
import unittest

test_dir = os.path.dirname(__file__)
data_dir = os.path.join(test_dir, 'data')
write_dir = os.path.join(test_dir, 'tmp')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

class DigitalCalcTests(unittest.TestCase):

    def test_closest_value(self):
        test_sequence = np.arange(100)
        test_radius = 8

        # test value within range (rounding down)
        test_timestamp = 16.5
        test_expected_value = 16
        value, idx = get_closest_value(test_timestamp, test_sequence, test_radius)
        self.assertEqual(test_expected_value, value)

        # test value within range (rounding up)
        test_timestamp = 16.51
        test_expected_value = 17
        value, idx = get_closest_value(test_timestamp, test_sequence, test_radius)
        self.assertEqual(test_expected_value, value)

        #test value out of range
        test_timestamp = 130
        test_expected_value = None
        value, idx = get_closest_value(test_timestamp, test_sequence, test_radius)
        self.assertEqual(value, test_expected_value)

        #test value out of range
        test_timestamp = -20
        test_expected_value = None
        value, idx = get_closest_value(test_timestamp, test_sequence, test_radius)
        self.assertEqual(value, test_expected_value)

    def test_find_measured_event_times(self):
        approx_times = [0.1, 0.5, 1.0, 55.8]
        measured_times = [0.1, 0.75, 1.2, 1.8, 20]
        radius = 0.25
        
        parsed_times = find_measured_event_times(approx_times, measured_times, radius)

        expected_array = np.array([0.1, 0.75, 1.2, np.nan])
        np.testing.assert_allclose(parsed_times, expected_array)

        # Check idx
        approx_times = [0.1, 0.5, 1.0, 1.2]
        measured_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.3]
        radius = 0.1
        parsed_times, parsed_idx = find_measured_event_times(approx_times, measured_times, radius, return_idx=True)

        expected_array = np.array([0.1, 0.5, 1.0, np.nan])
        expected_idx = np.array([0, 4, 7, np.nan])
        np.testing.assert_allclose(parsed_times, expected_array)
        np.testing.assert_allclose(parsed_idx, expected_idx)

    def test_get_measured_clock_timestamps(self):
        latency_estimate = 0.1
        search_radius = 1
        estimated_timestamps = np.array([0.5, 2., 3.8, 5.0])
        measured_timestamps = np.array([0.64, 2.1, 3.8, 4.9])
        uncorrected = get_measured_clock_timestamps(estimated_timestamps, measured_timestamps, latency_estimate, search_radius)
        self.assertCountEqual(measured_timestamps, uncorrected)

        search_radius = 0.005
        estimated_timestamps = np.arange(10000)/100
        measured_timestamps = estimated_timestamps.copy()*1.00001 + latency_estimate
        measured_timestamps = np.delete(measured_timestamps, [500])
        uncorrected = get_measured_clock_timestamps(estimated_timestamps, measured_timestamps, latency_estimate, search_radius)
        self.assertEqual(len(uncorrected), len(estimated_timestamps))
        self.assertEqual(np.count_nonzero(np.isnan(uncorrected)), 1)
        self.assertTrue(np.isnan(uncorrected[500]))

    def test_fill_missing_timestamps(self):
        uncorrected_timestamps = [0.01, 0.08, np.nan, np.nan, 0.25, np.nan, 0.38]
        expected = [0.01, 0.08, 0.25, 0.25, 0.25, 0.38, 0.38]
        filled = fill_missing_timestamps(uncorrected_timestamps)
        self.assertCountEqual(expected, filled)

    def test_interp_timestamps2timeseries(self):
        timestamps = np.array([1,2,3,4])
        timestamp_values = np.array([100,200,100,300])
        expected_timeseries = np.array([100,150,200,150,100,200,300])
        expected_new_samplepts = np.array([1,1.5,2,2.5,3,3.5,4])
        timeseries, new_samplepts = interp_timestamps2timeseries(timestamps, timestamp_values, samplerate=2)
        np.testing.assert_allclose(timeseries, expected_timeseries)
        np.testing.assert_allclose(new_samplepts, expected_new_samplepts)

        # Test if nans are included
        timestamps = np.array([1,2,np.nan,4])
        timestamp_values = np.array([100,np.nan,100,400])
        expected_timeseries = np.array([100,150,200,250,300,350,400])
        expected_new_samplepts = np.array([1,1.5,2,2.5,3,3.5,4])
        timeseries, new_samplepts = interp_timestamps2timeseries(timestamps, timestamp_values, samplerate=2)
        np.testing.assert_allclose(timeseries, expected_timeseries)
        np.testing.assert_allclose(new_samplepts, expected_new_samplepts)

        # Test is sample point array is input
        timestamps = np.array([1,2,3,4])+4
        timestamp_values = np.array([100,200,100,300])
        expected_timeseries = np.array([100,150,200,150,100,200,300])
        input_samplepts = np.array([1,1.5,2,2.5,3,3.5,4])+4
        timeseries, new_samplepts = interp_timestamps2timeseries(timestamps, timestamp_values, sampling_points=input_samplepts)
        np.testing.assert_allclose(timeseries, expected_timeseries)
        np.testing.assert_allclose(new_samplepts, input_samplepts)

        # Test warning messages:
        timestamps = np.array([1,2,3,4])
        timestamp_values = np.array([100,200,100,300])
        out = interp_timestamps2timeseries(timestamps, timestamp_values)
        self.assertEqual(out, None)
        timestamps = np.array([1,2,1,4])
        out = interp_timestamps2timeseries(timestamps, timestamp_values)
        self.assertEqual(out, None)

        # Test extrapolate
        timestamps = np.array([1,2,3,4])
        timestamp_values = np.array([100,200,100,300])
        expected_timeseries = np.array([100,150,200,150,100,200,300, 400, 500])
        input_samplepts = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5])
        timeseries, new_samplepts = interp_timestamps2timeseries(timestamps, timestamp_values, sampling_points=input_samplepts)
        np.testing.assert_allclose(timeseries, expected_timeseries)
        np.testing.assert_allclose(new_samplepts, input_samplepts)

class EventFilterTests(unittest.TestCase):

    def test_trial_align_events(self):
        # test trial_separate
        events = np.array([2, 7, 5, 7, 2, 5, 7, 4, 2, 3, 6, 2, 3, 6, 4, 6, 3, 1, 3, 2, 4, 2,
            6, 4, 5, 5, 0, 3, 2, 4, 2, 4, 2, 5, 3, 2, 4, 0, 5, 2, 2, 7, 4, 6,
            3, 0, 6, 0, 1, 2, 3, 5, 3, 1, 4, 1, 2, 2, 7, 1, 1, 0, 6, 0, 1, 7,
            4, 5, 3, 3, 2, 4, 4, 1, 1, 5, 2, 3, 1, 4, 0, 5, 0, 0, 4, 2, 2, 6,
            3, 4, 0, 0, 1, 6, 5, 2, 1, 0, 7, 2], dtype=np.uint32)
        times = np.arange(0, 10, 0.1)
        times[5] = 0.45 # Check that trial align works with different delta(T)

        expected_aligned_events = np.array([[2, 7],
            [2, 5],
            [2, 3],
            [2, 3],
            [2, 4],
            [2, 6],
            [2, 4],
            [2, 4],
            [2, 5],
            [2, 4],
            [2, 2],
            [2, 7],
            [2, 3],
            [2, 2],
            [2, 7],
            [2, 4],
            [2, 3],
            [2, 2],
            [2, 6],
            [2, 1],
            [2, -1]])
        
        expected_aligned_times = np.array([[0.0, 0.1],
            [0.4, 0.45],
            [0.8, 0.9],
            [1.1, 1.2],
            [1.9, 2. ],
            [2.1, 2.2],
            [2.8, 2.9],
            [3. , 3.1],
            [3.2, 3.3],
            [3.5, 3.6],
            [3.9, 4. ],
            [4. , 4.1],
            [4.9, 5. ],
            [5.6, 5.7],
            [5.7, 5.8],
            [7. , 7.1],
            [7.6, 7.7],
            [8.5, 8.6],
            [8.6, 8.7],
            [9.5, 9.6],
            [9.9, -1.]])
        
        event_to_align = 2
        aligned_events, aligned_times = trial_separate(events, times, event_to_align, n_events=2)
        aligned_events_offset, aligned_times_offset = trial_separate(events[:15], times[:15], event_to_align, n_events=2,nevent_offset=-1)
        aligned_events_offset2, aligned_times_offset2 = trial_separate(events[-15:], times[-15:], event_to_align, n_events=2,nevent_offset=1)
        expected_aligned_events_offset = np.array([[-1, 2],
            [7, 2],
            [4, 2],
            [6, 2]])
        expected_aligned_times_offset = np.array([[-1, 0.0],
            [0.3, 0.4],
            [0.7, 0.8],
            [1, 1.1]])
        expected_aligned_events_offset2 = np.array([[2, 6],
            [6, 3],
            [1, 0],
            [-1, -1]])
        expected_aligned_times_offset2 = np.array([[8.6, 8.7],
            [8.7, 8.8],
            [9.6, 9.7],
            [-1, -1]])



        np.testing.assert_allclose(expected_aligned_events, aligned_events)
        np.testing.assert_allclose(expected_aligned_times, aligned_times)
        np.testing.assert_allclose(expected_aligned_events_offset, aligned_events_offset)
        np.testing.assert_allclose(expected_aligned_times_offset, aligned_times_offset)
        np.testing.assert_allclose(expected_aligned_events_offset2, aligned_events_offset2)
        np.testing.assert_allclose(expected_aligned_times_offset2, aligned_times_offset2)

        expected_aligned_times = np.array([[0. , 0.1],
            [0. , 0.05],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.]])

        trial_aligned_times = trial_align_events(aligned_events, aligned_times, event_to_align)
        np.testing.assert_allclose(expected_aligned_times, trial_aligned_times)

        event_log_events_in_str = [
                    ('wait', 0.),
                    ('target',1.),
                    ('reward',4.),
                    ('wait',5.),
                    ('target',6.),
                    ('reward',10.),
                    ('wait',10.),
                    ('target',11.),
                    ('wait',18.),

        ]

        events, times = zip(*event_log_events_in_str)
        event_to_align = 'wait'

        expected_events = np.array([
            ['wait', 'target', 'reward'],
            ['wait', 'target', 'reward'],
            ['wait', 'target', 'wait'],
            ['wait', '', ''],
        ])

        aligned_events, aligned_times = trial_separate(np.array(events), np.array(times), event_to_align, n_events=3)
        np.testing.assert_array_equal(expected_events, aligned_events)

    def test_trial_align_events_to_list_of_tuples(self):

        NUM_WAIT =  0
        NUM_TARGET = 1
        NUM_REWARD = 2

        event_log_with_events_in_number = [
                        (NUM_WAIT, 4),
                        (NUM_TARGET,4.1),
                        (NUM_REWARD,4.2),
                        (NUM_WAIT,5),
                        (NUM_TARGET,6),
                        (NUM_REWARD,10),
                        (NUM_WAIT,14),
                        (NUM_TARGET,16),

        ]
        event_log = np.array(event_log_with_events_in_number)
        
        #get all the event codes
        events = event_log[:,0]
        #get all the timestamps
        times  = event_log[:,1]
        
        aligned_events, aligned_times = trial_separate(events, times, NUM_WAIT, n_events=2)

        expected_aligned_events = np.array([
                                            [NUM_WAIT, NUM_TARGET],
                                            [NUM_WAIT, NUM_TARGET],
                                            [NUM_WAIT, NUM_TARGET]
        ])

        expected_aligned_times = np.array([
                                [ 4.  , 4.1],
                                    [ 5.,   6. ],
                                    [14. , 16. ]],
        )

        np.testing.assert_allclose(expected_aligned_events, aligned_events)
        np.testing.assert_allclose(expected_aligned_times, aligned_times)

        expected_aligned_times = np.array(
            [[0. , 0.1],
            [0.,  1. ],
            [0. , 2. ]]
        )
        trial_aligned_times = trial_align_events(aligned_events, aligned_times, NUM_WAIT)
        
        np.testing.assert_allclose(expected_aligned_times, trial_aligned_times)
        
    def test_trial_align_data(self):
        
        # Test simple case with 0 time_before
        # Data looks like: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 ...
        # Trigger times:             *                              ...
        # Aligned trials:            5 6 7 8 9 10 11 12 13 14 15    ...
        data = np.arange(100)
        samplerate = 1
        time_before = 0
        time_after = 10
        trigger_times = np.array([5, 55])
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        self.assertEqual(len(trial_aligned), len(trigger_times))
        np.testing.assert_allclose(np.squeeze(trial_aligned[0]), np.arange(5, 15))
        np.testing.assert_allclose(np.squeeze(trial_aligned[1]), np.arange(55, 65))
        
        # Test with nonzero time_before
        # Data looks like: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 ...
        # Trigger times:             *                              ...
        # Aligned trials:        3 4 5 6 7 8 9 10 11 12 13 14 15    ...
        time_before = 2
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        self.assertEqual(len(trial_aligned), len(trigger_times))
        np.testing.assert_allclose(np.squeeze(trial_aligned[0]), np.arange(3, 15))
        np.testing.assert_allclose(np.squeeze(trial_aligned[1]), np.arange(53, 65))
        
        # Test shape is consistent with more dimensions in data
        data = np.ones((100,2))
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        self.assertEqual(trial_aligned.shape, (len(trigger_times), time_before + time_after, 2))
        
        # Test single trial
        data = np.ones((100,1))
        trigger_times = [5]
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        self.assertEqual(trial_aligned.shape, (1, time_before + time_after, 1))
        
        # Test with time_before bleeding into the start of data
        # Data looks like:            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 ...
        # Trigger times:                        *                              ...
        # Aligned trials:   $ $ $ $ $ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15    ...
        #                   where $ is NaN
        data = np.arange(100)
        time_before = 10
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        self.assertTrue(np.count_nonzero(np.isnan(trial_aligned)), 5)
        np.testing.assert_allclose(np.squeeze(trial_aligned)[5:], np.arange(0,15))
        
        # Test if trigger_times is after the length of data
        # Should ignore the trial at 55s
        data = np.arange(50)
        time_before = 0
        trigger_times = [5, 55]
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        np.testing.assert_allclose(np.squeeze(trial_aligned[0]), np.arange(5,15))
        self.assertTrue(np.count_nonzero(np.isnan(trial_aligned[1])), 15)

        # Test other samplerate
        # At 50 Hz, 0.1s should be 5 samples
        nevents = 4
        event_times = 0.2 + np.arange(nevents)
        samplerate = 50
        nch = 2
        data = np.zeros(((1+nevents)*samplerate, nch))
        event_samples = [int(t*samplerate) for t in event_times]
        for ch in range(nch):
            data[event_samples,ch] = ch+1
        time_before = 0.1
        time_after = 0.1
        aligned_data = trial_align_data(data, event_times, time_before, time_after, samplerate)
        for t in aligned_data:
            np.testing.assert_allclose(np.array(
                [[0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [1., 2.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.]]), t)


    def test_trial_align_times(self):
        timestamps = np.array([2, 6, 7, 10, 25, 27])
        trigger_times = np.array([5, 10, 15, 20, 25])
        time_before = 0
        time_after = 9
        trial_aligned, trial_indices = trial_align_times(timestamps, trigger_times, time_before, time_after, subtract=False)
        self.assertEqual(len(trial_aligned), len(trial_indices))
        self.assertEqual(len(trial_aligned), len(trigger_times))
        self.assertTrue(np.allclose(trial_aligned[0], [6, 7, 10]))
        self.assertEqual(len(trial_aligned[2]), 0)
        trial_aligned, trial_indices = trial_align_times(timestamps, trigger_times, time_before, time_after, subtract=True)
        self.assertTrue(np.allclose(trial_aligned[0], np.array([6, 7, 10])-5))


    def test_get_trial_segments(self):
        events = [0, 2, 4, 6, 0, 2, 3, 6]
        times = np.arange(len(events))
        start_evt = 2
        end_evt = [3, 4]
        segments, times = get_trial_segments(events, times, start_evt, end_evt)
        self.assertTrue(np.allclose(segments, [[2, 4], [2, 3]]))
        self.assertTrue(np.allclose(times, [[1, 2], [5, 6]]))

    def test_locate_trials_with_event(self):
        # Test with ints
        aligned_events = np.array([[2, 7],
            [2, 5],
            [2, 3],
            [2, 3],
            [2, 4],
            [2, 6]])
        split_events1, split_events_array1 = locate_trials_with_event(aligned_events, 3, 1)
        expected_split_events1 = np.array([2,3])
        
        np.testing.assert_allclose(expected_split_events1, split_events1[0])
        np.testing.assert_allclose(expected_split_events1, split_events_array1)

        # Test strings without assigned column to look at
        aligned_events_str = np.array([['Go', 'Target 1', 'Target 1'],
            ['Go', 'Target 2', 'Target 2'],
            ['Go', 'Target 4', 'Target 1'],
            ['Go', 'Target 1', 'Target 2'],
            ['Go', 'Target 2', 'Target 1'],
            ['Go', 'Target 3', 'Target 1']])
        split_events_str, split_events_array_str = locate_trials_with_event(aligned_events_str, ['Target 1','Target 2'])
        expected_split_events_str1 = np.array([0,2,3,4,5])
        expected_split_events_str2 = np.array([1,3,4])
        expected_split_events_all = np.array([0,2,3,4,5,1,3,4])
        np.testing.assert_allclose(expected_split_events_str1, split_events_str[0])
        np.testing.assert_allclose(expected_split_events_str2, split_events_str[1])
        np.testing.assert_allclose(expected_split_events_all, split_events_array_str)

        # Test strings with assigned column to look at
        split_events_str, split_events_array_str = locate_trials_with_event(aligned_events_str, ['Target 1','Target 2'],1)
        expected_split_events_str1 = np.array([0,3])
        expected_split_events_str2 = np.array([1,4])
        expected_split_events_all = np.array([0,3,1,4])
        np.testing.assert_allclose(expected_split_events_str1, split_events_str[0])
        np.testing.assert_allclose(expected_split_events_str2, split_events_str[1])
        np.testing.assert_allclose(expected_split_events_all, split_events_array_str)


    def test_get_data_segments(self):
        data = np.arange(100)
        segment_times = np.array([[0, 4], [50, 51]])
        samplerate = 1
        segments = get_data_segments(data, segment_times, samplerate)
        self.assertEqual(len(segments), 2)
        data = np.ones((100,3))
        segments = get_data_segments(data, segment_times, samplerate)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].shape, (4,3))

    def test_get_unique_conditions(self):
        trial_idx = np.array(range(10))
        conditions = np.array(range(10))
        trials = get_unique_conditions(trial_idx, conditions, condition_name='idx')
        self.assertEqual(len(trials), 10)
        self.assertEqual(trials[0]['trial'], 0)
        self.assertEqual(trials[-1]['idx'], [9])
        
        trial_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        conditions = np.array([[1, 1], [np.pi/2, 2], [1, 1], [np.pi/2, 2], [1, 1], [4, 4], [1, 1], [4, 4]])
        trials = get_unique_conditions(trial_idx, conditions, condition_name='foobar')
        self.assertEqual(len(trials), 4)
        self.assertCountEqual(trials['trial'], [0, 0, 1, 1])
        np.testing.assert_array_almost_equal(trials['foobar'], np.array([[1, 1], [np.pi/2, 2], [1, 1], [4, 4]]))

        trial_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        conditions = np.random.uniform(size=(8,))
        trials = get_unique_conditions(trial_idx, conditions, condition_name='rando')
        self.assertEqual(len(trials), 8)
        self.assertCountEqual(trials['trial'], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(np.sum(trials['index']), 28)

    def test_calc_eye_calibration(self):
        cursor_data = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]).T
        eye_data = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [3, 4, 5 ,6 ,7, 8, 9, 10]]).T
        cursor_samplerate = 1
        eye_samplerate = 1
        event_cycles = [2, 3, 4, 5]
        event_times = [2, 3, 4, 5]
        event_codes = [0, 1, 0, 1]
        coeff, corr_coeff = calc_eye_calibration(cursor_data, cursor_samplerate, eye_data, eye_samplerate, 
            event_cycles, event_times, event_codes, align_events=[0], trial_end_events=[1])
        
        expected_coeff = [[1., 1.], [1., -1.]]
        expected_corr_coeff = [1., 1.]

        np.testing.assert_allclose(expected_coeff, coeff)
        np.testing.assert_allclose(expected_corr_coeff, corr_coeff)

class TestPrepareExperiment(unittest.TestCase):

    def test_parse_bmi3d(self):

        # Test empty
        self.assertRaises(Exception, lambda: parse_bmi3d(data_dir, files))

        def check_required_fields(data, metadata):
            self.assertIn('fps', metadata)
            self.assertIn('sync_protocol_version', metadata)
            self.assertIn('source_dir', metadata)
            self.assertIn('source_files', metadata)
            self.assertIn('clock', data)
            self.assertIn('events', data)
            self.assertIn('task', data)

        # Test sync version 0 (and -1)
        files = {}
        files['hdf'] = 'test20210310_08_te1039.hdf'
        data, metadata = parse_bmi3d(data_dir, files)
        check_required_fields(data, metadata)
        self.assertEqual(metadata['sync_protocol_version'], -1)
        self.assertIn('fps', metadata)
        self.assertAlmostEqual(metadata['fps'], 120.)
        self.assertIn('timestamp', data['clock'].dtype.names)
        self.assertIn('timestamp', data['events'].dtype.names)
        n_cycles = data['clock']['time'][-1] + 1
        self.assertEqual(len(data['clock']), n_cycles)

        # Test sync version 1

        # Test sync version 2 
        files = {}
        files['hdf'] = 'beig20210407_01_te1315.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        check_required_fields(data, metadata)
        trials = data['trials']
        self.assertEqual(len(trials), 3)        
        files['ecube'] = '2021-04-07_BMI3D_te1315'
        data, metadata = parse_bmi3d(data_dir, files) # and with ecube data
        check_required_fields(data, metadata)
        self.assertEqual(metadata['sync_protocol_version'], 2)
        self.assertIn('sync_clock', data)
        self.assertIn('measure_clock_offline', data)
        self.assertEqual(len(data['measure_clock_offline']['timestamp']), 1054)
        self.assertEqual(len(data['measure_clock_online']['timestamp']), 1015)
        self.assertTrue(metadata['has_measured_timestamps'])
        self.assertIn('timestamp', data['clock'].dtype.names)
        self.assertIn('timestamp', data['events'].dtype.names)
        n_cycles = data['clock']['time'][-1] + 1
        # self.assertEqual(len(data['clock']), n_cycles)

        # Test sync version 3
        
        # Test sync version 4
        files = {}
        files['hdf'] = 'beig20210614_07_te1825.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        check_required_fields(data, metadata)
        trials = data['trials']
        self.assertEqual(len(trials), 7)        
        files['ecube'] = '2021-06-14_BMI3D_te1825'
        data, metadata = parse_bmi3d(data_dir, files) # and with ecube data
        check_required_fields(data, metadata)
        self.assertEqual(metadata['sync_protocol_version'], 4)
        self.assertIn('sync_clock', data)
        self.assertIn('measure_clock_offline', data)
        self.assertEqual(len(data['measure_clock_offline']['timestamp']), 1758)
        self.assertEqual(len(data['measure_clock_online']), 1682)
        self.assertTrue(metadata['has_measured_timestamps'])
        self.assertIn('timestamp', data['clock'].dtype.names)
        self.assertIn('timestamp', data['events'].dtype.names)
        n_cycles = data['clock']['time'][-1] + 1
        self.assertEqual(len(data['clock']), n_cycles)

        # Test sync version 5

        # Test sync version 6

        # Test sync version 7
        files = {}
        files['hdf'] = 'test20211213_01_te3498.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        check_required_fields(data, metadata)
        files['ecube'] = '2021-12-13_BMI3D_te3498'
        data, metadata = parse_bmi3d(data_dir, files) # and with ecube data
        self.assertEqual(metadata['sync_protocol_version'], 7)
        self.assertEqual(len(data['clock']), 4848)
        # self.assertEqual(len(data['events']), 66) # seems to be 67
        print(metadata['n_missing_markers'])
        self.assertTrue(metadata['has_measured_timestamps'])
        evt = data['events'][27]
        self.assertEqual(evt['event'], b'CURSOR_ENTER_TARGET')
        self.assertEqual(evt['time'], 1742)

        # Run some trial alignment to make sure the number of trials makes sense
        events = data['events']
        start_states = [b'TARGET_ON']
        end_states = [b'TRIAL_END'] 
        trial_states, trial_idx = get_trial_segments(events['event'], events['time'], start_states, end_states)
        self.assertEqual(len(trial_states), 10)

    def test_parse_oculomatic(self):
        files = {}
        files['ecube'] = '2021-09-30_BMI3D_te2952'
        files['hdf'] = 'beig20210930_02_te2952.hdf'
        data, metadata = parse_oculomatic(data_dir, files)

        self.assertIn('data', data)
        self.assertIn('samplerate', metadata)
        self.assertIn('channels', metadata)
        self.assertIn('labels', metadata)

    def test_parse_optitrack(self):
        files = {}
        files['optitrack'] = 'Take 2021-04-06 11_47_54 (1312).csv'
        data, metadata = parse_optitrack(data_dir, files)
        self.assertIn('data', data)
        self.assertIn('samplerate', metadata)
        self.assertEqual(metadata['samplerate'], 240.)

    def test_proc_exp(self):
        result_filename = 'test_proc_exp.hdf'
        files = {}
        files['hdf'] = 'beig20210407_01_te1315.hdf'
        files['ecube'] = '2021-04-07_BMI3D_te1315'
        proc_exp(data_dir, files, write_dir, result_filename, overwrite=True)
        exp_data = load_hdf_group(write_dir, result_filename, 'exp_data')
        exp_metadata = load_hdf_group(write_dir, result_filename, 'exp_metadata')
        self.assertIsNotNone(exp_data)
        self.assertIsNotNone(exp_metadata)

    def test_proc_mocap(self):
        result_filename = 'test_proc_mocap.hdf'
        files = {}
        files['optitrack'] = 'Pretend take (1315).csv'
        files['ecube'] = '2021-04-07_BMI3D_te1315'
        proc_mocap(data_dir, files, write_dir, result_filename, overwrite=True)
        mocap = load_hdf_group(write_dir, result_filename, 'mocap_data')
        mocap_meta = load_hdf_group(write_dir, result_filename, 'mocap_metadata')
        self.assertIsNotNone(mocap)
        self.assertIsNotNone(mocap_meta)

    def test_proc_eyetracking(self):
        result_filename = 'test_proc_eyetracking_short.hdf'
        files = {}
        files['ecube'] = '2021-09-30_BMI3D_te2952'
        files['hdf'] = 'beig20210930_02_te2952.hdf'

        # Should fail because no preprocessed experimental data
        if os.path.exists(os.path.join(write_dir, result_filename)):
            os.remove(os.path.join(write_dir, result_filename))
        self.assertRaises(ValueError, lambda: proc_eyetracking(data_dir, files, write_dir, result_filename))

        proc_exp(data_dir, files, write_dir, result_filename)

        # Should fail because not enough trials in this session
        self.assertRaises(ValueError, lambda: proc_eyetracking(data_dir, files, write_dir, result_filename))

        result_filename = 'test_proc_eyetracking.hdf'
        files['ecube'] = '2021-09-29_BMI3D_te2949'
        files['hdf'] = 'beig20210929_02_te2949.hdf'
        if os.path.exists(os.path.join(write_dir, result_filename)):
            os.remove(os.path.join(write_dir, result_filename))
        proc_exp(data_dir, files, write_dir, result_filename)

        # Test that eye calibration is returned, but results are not saved
        eye, meta = proc_eyetracking(data_dir, files, write_dir, result_filename, save_res=False)
        self.assertIsNotNone(eye)
        self.assertIsNotNone(meta)
        self.assertRaises(ValueError, lambda: load_hdf_group(write_dir, result_filename, 'eye_data'))
        self.assertRaises(ValueError, lambda: load_hdf_group(write_dir, result_filename, 'eye_metadata'))

        # Test that eye calibration is saved
        proc_eyetracking(data_dir, files, write_dir, result_filename, save_res=True)
        eye = load_hdf_group(write_dir, result_filename, 'eye_data')
        meta = load_hdf_group(write_dir, result_filename, 'eye_metadata')
        self.assertIsNotNone(eye)
        self.assertIsNotNone(meta)



    def preproc_multiple(self):
        result_filename = 'test_proc_multiple.hdf'
        files = {}
        files['hdf'] = 'beig20210407_01_te1315.hdf'
        files['ecube'] = '2021-04-07_BMI3D_te1315'
        files['optitrack'] = 'Pretend take (1315).csv'
        proc_exp(data_dir, files, write_dir, result_filename, overwrite=True)
        proc_mocap(data_dir, files, write_dir, result_filename, overwrite=True)
        contents = get_hdf_dictionary(write_dir, result_filename)
        self.assertIn('exp_data', contents)
        self.assertIn('mocap_data', contents)


    def test_proc_lfp(self):
        result_filename = 'test_proc_lfp.hdf'
        files = {'ecube': 'fake ecube data'}
        proc_lfp(data_dir, files, write_dir, result_filename, overwrite=True)

        contents = get_hdf_dictionary(write_dir, result_filename)
        self.assertIn('lfp_data', contents)
        self.assertIn('lfp_metadata', contents)

        lfp_data = load_hdf_data(write_dir, result_filename, 'lfp_data')
        lfp_metadata = load_hdf_group(write_dir, result_filename, 'lfp_metadata')

        self.assertEqual(lfp_data.shape, (1000, 8))
        self.assertEqual(lfp_metadata['lfp_samplerate'], 1000)

if __name__ == "__main__":
    unittest.main()
