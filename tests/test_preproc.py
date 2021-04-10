from aopy.preproc import *
import numpy as np
import unittest

data_dir = 'tests/data/'
write_dir = 'tests/tmp'
if not os.path.exists(write_dir):
    os.mkdir(write_dir)
    
class DigitalCalcTests(unittest.TestCase):

    def test_convert_analog_to_digital(self):

        #data_length = 3
        #nchan = 2
        analog_data = [[2.9, 0.23], 
                        [1.9, 2.9], 
                        [1.74, 4.76]]
        thresh = 0.5
        expected_digital_array = [[1.0, 0.0], 
                                    [0.0, 1.0], 
                                    [0.0, 1.0]]
        digital_data = convert_analog_to_digital(analog_data, thresh)
        np.testing.assert_almost_equal(expected_digital_array, digital_data)

    def test_detect_edges(self):
        test_bool = [True, False, False, True, True, False, True, True, False, False, False, True]
        test_01 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        test_02 = [0b11, 0, 0, 0, 0, 0, 0, 0b01, 0b10, 0b01, 0b11, 0b01, 0b00, 0b10, 0b00, 0b01, 0, 0b01, 0, 0b01, 0, 0b01, 0]

        ts, values = detect_edges(test_bool, 1)
        assert len(ts) == 6
        assert np.array_equal(ts, [1, 3, 5, 6, 8, 11])
        assert np.array_equal(values, [0, 1, 0, 1, 0, 1])
        ts, values = detect_edges(test_bool, 1, hilow=False)
        assert len(ts) == 3
        assert np.array_equal(ts, [3, 6, 11])
        assert np.array_equal(values, [1, 1, 1])
        ts, values = detect_edges(test_bool, 1, lowhi=False)
        assert len(ts) == 3
        assert np.array_equal(ts, [1, 5, 8])
        assert np.array_equal(values, [0, 0, 0])
        ts, values = detect_edges(test_01, 1, hilow=False)
        assert len(ts) == 6
        assert np.array_equal(ts, [7, 13, 15, 17, 19, 21])
        assert np.array_equal(values, [1, 1, 1, 1, 1, 1])
        ts, values = detect_edges(test_02, 1, hilow=False)
        assert len(ts) == 9
        assert np.array_equal(ts, [7, 8, 9, 10, 13, 15, 17, 19, 21])
        assert np.array_equal(values, [1, 2, 1, 3, 2, 1, 1, 1, 1])

    def test_find_first_significant_bit(self):
        data = 0b0100
        ffs = find_first_significant_bit(data)
        self.assertEqual(ffs, 2)
        data = 0
        ffs = find_first_significant_bit(data)
        self.assertEqual(ffs, -1)

    def test_mask_and_shift(self):
        mask = 0x0000000000ff0000 # bits 17-24
        digital_data = [2, 4, 12*0x10000, 140*0x10000, 0xff0000, 0xff0000]
        masked = mask_and_shift(digital_data, mask)
        assert np.array_equal(masked, [0, 0, 12, 140, 255, 255])

    def test_digital_to_channels(self):
        testdata = [0b0110100001011100101100000100011001010001101110101001000100111000, 0xff0ff00fffffff0f]
        unpacked = convert_digital_to_channels(testdata)
        packed = np.packbits(unpacked, bitorder='little').view(np.uint64)
        assert packed[0] == testdata[0]
        assert packed[1] == testdata[1]
        assert unpacked[0,0] == 0
        assert unpacked[1,0] == 1
        assert unpacked[0,1] == 0

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

    def test_get_measured_frame_timestamps(self):
        latency_estimate = 0.1
        search_radius = 0.001
        estimated_timestamps = np.arange(10000)/100
        measured_timestamps = estimated_timestamps.copy()*1.00001 + latency_estimate
        measured_timestamps = np.delete(measured_timestamps, [500])
        corrected, uncorrected = get_measured_frame_timestamps(estimated_timestamps, measured_timestamps, latency_estimate, search_radius)
        self.assertEqual(len(corrected), len(estimated_timestamps))
        self.assertEqual(corrected[500], corrected[501])
        self.assertEqual(len(uncorrected), len(corrected))
        self.assertEqual(np.count_nonzero(np.isnan(uncorrected)), 1)

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
expected_wait_events = [
                        ('wait', 0.),
                        ('wait',5.),
                        ('wait',10.),
                        ('wait',18.),
]

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
            (NUM_WAIT,14)
]
class EventFilterTests(unittest.TestCase):

    def test_get_matching_events(self):
        wait_events_in_list = get_matching_events(event_log_events_in_str, 'wait')
        assert wait_events_in_list == expected_wait_events

    def test_get_event_occurrences(self):
        # Events as strings
        NUM_REWARD_OCCURANCES = 2
        reward_counts = get_event_occurrences(event_log_events_in_str,'reward')
        assert reward_counts == NUM_REWARD_OCCURANCES

        # Events as numbers
        reward_counts = get_event_occurrences(event_log_with_events_in_number, NUM_REWARD)
        assert reward_counts == NUM_REWARD_OCCURANCES

        # A missing event
        reward_counts = get_event_occurrences(event_log_events_in_str, 'banana')
        assert reward_counts == 0

    def test_calc_events(self):
        # Events as strings
        EVENT_LOG_DURATION = 18.0
        NUM_TAREGET_OCCURANCES = 3
        NUM_REWARD_OCCURANCES = 2
        REWARD_RATE = NUM_REWARD_OCCURANCES / EVENT_LOG_DURATION
        TARGET_RATE = NUM_TAREGET_OCCURANCES / EVENT_LOG_DURATION

        expected_duration = calc_events_duration(event_log_events_in_str)
        np.testing.assert_allclose(EVENT_LOG_DURATION,
                                    expected_duration)
        expected_target_rate = calc_event_rate(event_log_events_in_str, 'target')
        np.testing.assert_almost_equal(TARGET_RATE, expected_target_rate)

        expected_reward_rate = calc_reward_rate(event_log_events_in_str, 'reward')
        np.testing.assert_almost_equal(REWARD_RATE, expected_reward_rate)

        # Events as numbers
        EVENT_LOG_DURATION = 10.0
        NUM_TAREGET_OCCURANCES = 2
        NUM_REWARD_OCCURANCES = 2
        REWARD_RATE = NUM_REWARD_OCCURANCES / EVENT_LOG_DURATION
        TARGET_RATE = NUM_TAREGET_OCCURANCES / EVENT_LOG_DURATION
        np.testing.assert_almost_equal(EVENT_LOG_DURATION,
                                        calc_events_duration(event_log_with_events_in_number))
        
        expected_target_rate = calc_event_rate(event_log_with_events_in_number, NUM_TARGET)
        np.testing.assert_almost_equal(TARGET_RATE, expected_target_rate)

        expected_reward_rate = calc_reward_rate(event_log_with_events_in_number, NUM_REWARD)
        np.testing.assert_almost_equal(REWARD_RATE, expected_reward_rate)

        expected_reward_rate = calc_reward_rate(event_log_with_events_in_number, NUM_REWARD)
        np.testing.assert_almost_equal(REWARD_RATE, expected_reward_rate)

        # Missing events
        rate = calc_event_rate(event_log_events_in_str, 'foobar')
        assert rate == 0

    def test_trial_align_events(self):
        # test trial_separate
        events = np.array([6, 5, 2, 7, 2, 5, 7, 4, 2, 3, 6, 2, 3, 6, 4, 6, 3, 1, 3, 2, 4, 2,
            6, 4, 5, 5, 0, 3, 2, 4, 2, 4, 2, 5, 3, 2, 4, 0, 5, 2, 2, 7, 4, 6,
            3, 0, 6, 0, 1, 2, 3, 5, 3, 1, 4, 1, 2, 2, 7, 1, 1, 0, 6, 0, 1, 7,
            4, 5, 3, 3, 2, 4, 4, 1, 1, 5, 2, 3, 1, 4, 0, 5, 0, 0, 4, 2, 2, 6,
            3, 4, 0, 0, 1, 6, 5, 2, 1, 0, 7, 0])
        times = np.arange(0, 10, 0.1)

        expected_aligned_events = np.array([[2., 7.],
            [2., 5.],
            [2., 3.],
            [2., 3.],
            [2., 4.],
            [2., 6.],
            [2., 4.],
            [2., 4.],
            [2., 5.],
            [2., 4.],
            [2., 2.],
            [2., 7.],
            [2., 3.],
            [2., 2.],
            [2., 7.],
            [2., 4.],
            [2., 3.],
            [2., 2.],
            [2., 6.],
            [2., 1.]])
        
        expected_aligned_times = np.array([[0.2, 0.3],
            [0.4, 0.5],
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
            [9.5, 9.6]])
        
        NUM_TO_ALIGN = 2
        aligned_events, aligned_times = trial_separate(events, times, NUM_TO_ALIGN, n_events=2)

        np.testing.assert_allclose(expected_aligned_events, aligned_events)
        np.testing.assert_allclose(expected_aligned_times, aligned_times)

        expected_aligned_times = np.array([[0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0. ],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0. ],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0.1],
            [0. , 0. ],
            [0. , 0.1],
            [0. , 0.1]])

        trial_aligned_times = trial_align_events(aligned_events, aligned_times, NUM_TO_ALIGN)
        np.testing.assert_allclose(expected_aligned_times, trial_aligned_times)

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
        data = np.arange(100)
        samplerate = 1
        time_before = 0
        time_after = 10
        trigger_times = np.array([5, 55])
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        self.assertEqual(len(trial_aligned), len(trigger_times))
        self.assertTrue(np.allclose(trial_aligned[0], np.arange(5, 15)))
        self.assertTrue(np.allclose(trial_aligned[1], np.arange(55, 65)))
        data = np.ones((100,2))
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        self.assertEqual(trial_aligned.shape, (len(trigger_times), time_after, 2))

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

class TestPrepareExperiment(unittest.TestCase):

    def test_parse_bmi3d(self):
        files = {}
        self.assertRaises(Exception, lambda: parse_bmi3d(data_dir, files))
        files['bmi3d'] = 'test20210310_08_te1039.hdf'
        data, metadata = parse_bmi3d(data_dir, files)
        self.assertIn('bmi3d_fps', metadata)
        self.assertAlmostEqual(metadata['bmi3d_fps'], 120.)
        files['bmi3d'] = 'test20210310_08_te1039.hdf'
        data, metadata = parse_bmi3d(data_dir, files)

    def test_parse_optitrack(self):
        files = {}
        files['optitrack'] = 'Take 2021-04-06 11_47_54 (1312).csv'
        data, metadata = parse_optitrack(data_dir, files)

    def test_proc_exp(self):
        result_filename = 'test_proc_exp.hdf'
        files = get_filenames(data_dir, 1315)
        proc_exp(data_dir, files, write_dir, result_filename, overwrite=True)
        bmi3d_cycles = load_hdf_data(write_dir, result_filename, 'bmi3d_cycles')
        optitrack = load_hdf_data(write_dir, result_filename, 'optitrack')
        reward_system = load_hdf_data(write_dir, result_filename, 'reward_system')


if __name__ == "__main__":
    unittest.main()
