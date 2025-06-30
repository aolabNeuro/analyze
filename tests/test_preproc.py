from matplotlib import pyplot as plt
from aopy import visualization
from aopy import preproc
from aopy import analysis
from aopy.preproc import *
from aopy.preproc.bmi3d import *
from aopy.preproc import oculomatic
from aopy.preproc import quality
from aopy.data import *
import numpy as np
import unittest
from pathlib import Path
import datetime
from aopy.preproc.wrappers import proc_emg
import shutil


test_dir = os.path.dirname(__file__)
data_dir = os.path.join(test_dir, 'data')
write_dir = os.path.join(test_dir, 'tmp')
docs_dir = os.path.join(test_dir, '../docs/source/_images')
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

    def test_validate_measurements(self):
        expected_values = [1, 2, 3]
        measured_values = [1.5, 2., 2]
        diff_thr = 0.5
        corrected_values, above_thr = validate_measurements(expected_values, measured_values, diff_thr)
        np.testing.assert_allclose(corrected_values, np.array([1.5, 2., 3.]))
        self.assertEqual(np.count_nonzero(above_thr), 1)

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

        # Test invalid inputs:
        timestamps = np.array([1,2,3,4])
        timestamp_values = np.array([100,200,100,300])
        fun = lambda: interp_timestamps2timeseries(timestamps, timestamp_values) # not enough inputs
        self.assertRaises(ValueError, fun)
        fun = lambda: interp_timestamps2timeseries(timestamps, timestamp_values, samplerate=2, interp_kind='foobar') # invalid method
        self.assertRaises(Exception, fun)
        
        # Test non-monotonic input timestamps
        timestamps = np.array([0,2,1,4])
        timeseries, t = interp_timestamps2timeseries(timestamps, timestamp_values, samplerate=2)
        self.assertTrue(len(timeseries) > 0)
        self.assertEqual(np.count_nonzero(np.isnan(timeseries)), 0)
        timestamps = np.array([0,1,1,4])
        timeseries, t = interp_timestamps2timeseries(timestamps, timestamp_values, samplerate=2)
        self.assertTrue(len(timeseries) > 0)
        self.assertEqual(np.count_nonzero(np.isnan(timeseries)), 0)

        # Test extrapolate
        timestamps = np.array([1,2,3,4])
        timestamp_values = np.array([100,200,100,300])
        expected_timeseries = np.array([50, 100,150,200,150,100,200,300, 400, 500])
        input_samplepts = np.array([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
        timeseries, new_samplepts = interp_timestamps2timeseries(timestamps, timestamp_values, sampling_points=input_samplepts, extrapolate=True)
        np.testing.assert_allclose(timeseries, expected_timeseries)
        np.testing.assert_allclose(new_samplepts, input_samplepts)
        expected_timeseries = np.array([100, 100,150,200,150,100,200,300, 300,300])
        timeseries, new_samplepts = interp_timestamps2timeseries(timestamps, timestamp_values, sampling_points=input_samplepts, extrapolate=False)
        np.testing.assert_allclose(timeseries, expected_timeseries)
        np.testing.assert_allclose(new_samplepts, input_samplepts)

    def test_sample_timestamped_data(self):
        np.random.seed(0)
        duration = 4
        freq = 5
        samplerate = 25000
        ground_truth_data = utils.generate_multichannel_test_signal(duration*2, samplerate, 2, freq, 1)
        
        fps = 120
        nt = fps*duration
        offset = 0.01 # timestamps start strictly after time zero
        framerate_error = 0.01*np.random.uniform(size=(nt,)) # 10 ms jitter
        drift = np.cumsum(0.0001*np.random.uniform(size=(nt,))) # 0.1 ms drift
        timestamps = offset + np.arange(nt)/fps + framerate_error + drift
        samples = (timestamps * samplerate).astype(int)

        frame_data = ground_truth_data[samples,:]
        interp_samplerate = 120
        interp_data = sample_timestamped_data(frame_data, timestamps, interp_samplerate)

        fig, ax = plt.subplots(3,1, figsize=(5,6))
        visualization.plot_timeseries(frame_data[:,0], fps, ax=ax[0])
        visualization.plot_timeseries(interp_data[:,0], interp_samplerate, ax=ax[0])
        ax[0].set_title(f'{freq} Hz signal')
        ax[0].set_ylabel('pos (cm)')
        ax[0].legend(['without sampling', 'with sampling'])

        visualization.plot_freq_domain_amplitude(frame_data[:,0], fps, ax=ax[1])
        visualization.plot_freq_domain_amplitude(interp_data[:,0], interp_samplerate, ax=ax[1])
        ax[1].set_xscale('linear')
        ax[1].set_xlim(0,30)
        ax[1].set_ylabel('Peak amplitude')
        ax[1].legend(['without sampling', 'with sampling'])
        plt.tight_layout()

        # Compare with different upsampling rates
        visualization.plot_timeseries(ground_truth_data[:,0], samplerate, ax=ax[2])
        interp_data = sample_timestamped_data(frame_data, timestamps, interp_samplerate, upsamplerate=120)
        visualization.plot_timeseries(interp_data[:,0], interp_samplerate, ax=ax[2])
        interp_data = sample_timestamped_data(frame_data, timestamps, interp_samplerate, upsamplerate=120*100)
        visualization.plot_timeseries(interp_data[:,0], interp_samplerate, ax=ax[2])
        ax[2].set_xlim(0.0,0.3)
        ax[2].legend(['original', 'no upsample', 'upsample to 10,000 Hz'])

        filename = 'sample_timestamped_data.png'
        visualization.savefig(docs_dir, filename, transparent=False)

    def test_get_dch_data(self):
        dig_data = [0, 1, 0, 1, 1, 0]
        samplerate = 1
        expected_ts = [1, 2, 3, 5]
        expected_values = [1, 0, 1, 0]
        data = get_dch_data(dig_data, samplerate, 0)
        np.testing.assert_allclose(expected_ts, data['timestamp'])
        np.testing.assert_allclose(expected_values, data['value'])

        dig_data = [0, 3, 0, 0, 2]
        samplerate = 1
        expected_ts = [1, 2, 4]
        expected_values = [3, 0, 2]
        data = get_dch_data(dig_data, samplerate, [0,1])
        np.testing.assert_allclose(expected_ts, data['timestamp'])
        np.testing.assert_allclose(expected_values, data['value'])
        

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
        self.assertEqual(trial_aligned.shape[2], len(trigger_times))
        np.testing.assert_allclose(np.squeeze(trial_aligned[:,:,0]), np.arange(5, 15))
        np.testing.assert_allclose(np.squeeze(trial_aligned[:,:,1]), np.arange(55, 65))
        
        # Test with nonzero time_before
        # Data looks like: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 ...
        # Trigger times:             *                              ...
        # Aligned trials:        3 4 5 6 7 8 9 10 11 12 13 14 15    ...
        time_before = 2
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        self.assertEqual(trial_aligned.shape[2], len(trigger_times))
        np.testing.assert_allclose(np.squeeze(trial_aligned[:,:,0]), np.arange(3, 15))
        np.testing.assert_allclose(np.squeeze(trial_aligned[:,:,1]), np.arange(53, 65))
        
        # Test shape is consistent with more dimensions in data
        data = np.ones((100,2))
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        self.assertEqual(trial_aligned.shape, (time_before + time_after, 2, len(trigger_times)))
        
        # Test single trial
        data = np.ones((100,1))
        trigger_times = [5]
        trial_aligned = trial_align_data(data, trigger_times, time_before, time_after, samplerate)
        self.assertEqual(trial_aligned.shape, (time_before + time_after, 1, 1))
        
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
        np.testing.assert_allclose(np.squeeze(trial_aligned[:,:,0]), np.arange(5,15))
        self.assertTrue(np.count_nonzero(np.isnan(trial_aligned[:,:,1])), 15)

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
        for t in range(aligned_data.shape[2]):
            t = aligned_data[:,:,t]
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

    def test_get_trial_segments_and_times(self):
        events = [0, 2, 4, 6, 0, 2, 3, 6]
        event_times = [0, 1, 2, 3, 4, 5, 6, 7]
        start_evt = 2
        end_evt = 6
        segments, times = get_trial_segments_and_times(events, event_times,  start_evt, end_evt)
        self.assertTrue(np.allclose(segments, [[2, 4, 6], [2, 3, 6]]))
        self.assertTrue(np.allclose(times, [[1, 2, 3], [5, 6, 7]]))
        
        # Test return_idx
        segments, times, idx = get_trial_segments_and_times(events, event_times,  start_evt, end_evt, return_idx=True)
        self.assertTrue(np.allclose(idx, [[1, 2, 3], [5, 6, 7]]))

        # Test repeating start events
        events = [18, 82, 19, 34, 83, 64, 239, 18, 82, 19, 34, 83, 48, 239, 19, 83, 18, 66, 239]
        event_times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        start_evt = [18, 19]
        end_evt = [239]
        expected_segments = [[18, 82, 19, 34, 83, 64, 239], [18, 82, 19, 34, 83, 48, 239], [19, 83, 18, 66, 239]]
        expected_times = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18]]

        segments, times = get_trial_segments_and_times(events, event_times,  start_evt, end_evt, repeating_start_events=True)
        for i in range(len(segments)):
            np.testing.assert_allclose(segments[i], expected_segments[i])
            np.testing.assert_allclose(times[i], expected_times[i])

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
        event_times = [2, 3, 3.5, 3.6, 3.7, 4, 5]
        event_codes = [0, 1, 0, 2, 1, 0, 1]
        coeff, corr_coeff = calc_eye_calibration(cursor_data, cursor_samplerate, eye_data, eye_samplerate, 
            event_times, event_codes, align_events=[0], penalty_events=[2])
        
        expected_coeff = [[1., 1.], [1., -1.]]
        expected_corr_coeff = [1., 1.]

        np.testing.assert_allclose(expected_coeff, coeff)
        np.testing.assert_allclose(expected_corr_coeff, corr_coeff)
        
    def test_calc_eye_target_calibration(self):
        eye_data = np.array([[0, 1, 2, 3, 4, 5, 6, 5], [3, 4, 5, 6, 7, 8, 9, 8]]).T
        eye_samplerate = 1
        event_times = [1, 2, 3, 4, 5, 6, 7]
        event_codes = [1, 3, 1, 4, 2, 3, 2]
        target_pos_test = np.array([[0,0,0],[1,1,0]])
        expected_coeff = [[0.25, -0.25], [0.25, -1]]
        expected_corr_coeff = [1., 1.]
        # test for duration = 0.1
        coeff, corr_coeff = calc_eye_target_calibration(eye_data, eye_samplerate, 
            event_times, event_codes, target_pos_test, align_events=[1,2], penalty_events=[4], cursor_enter_center_target=0, duration=1)
        np.testing.assert_allclose(expected_coeff, coeff)
        np.testing.assert_allclose(expected_corr_coeff, corr_coeff)
        
        # test for duration = 0.
        coeff, corr_coeff = calc_eye_target_calibration(eye_data, eye_samplerate, 
            event_times, event_codes, target_pos_test, align_events=[1,2], penalty_events=[4], cursor_enter_center_target=0, duration=0)
        np.testing.assert_allclose(expected_coeff, coeff)
        np.testing.assert_allclose(expected_corr_coeff, corr_coeff)

class TestPrepareExperiment(unittest.TestCase):

    def check_required_fields(self, data, metadata):
        self.assertIn('fps', metadata)
        self.assertIn('sync_protocol_version', metadata)
        self.assertIn('source_dir', metadata)
        self.assertIn('source_files', metadata)
        self.assertIn('clock', data)
        self.assertIn('events', data)
        self.assertIn('task', data)
        self.assertIsNotNone(data['clock'])
        self.assertIsNotNone(data['events'])
        self.assertIsNotNone(data['task'])
        self.assertIn('time', data['clock'].dtype.names)
        self.assertIn('timestamp_bmi3d', data['clock'].dtype.names)
        self.assertIn('cursor', data['task'].dtype.names)
        self.assertIn('timestamp_bmi3d', data['events'].dtype.names)
        self.assertIn('timestamp', data['events'].dtype.names)
        self.assertIn('event', data['events'].dtype.names)
        self.assertIn('code', data['events'].dtype.names)
        self.assertIn('data', data['events'].dtype.names)

    def test_decode_events(self):
        dictionary = {
            'event1': 1,
            'event2': 3,
            'event3': 10
        }
        values = [1, 2, 3, 10]
        event_names, event_data = decode_events(dictionary, values)
        self.assertEqual(len(event_names), len(values))
        self.assertEqual(event_names[0], 'event1')
        self.assertEqual(event_names[1], 'event1')
        self.assertEqual(event_names[2], 'event2')
        self.assertEqual(event_names[3], 'event3')
        self.assertEqual(event_data[0], 0)
        self.assertEqual(event_data[1], 1)
        self.assertEqual(event_data[2], 0)
        self.assertEqual(event_data[3], 0)

    def test_parse_bmi3d_empty(self):
        files = {}
        data, metadata = parse_bmi3d(data_dir, files)
        self.assertRaises(AssertionError, lambda: self.check_required_fields(data, metadata))

    def test_parse_bmi3d_v0(self):
        # Test sync version 0 (and -1)
        files = {}
        files['hdf'] = 'test20210310_08_te1039.hdf'
        data, metadata = parse_bmi3d(data_dir, files)
        self.check_required_fields(data, metadata)
        self.assertEqual(metadata['sync_protocol_version'], -1)
        self.assertIn('fps', metadata)
        self.assertAlmostEqual(metadata['fps'], 120.)
        self.assertIn('timestamp_bmi3d', data['clock'].dtype.names)
        n_cycles = data['clock']['time'][-1] + 1
        self.assertEqual(len(data['clock']), n_cycles)

        # Test tablet data (sync version 0)
        files = {}
        files['hdf'] = 'chur20231002_02_te375.hdf'
        data, metadata = parse_bmi3d(data_dir, files)
        self.check_required_fields(data, metadata)
        self.assertEqual(metadata['sync_protocol_version'], 0)
        self.assertIn('fps', metadata)
        self.assertAlmostEqual(metadata['fps'], 120.)
        self.assertIn('timestamp_bmi3d', data['clock'].dtype.names)
        n_cycles = data['clock']['time'][-1] + 1
        self.assertEqual(len(data['clock']), n_cycles)     
        for key in ['timestamp','code','event']:
            self.assertIn(key, data['events'].dtype.names)

    def test_parse_bmi3d_v1(self):
        pass

    def test_parse_bmi3d_v2(self):
        # Test sync version 2 
        files = {}
        files['hdf'] = 'beig20210407_01_te1315.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        self.check_required_fields(data, metadata)
        trials = data['bmi3d_trials']
        self.assertEqual(len(trials), 3)        
        files['ecube'] = '2021-04-07_BMI3D_te1315'
        data, metadata = parse_bmi3d(data_dir, files) # and with ecube data
        self.check_required_fields(data, metadata)
        self.assertEqual(metadata['sync_protocol_version'], 2)
        self.assertIn('sync_clock', data)
        self.assertIn('measure_clock_offline', data)
        self.assertEqual(len(data['measure_clock_offline']['timestamp']), 1042)
        self.assertEqual(len(data['measure_clock_online']['timestamp']), 1015)
        self.assertTrue(metadata['has_measured_timestamps'])
        self.assertIn('timestamp_bmi3d', data['clock'].dtype.names)
        self.assertIn('timestamp_sync', data['clock'].dtype.names)
        self.assertIn('timestamp', data['sync_events'].dtype.names)
        n_cycles = data['clock']['time'][-1] + 1
        # self.assertEqual(len(data['clock']), n_cycles)

    def test_parse_bmi3d_v3(self):
        pass

    def test_parse_bmi3d_v4(self):        
        # Test sync version 4
        files = {}
        files['hdf'] = 'beig20210614_07_te1825.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        self.check_required_fields(data, metadata)
        trials = data['bmi3d_trials']
        self.assertEqual(len(trials), 7)        
        files['ecube'] = '2021-06-14_BMI3D_te1825'
        data, metadata = parse_bmi3d(data_dir, files) # and with ecube data
        self.check_required_fields(data, metadata)
        self.assertEqual(metadata['sync_protocol_version'], 4)
        self.assertIn('sync_clock', data)
        self.assertIn('measure_clock_offline', data)
        self.assertEqual(len(data['measure_clock_offline']['timestamp']), 1754)
        self.assertEqual(len(data['measure_clock_online']), 1682)
        self.assertTrue(metadata['has_measured_timestamps'])
        self.assertIn('timestamp_bmi3d', data['clock'].dtype.names)
        self.assertIn('timestamp_sync', data['clock'].dtype.names)
        self.assertIn('timestamp', data['sync_events'].dtype.names)
        n_cycles = data['clock']['time'][-1]
        self.assertEqual(len(data['clock']), n_cycles)
        self.assertIn('clock', data)
  
    def test_parse_bmi3d_v5(self):
        pass

    def test_parse_bmi3d_v6(self):
        pass

    def test_parse_bmi3d_v7(self):
        # Test sync version 7
        files = {}
        files['hdf'] = 'fake_ecube_data_bmi3d.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        self.check_required_fields(data, metadata)
        files['ecube'] = '2021-12-13_BMI3D_te3498'
        data, metadata = parse_bmi3d(data_dir, files) # and with ecube data
        self.assertEqual(metadata['sync_protocol_version'], 7)
        self.assertEqual(len(data['clock']), 4848)
        # self.assertEqual(len(data['events']), 66) # seems to be 67
        print(metadata['n_missing_markers'])
        self.assertTrue(metadata['has_measured_timestamps'])
        evt = data['events'][27]
        self.assertEqual(evt['code'], 87)

        # Run some trial alignment to make sure the number of trials makes sense
        events = data['events']
        start_states = [16]
        end_states = [239] 
        trial_states, trial_idx = get_trial_segments(events['code'], events['timestamp'], start_states, end_states)
        self.assertEqual(len(trial_states), 10)

        # Test a file with no sync events to make sure we're not breaking things
        files['hdf'] = 'beig20210930_02_te2952.hdf'
        files['ecube'] = '2021-09-30_BMI3D_te2952'
        data, metadata = parse_bmi3d(data_dir, files) # with ecube data
        self.assertEqual(len(data['sync_events']), 3)

    def test_parse_bmi3d_v8(self):
        pass

    def test_parse_bmi3d_v9(self):
        files = {}
        files['hdf'] = 'test20220311_07_te4298.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        self.check_required_fields(data, metadata)
        files['ecube'] = '2022-03-11_BMI3D_te4298'
        data, metadata = parse_bmi3d(data_dir, files) # and with ecube data
        self.assertEqual(metadata['sync_protocol_version'], 9)

        # This file has analog voltage from photodiode recorded showing the screen turn on and off
        lfp_data, metadata = aodata.load_ecube_analog(data_dir, files['ecube'], channels=[31])
        n_channels = metadata['n_channels']
        raw_samplerate = metadata['samplerate']
        samplerate = 1000
        lfp_data = precondition.downsample(lfp_data, raw_samplerate, samplerate)
        time_before = 0.1
        time_after = 0.4

        plt.figure()
        visualization.plot_timeseries(lfp_data, samplerate)
        filename = 'parse_bmi3d_downsample.png'
        visualization.savefig(write_dir, filename)

        # Plot aligned flash times based on events
        event_timestamps = data['events']['timestamp']
        flash_times = event_timestamps[np.logical_and(16 <= data['events']['code'], data['events']['code'] < 32)]
        evoked_lfp = analysis.calc_erp(lfp_data, flash_times, time_before, time_after, samplerate)
        time = np.arange(evoked_lfp.shape[0])/samplerate - time_before
        plt.figure()
        im = visualization.plot_image_by_time(time, evoked_lfp[:,0,:])
        im.set_clim(-300, 300)
        plt.colorbar(im, label='uV')        
        filename = 'parse_bmi3d_flash_events.png'
        visualization.savefig(docs_dir, filename)

        # Plot aligned flash times based on sync clock
        target_on_events = np.logical_and(16 <= data['bmi3d_events']['code'], data['bmi3d_events']['code'] < 32)
        flash_times = data['clock']['timestamp_sync'][data['bmi3d_events']['time'][target_on_events]]
        evoked_lfp = analysis.calc_erp(lfp_data, flash_times, time_before, time_after, samplerate)
        plt.figure()
        im = visualization.plot_image_by_time(time, evoked_lfp[:,0,:])
        im.set_clim(-300, 300)
        plt.colorbar(im, label='uV')
        filename = 'parse_bmi3d_flash_sync_clock.png'
        visualization.savefig(docs_dir, filename)

        # Plot aligned flash times based on measure clock
        target_on_events = np.logical_and(16 <= data['bmi3d_events']['code'], data['bmi3d_events']['code'] < 32)
        flash_times = data['clock']['timestamp_measure_offline'][data['bmi3d_events']['time'][target_on_events]]
        evoked_lfp = analysis.calc_erp(lfp_data, flash_times, time_before, time_after, samplerate)
        plt.figure()
        im = visualization.plot_image_by_time(time, evoked_lfp[:,0,:])
        im.set_clim(-300, 300)
        plt.colorbar(im, label='uV')
        filename = 'parse_bmi3d_flash_measure_clock.png'
        visualization.savefig(docs_dir, filename)


    def test_parse_bmi3d_v10(self):
        pass

    def test_parse_bmi3d_v11(self):
        files = {}
        files['hdf'] = 'test20220524_11_te5351.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        self.check_required_fields(data, metadata)
        files['ecube'] = '2022-05-24_BMI3D_te5351'

        # Reduce the file size so we can upload it to github
        # analog_data, metadata = aodata.load_ecube_analog(data_dir, files['ecube'])
        # analog_data = analog_data[:,:16]
        # filename = utils.save_test_signal_ecube(analog_data, data_dir, 1, datasource='AnalogPanel')

        data, metadata = parse_bmi3d(data_dir, files) # and with ecube data
        self.assertEqual(metadata['sync_protocol_version'], 11)
        self.assertIn('cursor_analog_cm', data)
        self.assertIn('cursor_interp', data)
        self.assertIn('cursor_interp_samplerate', metadata)
        self.assertIn('clean_hand_position', data)

        # Plot the cursor data
        bounds = np.array(metadata['cursor_bounds'])
        bounds = bounds[[0,1,4,5,2,3]] # (-x, x, -z, z, -y, y) -> (-x, x, -y, y, -z, z)
        plt.figure()
        visualization.plot_trajectories([data['cursor_analog_cm']], bounds)
        trials = data['bmi3d_trials']
        trial_targets = postproc.get_trial_targets(trials['trial'], trials['target'][:,[0,2,1]]) # (x, z, y) -> (x, y, z)
        unique_targets = np.unique(np.vstack(trial_targets), axis=0)
        visualization.plot_targets(unique_targets, metadata['target_radius'])
        filename = 'parse_bmi3d_cursor_v11.png'
        visualization.savefig(write_dir, filename)

        # Run some trial alignment to make sure the number of trials makes sense
        events = data['events']
        start_states = [32] # center target off
        end_states = [239] # trial end
        trial_states, trial_times = get_trial_segments(events['code'], events['timestamp'], start_states, end_states)
        trajectories = get_data_segments(data['cursor_analog_cm'], trial_times, metadata['analog_samplerate'])
        plt.figure()
        visualization.plot_trajectories(trajectories, bounds)
        trials = data['bmi3d_trials']
        visualization.plot_targets(unique_targets, metadata['target_radius'])
        filename = 'parse_bmi3d_cursor_trajectories_v11.png'
        visualization.savefig(write_dir, filename)

        trajectories = get_data_segments(data['cursor_interp'], trial_times, metadata['cursor_interp_samplerate'])
        plt.figure()
        visualization.plot_trajectories(trajectories, bounds)
        trials = data['bmi3d_trials']
        visualization.plot_targets(unique_targets, metadata['target_radius'])
        filename = 'parse_bmi3d_cursor_trajectories_interp_v11.png'
        visualization.savefig(write_dir, filename)

        trajectories = get_data_segments(data['cursor_analog_cm'], trial_times, metadata['cursor_interp_samplerate'])
        plt.figure()
        visualization.plot_trajectories(trajectories, bounds)
        trials = data['bmi3d_trials']
        visualization.plot_targets(unique_targets, metadata['target_radius'])
        filename = 'parse_bmi3d_cursor_trajectories_filt_v11.png'
        visualization.savefig(write_dir, filename)

    def test_parse_bmi3d_v12(self):
        files = {}
        files['hdf'] = 'beig20221002_09_te6890.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        self.check_required_fields(data, metadata)
        files['ecube'] = '2022-10-02_BMI3D_te6890'

        # This is a laser only experiment which does not contain any task data
        data, metadata = parse_bmi3d(data_dir, files) # and with ecube data
        self.assertEqual(metadata['sync_protocol_version'], 12)

    def test_parse_bmi3d_v13(self):
        files = {}
        files['hdf'] = 'beig20230109_15_te7977.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        self.check_required_fields(data, metadata)
        files['ecube'] = '2023-01-09_BMI3D_te7977'

        # Reduce the file size so we can upload it to github
        # analog_data, metadata = aodata.load_ecube_analog(data_dir, files['ecube'])
        # analog_data = analog_data[:25000*60,:8]
        # filename = utils.save_test_signal_ecube(analog_data, data_dir, 1, datasource='AnalogPanel')

        # There is a pause bug in this task
        data, metadata = parse_bmi3d(data_dir, files) # with ecube data
        self.assertNotEqual(len(data['sync_events']), len(data['bmi3d_events']))

        # Test what happens if no HDF file is provided
        n_events = len(data['sync_events'])
        files = {}
        files['ecube'] = '2023-01-09_BMI3D_te7977'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        self.assertRaises(AssertionError, lambda: self.check_required_fields(data, metadata))

        # Test if a dummy HDF file is provided
        files['hdf'] = '../tmp/dummy_hdf_rig1_v13.hdf'

        # Make a dummy HDF file by copying an existing HDF file from the same experiment
        import shutil
        shutil.copy(os.path.join(data_dir, 'beig20230109_15_te7977.hdf'), os.path.join(data_dir, files['hdf']))
        with tables.open_file(os.path.join(data_dir, files['hdf']), 'r+') as f:
            f.get_node('/task').truncate(0)
            f.get_node('/sync_events').truncate(0)
            f.get_node('/sync_clock').truncate(0)

        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        self.assertEqual(n_events, len(data['events']))

    def test_parse_bmi3d_v14(self):
        files = {}
        files['hdf'] = 'beig20231229_21_te13154.hdf'
        data, metadata = parse_bmi3d(data_dir, files) # without ecube data
        self.check_required_fields(data, metadata)
        files['ecube'] = '2023-12-29_BMI3D_te13154'
        data, metadata = parse_bmi3d(data_dir, files) # with ecube data
        self.assertEqual(len(data['sync_events']), len(data['bmi3d_events']))
        self.check_required_fields(data, metadata)

    def test_parse_bmi3d_v16(self):
        files = {}
        files['hdf'] = 'leo20250213_41_te1966.hdf'
        data, metadata = parse_bmi3d(data_dir, files)
        self.check_required_fields(data, metadata)
        self.assertEqual(metadata['sync_protocol_version'], 0)

        files['hdf'] = 'vrc020250410_02_te2677.hdf'
        data, metadata = parse_bmi3d(data_dir, files)
        self.check_required_fields(data, metadata)
        self.assertEqual(metadata['sync_protocol_version'], 0)
        self.assertIn('clean_hand_position', data.keys())

    def test_parse_bmi3d_v17(self):
        
        emg_data_dir = os.path.join(data_dir, 'quatt_emg')
        files = {'emg':'aj20250319_05_te2540_emg.hdf',
                 'hdf':'aj20250319_05_te2540_exp.hdf'
        }

        data, metadata = parse_bmi3d(emg_data_dir, files)

        self.check_required_fields(data, metadata)
        self.assertEqual(metadata['sync_protocol_version'], 17)


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
        self.assertRaises(ValueError, lambda: proc_eyetracking(data_dir, files, write_dir, result_filename, result_filename))

        proc_exp(data_dir, files, write_dir, result_filename, result_filename)

        # Not enough trials in this session to calibrate, so only raw data should be processed
        eye, meta = proc_eyetracking(data_dir, files, write_dir, result_filename, result_filename, save_res=False)
        self.assertIsNotNone(eye)
        self.assertIsNotNone(meta)
        self.assertIn('raw_data', eye)
        self.assertIn('eye_closed_mask', eye)
        self.assertIn('samplerate', meta)

        # This dataset has more trials
        result_filename = 'test_proc_eyetracking.hdf'
        files['ecube'] = '2021-09-29_BMI3D_te2949'
        files['hdf'] = 'beig20210929_02_te2949.hdf'
        if os.path.exists(os.path.join(write_dir, result_filename)):
            os.remove(os.path.join(write_dir, result_filename))
        exp_data, exp_metadata = proc_exp(data_dir, files, write_dir, result_filename)

        # Test that eye calibration is returned, but results are not saved
        eye, meta = proc_eyetracking(data_dir, files, write_dir, result_filename, result_filename, save_res=False)
        self.assertIsNotNone(eye)
        self.assertIsNotNone(meta)
        self.assertRaises(ValueError, lambda: load_hdf_group(write_dir, result_filename, 'eye_data'))
        self.assertRaises(ValueError, lambda: load_hdf_group(write_dir, result_filename, 'eye_metadata'))

        # Test that eye calibration is saved
        proc_eyetracking(data_dir, files, write_dir, result_filename, result_filename, save_res=True)
        eye = load_hdf_group(write_dir, result_filename, 'eye_data')
        meta = load_hdf_group(write_dir, result_filename, 'eye_metadata')
        self.assertIsNotNone(eye)
        self.assertIsNotNone(meta)
        self.assertIn('raw_data', eye)
        self.assertIn('calibrated_data', eye)
        self.assertGreater(eye['correlation_coeff'][0], 0.5)
        self.assertGreater(eye['correlation_coeff'][1], 0.5)
        self.assertGreater(eye['correlation_coeff'][2], 0.5)
        self.assertGreater(eye['correlation_coeff'][3], 0.5)
        self.assertIn('samplerate', meta)

        # This is a more recent dataset
        # result_filename = 'test_proc_eyetracking_220422.hdf'
        # files['ecube'] = '2022-04-22_BMI3D_te5062'
        # files['hdf'] = 'beig20220422_03_te5062.hdf'

        # # Some code to remove unneeded analog channels to reduce the file size
        # eye_data, metadata = aodata.load_ecube_analog(data_dir, files['ecube'])
        # eye_data = eye_data[:,:12]
        # filename = utils.save_test_signal_ecube(eye_data, data_dir, 1, datasource='AnalogPanel')

        if os.path.exists(os.path.join(write_dir, result_filename)):
            os.remove(os.path.join(write_dir, result_filename))
        exp_data, exp_metadata = proc_exp(data_dir, files, write_dir, result_filename)
        eye, meta = proc_eyetracking(data_dir, files, write_dir, result_filename, result_filename, save_res=False)

        # Plot calibrated eye data to make sure everything is working properly
        raw_data = eye['raw_data']
        bounds = np.array(exp_metadata['cursor_bounds'])[[0,1,4,5]]
        plt.figure()
        visualization.plot_trajectories([raw_data], bounds=bounds)
        figname = 'eye_trajectories_raw.png'
        visualization.savefig(docs_dir, figname) # should have uncalibrated eye data

        plt.figure()
        eye_data = eye['calibrated_data']
        visualization.plot_trajectories([eye_data], bounds=bounds)
        figname = 'eye_trajectories_calibrated.png'
        visualization.savefig(docs_dir, figname) # should have centered eye data

        # Test putting eye data into a separate HDF file
        eye_filename = 'test_proc_eyetracking_short_eye.hdf'
        proc_eyetracking(data_dir, files, write_dir, result_filename, eye_filename)
        eye = load_hdf_group(write_dir, eye_filename, 'eye_data')
        meta = load_hdf_group(write_dir, eye_filename, 'eye_metadata')
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

    def test_get_switched_stimulation_sites(self):
        subject = 'test'
        te_id = 15494
        date = '2024-03-19'

        # Some code to make the preprocessed file (raw data too big to store here)
        # files = {}
        # files['hdf'] = 'test20240319_08_te15494.hdf'
        # files['ecube'] = '2024-03-19_BMI3D_te15494'
        # proc_single(data_dir, files, data_dir, subject, te_id, date, ['exp'], overwrite=True)

        exp_data, exp_metadata = load_preproc_exp_data(data_dir, subject, te_id, date)
        times, _, _, _ = preproc.bmi3d._get_laser_trial_times(
            exp_data, exp_metadata, laser_trigger='qwalor_trigger', laser_sensor='qwalor_sensor'
        )
        sites = get_switched_stimulation_sites(data_dir, subject, te_id, date, times, debug=True)
        visualization.savefig(docs_dir, 'switched_stimulation_sites.png', transparent=False)

        np.testing.assert_allclose(sites, 0)

    def test_get_laser_trial_times(self):
        time_before = 0.05
        time_after = 0.05
        subject = 'test'
        te_id = 6581
        date = '2022-08-19'
        preproc_dir = data_dir

        # This preprocessed file doesn't contain laser sensor data so it will look for the raw data.
        # Since the raw data isn't availble in this repo we should get an error. Unless you're running
        # this test on the lab server...
        self.assertRaises(FileNotFoundError, lambda: get_laser_trial_times(preproc_dir, subject, te_id, date))
        
        # This other preprocessed file does contain laser sensor data. Response on ch. 36
        te_id = 6577
        exp_data, exp_metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)
        self.assertIn('laser_sensor', exp_data.keys())
        self.assertNotIn('qwalor_trigger_dch', exp_metadata.keys())
        self.assertNotIn('laser_trigger', exp_data.keys())

        trial_times, trial_widths, trial_gains, trial_powers = get_laser_trial_times(
            preproc_dir, subject, te_id, date, laser_trigger='laser_trigger', 
            laser_sensor='laser_sensor', debug=True)
        visualization.savefig(write_dir, 'laser_aligned_sensor_debug.png')

        lfp_data, lfp_metadata = load_preproc_lfp_data(preproc_dir, subject, te_id, date)
        
        # Plot lfp response
        ch = 36
        samplerate = lfp_metadata['lfp_samplerate']
        erp = analysis.calc_erp(lfp_data, trial_times, time_before, time_after, samplerate)
        erp_voltage = 1e6*lfp_metadata['voltsperbit']*np.mean(erp, axis=2)
        t = 1000*(np.arange(erp_voltage.shape[0])/samplerate - time_before)
        ch_data = 1e6*lfp_metadata['voltsperbit']*erp_voltage[:,ch]
        plt.figure()
        plt.plot(t, ch_data)
        plt.plot([0,0], [-3, 0], 'k--')
        visualization.savefig(write_dir, 'lfp_aligned_laser.png')
        
        # Also plot the individual trials
        plt.figure()
        im = visualization.plot_image_by_time(t, 1e6*lfp_metadata['voltsperbit']*erp[:,ch,:], ylabel='trials')
        im.set_clim(-100,100)
        visualization.savefig(docs_dir, 'laser_aligned_lfp.png')
        
        # And compare to the sensor data
        sensor_data = exp_data['laser_sensor']
        sensor_voltsperbit = exp_metadata['analog_voltsperbit']
        samplerate = exp_metadata['analog_samplerate']

        plt.figure()
        ds_data = precondition.downsample(sensor_data, samplerate, 1000)
        ds_data = ds_data - np.mean(ds_data)
        analog_erp = analysis.calc_erp(ds_data, trial_times, time_before, time_after, 1000)
        im = visualization.plot_image_by_time(t, sensor_voltsperbit*analog_erp[:,0,:], ylabel='trials')
        im.set_clim(-0.01,0.01)
        visualization.savefig(docs_dir, 'laser_aligned_sensor.png')

        plt.figure()
        plt.hist(trial_widths, 20)
        visualization.savefig(write_dir, 'laser_widths.png') # Should be all 0.005 s

        plt.figure()
        plt.hist(trial_powers, 20)
        visualization.savefig(write_dir, 'laser_powers.png') # Should be all 0.5

        # Another file, with digital and analog data
        subject = 'affi'
        te_id = 8844
        date = '2023-03-22'

        # Preprocess the raw data
        # files = {}
        # files['hdf'] = '/Users/leoscholl/raw/hdf/affi20230322_17_te8844.hdf'
        # files['ecube'] = '/Users/leoscholl/raw/ecube/2023-03-22_BMI3D_te8844'
        # # os.remove(os.path.join(data_dir, 'affi/preproc_2023-03-22_affi_8844_exp.hdf'))
        # proc_exp('', files, data_dir, 'affi/preproc_2023-03-22_affi_8844_exp.hdf', overwrite=True)
        # exp_data, exp_metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)

        # # Make the lfp file only contain 8 channels
        # lfp_data, lfp_metadata = load_preproc_lfp_data(preproc_dir, subject, te_id, date)
        # lfp_data = lfp_data[:,:8]
        # os.remove(os.path.join(data_dir, 'affi/preproc_2023-03-22_affi_8844_lfp.hdf'))
        # print('lfp data', lfp_data.nbytes)
        # save_hdf(data_dir, 'affi/preproc_2023-03-22_affi_8844_lfp.hdf', {'lfp_data': lfp_data})
        # save_hdf(data_dir, 'affi/preproc_2023-03-22_affi_8844_lfp.hdf', lfp_metadata, "/lfp_metadata", append=True)
        
        exp_data, exp_metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)
        self.assertIn('qwalor_trigger_dch', exp_metadata.keys())
        self.assertIn('qwalor_trigger', exp_data.keys())

        trial_times, trial_widths, trial_gains, trial_powers = get_laser_trial_times(preproc_dir, subject, te_id, date, debug=True)
        visualization.savefig(write_dir, 'laser_aligned_sensor_debug_dch_trigger.png')

        lfp_data, lfp_metadata = load_preproc_lfp_data(preproc_dir, subject, te_id, date)
        
        # Plot lfp response
        ch = 1
        samplerate = lfp_metadata['lfp_samplerate']
        erp = analysis.calc_erp(lfp_data, trial_times, time_before, time_after, samplerate)
        erp_voltage = 1e6*lfp_metadata['voltsperbit']*np.mean(erp, axis=2)
        t = 1000*(np.arange(erp_voltage.shape[0])/samplerate - time_before)
        ch_data = 1e6*lfp_metadata['voltsperbit']*erp_voltage[:,ch]
        plt.figure()
        plt.plot(t, ch_data)
        plt.plot([0,0], [-3, 0], 'k--')
        visualization.savefig(write_dir, 'lfp_aligned_laser_dch_trigger.png')
        
        # Also plot the individual trials
        plt.figure()
        im = visualization.plot_image_by_time(t, 1e6*lfp_metadata['voltsperbit']*erp[:,ch,:], ylabel='trials')
        im.set_clim(-100,100)
        visualization.savefig(docs_dir, 'laser_aligned_lfp_dch_trigger.png')
        
        # And compare to the sensor data
        sensor_data = exp_data['qwalor_sensor']
        sensor_voltsperbit = exp_metadata['analog_voltsperbit']
        samplerate = exp_metadata['analog_samplerate']

        plt.figure()
        ds_data = precondition.downsample(sensor_data, samplerate, 1000)
        ds_data = ds_data - np.mean(ds_data)
        analog_erp = analysis.calc_erp(ds_data, trial_times, time_before, time_after, 1000)
        im = visualization.plot_image_by_time(t, sensor_voltsperbit*analog_erp[:,0,:], ylabel='trials')
        im.set_clim(-0.01,0.01)
        visualization.savefig(docs_dir, 'laser_aligned_sensor_dch_trigger.png')

        # One more file, with no lfp data but it has multiple channels of stimulation using MultiQwalorLaser feature.
        subject = 'test'
        te_id = 8940
        date = '2023-03-27'
            
        # Preprocess the raw data
        # files = {}
        # files['hdf'] = '/Users/leoscholl/raw/hdf/test20230327_11_te8940.hdf'
        # files['ecube'] = '/Users/leoscholl/raw/ecube/2023-03-27_BMI3D_te8940'
        # # os.remove(os.path.join(data_dir, 'test/preproc_2023-03-27_test_8940_exp.hdf'))
        # proc_exp('', files, data_dir, 'test/preproc_2023-03-27_test_8940_exp.hdf', overwrite=True)
        exp_data, exp_metadata = load_preproc_exp_data(preproc_dir, subject, te_id, date)

        self.assertEqual(exp_metadata['sync_protocol_version'], 13)
        self.assertIn('qwalor_ch2_trigger_dch', exp_metadata.keys())
        self.assertIn('qwalor_ch2_trigger', exp_data.keys())

        trial_times, trial_widths, trial_gains, trial_powers = get_laser_trial_times(
            preproc_dir, subject, te_id, date, laser_trigger='qwalor_ch2_trigger', 
            laser_sensor='qwalor_ch2_sensor', debug=True
        )
        visualization.savefig(write_dir, 'laser_aligned_sensor_debug_dch_trigger.png')

    def test_get_target_events(self):

        subject = 'test'
        te_id = 8940
        date = '2023-03-27'

        exp_data, exp_metadata = load_preproc_exp_data(data_dir, subject, te_id, date)
        target = get_target_events(exp_data, exp_metadata)
        
        plt.figure()
        time = exp_data['events']['timestamp']
        plt.plot(time, target[:,:,0]) # plot just the x coordinate
        plt.xlim(10, 20)
        plt.xlabel('time (s)')
        plt.ylabel('x position (cm)')
        filename = 'get_target_events.png'
        visualization.savefig(docs_dir, filename)

    def test_get_ref_dis_frequencies(self):
        subject = 'test'
        te_id = '8461'
        date = '2023-02-25'

        data, metadata = load_preproc_exp_data(data_dir, subject, te_id, date)
        freq_r, freq_d = get_ref_dis_frequencies(data, metadata)

        plt.figure()
        plt.plot(freq_r, 'darkorange')
        plt.plot(freq_d, 'tab:red', linestyle='--')
        plt.xlabel('Trial #'); plt.ylabel('Frequency (Hz)')
        filename = 'get_ref_dis_freqs_test.png'
        visualization.savefig(docs_dir, filename)

        subject = 'churro'
        te_id = '375'
        date = '2023-10-02'

        data, metadata = load_preproc_exp_data(data_dir, subject, te_id, date)
        freq_r, freq_d = get_ref_dis_frequencies(data, metadata)

        plt.figure()
        plt.plot(freq_r, 'darkorange')
        plt.plot(freq_d, 'tab:red', linestyle='--')
        plt.xlabel('Trial #'); plt.ylabel('Frequency (Hz)')
        filename = 'get_ref_dis_freqs_churro.png'
        visualization.savefig(docs_dir, filename)

        subject = 'beignet'
        te_id = '8380'
        date = '2023-02-13'

        data, metadata = load_preproc_exp_data(data_dir, subject, te_id, date)
        freq_r, freq_d = get_ref_dis_frequencies(data, metadata)

        plt.figure()
        plt.plot(freq_r, 'darkorange')
        plt.plot(freq_d, 'tab:red', linestyle='--')
        plt.xlabel('Trial #'); plt.ylabel('Frequency (Hz)')
        filename = 'get_ref_dis_freqs_before_gen_idx_saved.png'
        visualization.savefig(docs_dir, filename) 

class ProcTests(unittest.TestCase):

    def test_proc_single(self):
        files = {}
        files['ecube'] = 'fake ecube data'
        files['hdf'] = 'fake_ecube_data_bmi3d.hdf'
        proc_single(data_dir, files, write_dir, 'test', 3498, '2021-12-13', ['exp', 'eye', 'broadband', 'lfp'], overwrite=True)

    def test_proc_broadband(self):
        files = {'ecube': "short headstage test"}
        result_filename = 'short_headstage_test_broadband.hdf'
        result_filepath = os.path.join(write_dir, result_filename)
        if os.path.exists(result_filepath):
            os.remove(result_filepath)
        proc_broadband(data_dir, files, write_dir, result_filename, overwrite=False)
        self.assertTrue(os.path.exists(result_filepath))
        contents = get_hdf_dictionary(write_dir, result_filename)
        self.assertIn('broadband_data', contents)
        self.assertIn('broadband_metadata', contents)

        # Don't overwrite
        test_fun = lambda: proc_broadband(data_dir, files, write_dir, result_filename, overwrite=False)
        self.assertRaises(FileExistsError, test_fun)

        # Overwrite
        proc_broadband(data_dir, files, write_dir, result_filename, overwrite=True)

    def test_proc_spikes(self):
        np_recorddir = '2024-08-27_Neuropixel_test_te0001'
        ecube_files = '2024-08-27_BMI3D_te0001'
        files = {'ecube': ecube_files, 'neuropixels':np_recorddir}
        kilosort_dir = Path(data_dir)/'kilosort'
        save_dir = Path(data_dir)/'test'
        result_filename = aodata.get_preprocessed_filename('test', '0001', '2024-08-27', 'spike')

        proc_spikes(data_dir, files, save_dir, result_filename, kilosort_dir=kilosort_dir, overwrite=True)

        spikes1, metadata1 = load_preproc_spike_data(data_dir, 'test', '0001', '2024-08-27', drive_number=1)
        waveforms1  = load_spike_waveforms(data_dir, 'test', '0001', '2024-08-27', drive_number=1)
        self.assertIn('0', spikes1)
        self.assertIn('0', waveforms1)
        self.assertIn('1', spikes1)
        self.assertIn('1', waveforms1)
        self.assertIn('sync_timestamps', metadata1)
        self.assertIn('spike_pos', metadata1)

        spikes2, metadata2 = load_preproc_spike_data(data_dir, 'test', '0001', '2024-08-27', drive_number=2)
        waveforms2  = load_spike_waveforms(data_dir, 'test', '0001', '2024-08-27', drive_number=2)
        self.assertIn('0', spikes2)
        self.assertIn('0', waveforms2)
        self.assertIn('1', spikes2)
        self.assertIn('1', waveforms2)
        self.assertIn('sync_timestamps', metadata2)
        self.assertIn('spike_pos', metadata2)

        # Delete the file because its size is big
        result_hdffile = save_dir / result_filename
        result_hdffile.unlink()

    def test_proc_ap(self):
        # Test proc from ap data in neuropixels
        np_recorddir = '2024-08-27_Neuropixel_test_te0001'
        ecube_files = '2024-08-27_BMI3D_te0001'
        files = {'ecube': ecube_files, 'neuropixels':np_recorddir}
        kilosort_dir = Path(data_dir)/'kilosort'
        save_dir = Path(data_dir)/'test'

        datatype = 'ap'
        result_filename = aodata.get_preprocessed_filename('test', '0001', '2024-08-27', datatype)

        proc_ap(data_dir,files,save_dir,result_filename,kilosort_dir=kilosort_dir,overwrite=True,max_memory_gb=1.)
        ap_data1, ap_metadata1 = load_preproc_ap_data(data_dir, 'test', '0001', '2024-08-27', drive_number=1)
        ap_data2, ap_metadata2 = load_preproc_ap_data(data_dir, 'test', '0001', '2024-08-27', drive_number=2)

        self.assertEqual(ap_data1.shape[1], ap_metadata1['n_channels'])
        self.assertIn('ap_samplerate', ap_metadata1)
        self.assertIn('sync_timestamps', ap_metadata1)
        self.assertEqual(ap_data2.shape[1], ap_metadata2['n_channels'])
        self.assertIn('ap_samplerate', ap_metadata2)
        self.assertIn('sync_timestamps', ap_metadata2)

        # Delete the file because its size is big
        result_hdffile = save_dir / result_filename
        result_hdffile.unlink()

    def test_proc_lfp(self):

        # Test proc from ecube
        ecube_result_filename = 'test_proc_lfp.hdf'
        files = {'ecube': 'fake ecube data'}
        ecube_metadata = load_ecube_metadata(os.path.join(data_dir, files['ecube']), 'Headstages')

        proc_lfp(data_dir, files, write_dir, ecube_result_filename, max_memory_gb=0.0001, overwrite=True)

        contents = get_hdf_dictionary(write_dir, ecube_result_filename)
        self.assertIn('lfp_data', contents)
        self.assertIn('lfp_metadata', contents)

        lfp_data = load_hdf_data(write_dir, ecube_result_filename, 'lfp_data')
        lfp_metadata = load_hdf_group(write_dir, ecube_result_filename, 'lfp_metadata')

        approx_n_samples = int(ecube_metadata['n_samples']*lfp_metadata['samplerate']/ecube_metadata['samplerate'])

        self.assertTrue(abs(lfp_data.shape[0] - approx_n_samples) < 100) # within 100 samples
        self.assertEqual(lfp_data.shape[1], 8)
        self.assertEqual(lfp_metadata['lfp_samplerate'], 1000)
        self.assertEqual(lfp_metadata['samplerate'], 1000)

        # Test proc from broadband hdf
        bb_filename = 'test_proc_lfp_bb.hdf'
        bb_result_filename = 'test_proc_lfp_from_bb.hdf'
        files['broadband'] = bb_filename
        proc_broadband(data_dir, files, write_dir, bb_filename, overwrite=True)
        proc_lfp(data_dir, files, write_dir, bb_result_filename, max_memory_gb=0.0001, overwrite=True)

        contents = get_hdf_dictionary(write_dir, bb_result_filename)
        self.assertIn('lfp_data', contents)
        self.assertIn('lfp_metadata', contents)

        lfp_data = load_hdf_data(write_dir, bb_result_filename, 'lfp_data')
        lfp_metadata = load_hdf_group(write_dir, bb_result_filename, 'lfp_metadata')

        self.assertTrue(abs(lfp_data.shape[0] - approx_n_samples) < 100) # within 100 samples
        self.assertEqual(lfp_data.shape[1], 8)
        self.assertEqual(lfp_metadata['lfp_samplerate'], 1000)
        self.assertEqual(lfp_metadata['samplerate'], 1000)

        # Test proc from lfp data in neuropixels
        np_recorddir = '2024-08-27_Neuropixel_test_te0001'
        ecube_files = '2024-08-27_BMI3D_te0001'
        files = {'ecube': ecube_files, 'neuropixels':np_recorddir}
        kilosort_dir = Path(data_dir)/'kilosort'
        save_dir = Path(data_dir)/'test'

        datatype = 'lfp'
        result_filename = aodata.get_preprocessed_filename('test', '0001', '2024-08-27', datatype)

        proc_lfp(data_dir,files,save_dir,result_filename,kilosort_dir=kilosort_dir,overwrite=True,max_memory_gb=1.)
        lfp_data1, lfp_metadata1 = load_preproc_lfp_data(data_dir, 'test', '0001', '2024-08-27', drive_number=1)
        lfp_data2, lfp_metadata2 = load_preproc_lfp_data(data_dir, 'test', '0001', '2024-08-27', drive_number=2)

        self.assertEqual(lfp_data1.shape[1], lfp_metadata1['n_channels'])
        self.assertIn('lfp_samplerate', lfp_metadata1)
        self.assertIn('sync_timestamps', lfp_metadata1)
        self.assertEqual(lfp_data2.shape[1], lfp_metadata2['n_channels'])
        self.assertIn('lfp_samplerate', lfp_metadata2)
        self.assertIn('sync_timestamps', lfp_metadata2)

        # Delete the file because its size is big
        result_hdffile = save_dir / result_filename
        result_hdffile.unlink()

        # Compare the two files
        ecube_lfp_data = load_hdf_data(write_dir, ecube_result_filename, 'lfp_data')
        bb_lfp_data = load_hdf_data(write_dir, bb_result_filename, 'lfp_data')
        lfp_metadata = load_hdf_group(write_dir, bb_result_filename, 'lfp_metadata')

        ch = 0

        fig, ax = plt.subplots(3,1, figsize=(5,8))
        visualization.plot_timeseries(1e6*lfp_metadata['voltsperbit']*ecube_lfp_data[:100,ch], 1000, ax=ax[0])
        visualization.plot_timeseries(1e6*lfp_metadata['voltsperbit']*bb_lfp_data[:100,ch], 1000, ax=ax[0])
        ax[0].set_title('first 100ms')
        ax[0].set_ylabel('Voltage (uV)')
        ax[0].set_ylim(-2000,2000)
        ax[0].legend(['from ecube', 'from hdf'])

        visualization.plot_timeseries(1e6*lfp_metadata['voltsperbit']*ecube_lfp_data[-100:,ch], 1000, ax=ax[1])
        visualization.plot_timeseries(1e6*lfp_metadata['voltsperbit']*bb_lfp_data[-100:,ch], 1000, ax=ax[1])
        ax[1].set_title('last 100ms')
        ax[1].set_ylabel('Voltage (uV)')
        ax[1].set_ylim(-2000,2000)

        chunksize = 250
        visualization.plot_timeseries(1e6*lfp_metadata['voltsperbit']*ecube_lfp_data[chunksize-50:chunksize+50,ch], 1000, ax=ax[2])
        visualization.plot_timeseries(1e6*lfp_metadata['voltsperbit']*bb_lfp_data[chunksize-50:chunksize+50,ch], 1000, ax=ax[2])
        ax[2].set_title('chunk boundary')
        ax[2].set_ylabel('Voltage (uV)')
        ax[2].set_ylim(-2000,2000)

        plt.tight_layout()
        visualization.savefig(docs_dir, 'proc_lfp_comparison.png')

    def test_proc_emg(self):

        emg_data_dir = os.path.join(data_dir, 'quatt_emg')
        files = {'emg':'aj20250319_05_te2540_emg.hdf',
                 'exp':'aj20250319_05_te2540_exp.hdf'
        }

        result_filename = 'test_proc_emg_out.hdf'

        proc_emg(emg_data_dir, files, write_dir, result_filename, overwrite=True)


class QualityTests(unittest.TestCase):

    def setUp(self):
        # Load some test data
        test_filepath = os.path.join(data_dir, "short headstage test")
        self.data = load_ecube_data(test_filepath, 'Headstages', channels=range(1,9))
        self.samplerate = 25000
        self.num_th = 3.
        self.lf_c = 100.
        self.win_t = 0.1
        self.over_t = 0.05
        self.bandwidth = 10 # short sequences
        print(f"Testing signal quality with {self.data.shape[0]/self.samplerate:.1f} seconds of {self.data.shape[1]} channel data.")

    def test_bad_channel_detection(self):
        bad_ch = quality.bad_channel_detection(
            data = self.data, 
            srate = self.samplerate,
            num_th = self.num_th,
            lf_c = self.lf_c,
            sg_win_t = self.win_t,
            sg_over_t = self.over_t,
            sg_bw = self.bandwidth
        )
        self.assertEqual(bad_ch.shape, (8,))
        # self.assertEqual(np.count_nonzero(bad_ch), 64)
        
    def test_detect_bad_ch_outliers(self):
        np.random.seed(0)
        test_data = np.random.normal(10,0.5,(10000, 200))
        test_data[0, 10] = 25
        test_data[5, 150] = 30
        bad_ch = quality.detect_bad_ch_outliers(test_data, nbins=10000, thr=0.05, numsd=5.0, debug=True, verbose=False)
        self.assertEqual(np.where(bad_ch)[0][0], 10)
        self.assertEqual(np.where(bad_ch)[0][1], 150)
        
        filename = 'detect_bad_ch_outliers.png'
        visualization.savefig(docs_dir, filename)

    def test_detect_bad_trials(self):
        nt = 50
        nch = 10
        ntr = 100
        np.random.seed(0)
        erp = np.random.normal(size=(nt, nch, ntr)) 
        erp[:,:,0] += 10 # entire trial is noisy across all electrodes
        erp[:,:8,1] -= 10 # entire trial is noisy on most electrodes
        erp[0,:,2] += 10 # single timepoint within the trial is noisy on all electrodes
        for t in range(nt):
            erp[t,t%nch,3] -= 10 # single timepoint is noisy but different timepoint for each channel
                
        bad_trials = detect_bad_trials(erp, sd_thr=5, ch_frac=0.5, debug=True)
            
        self.assertEqual(len(bad_trials), ntr)  
        self.assertTrue(bad_trials[0])
        self.assertTrue(bad_trials[1])
        self.assertTrue(bad_trials[2])
        self.assertTrue(bad_trials[3])
        self.assertTrue(np.all(~bad_trials[4:]))

        filename = 'detect_bad_trials.png'
        visualization.savefig(docs_dir, filename, transparent=False)

    def test_detect_bad_timepoints(self):
        nt = 500
        nch = 10
        np.random.seed(0)
        data = np.random.normal(size=(nt, nch)) 
        data[50:52,:] += 10 # timepoint is noisy across all electrodes
        data[100:102,2:] -= 10 # timepoint is noisy on most electrodes
        for t in range(nt-100,nt):
            data[t,t%nch] -= 10 # single timepoint is noisy but different timepoint for each channel
                
        bad_timepoints = quality.detect_bad_timepoints(data, sd_thr=5, ch_frac=0.5, debug=True)
            
        filename = 'detect_bad_timepoints.png'
        visualization.savefig(docs_dir, filename, transparent=False)
        self.assertEqual(len(bad_timepoints), nt)
        self.assertTrue(np.all(bad_timepoints[50:52]))
        self.assertTrue(np.all(bad_timepoints[100:102]))
        self.assertFalse(np.any(bad_timepoints[-100:]))


    def test_high_freq_data_detection(self):
        bad_data_mask, bad_data_mask_all_ch = quality.high_freq_data_detection(
            self.data, 
            self.samplerate,
            lf_c = self.lf_c,
            sg_win_t = self.win_t,
            sg_over_t = self.over_t,
            sg_bw = self.bandwidth
        )
        self.assertEqual(bad_data_mask.shape, (self.data.shape[0],))
        # self.assertEqual(np.count_nonzero(bad_data_mask), 64)

    def test_saturated_data_detection(self):
        sat_data_mask, sat_data_mask_all_ch = quality.saturated_data_detection(self.data, self.samplerate)
        self.assertEqual(sat_data_mask.shape, (self.data.shape[0],))
        self.assertEqual(np.count_nonzero(sat_data_mask), 0)

    def test_detect_bad_exp_data(self):
        # first entry doesn't exist, second has buggy sync events, third is normal
        subjects = ['beignet', 'beignet', 'beignet']
        ids = [100, 8694, 8695]
        dates = [datetime.date(2023, 3, 10), datetime.date(2023, 3, 10), datetime.date(2023, 3, 10)]
        bad_entries = detect_bad_exp_data(data_dir, subjects, ids, dates)
        np.testing.assert_array_equal(bad_entries, [ [subjects[0],dates[0],ids[0]], [subjects[1],dates[1],ids[1]] ])
        
class OculomaticTests(unittest.TestCase):
    
    def test_parse_oculomatic(self):
        files = {}
        files['ecube'] = '2021-09-30_BMI3D_te2952'
        files['hdf'] = 'beig20210930_02_te2952.hdf'
        data, metadata = parse_oculomatic(data_dir, files)

        self.assertIn('data', data)
        self.assertIn('mask', data)
        self.assertIn('samplerate', metadata)
        self.assertIn('channels', metadata)
        self.assertIn('labels', metadata)

        eye_channels = metadata['channels']
        data_path = metadata['source']
        analog_metadata = aodata.load_ecube_metadata(data_path, 'AnalogPanel')
        analog_data = analog_metadata['voltsperbit']*aodata.load_ecube_data(data_path, 'AnalogPanel', channels=eye_channels)

        old_samplerate = metadata['raw_samplerate']
        new_samplerate = metadata['samplerate']
        downsample_data = data['data']

        t_25k = np.arange(len(analog_data))/old_samplerate
        t_1k = np.arange(len(downsample_data))/new_samplerate

        self.assertEqual(old_samplerate, 25000)
        self.assertEqual(new_samplerate, 1000)

        t_range = [6,8]

        fig,ax = plt.subplots(2,1)
        ax[0].plot(t_25k[int(t_range[0]*old_samplerate):int(t_range[1]*old_samplerate)], analog_data[int(t_range[0]*old_samplerate):int(t_range[1]*old_samplerate),0])
        ax[1].plot(t_1k[int(t_range[0]*new_samplerate):int(t_range[1]*new_samplerate)], downsample_data[int(t_range[0]*new_samplerate):int(t_range[1]*new_samplerate),0])
        ax[0].set_ylabel('25khz')
        ax[1].set_ylabel('100hz')
        plt.tight_layout()
        filename =  'proc_oculomatic_downsample.png'
        visualization.savefig(docs_dir, filename)

        fig,ax = plt.subplots(2,1)
        visualization.plot_freq_domain_amplitude(analog_data, old_samplerate, ax=ax[0])
        visualization.plot_freq_domain_amplitude(downsample_data, new_samplerate, ax=ax[1])
        ax[0].set_ylabel('25khz')
        ax[1].set_ylabel('100hz')
        ax[0].set_ylim(0,1)
        ax[1].set_ylim(0,1)
        ax[0].set_xlim(0,100)
        ax[1].set_xlim(0,100)
        plt.tight_layout()
        filename =  'proc_oculomatic_freq.png'
        visualization.savefig(docs_dir, filename)


    def test_detect_noise(self):
        test_data = np.array([1,2,3,4,5,7,7,7,7,5,4,3,2,1])
        test_data = np.expand_dims(test_data, axis=1)
        samplerate = 10
        eye_closed_mask = oculomatic.detect_noise(test_data, samplerate, min_step=0, step_thr=1)
        
        expected_mask = np.concatenate((np.zeros(5), np.ones(4), np.zeros(5)))
        expected_mask = np.expand_dims(expected_mask, axis=1)
        np.testing.assert_allclose(eye_closed_mask, expected_mask)

        data_path = os.path.join(data_dir, '2021-09-29_BMI3D_te2949')
        eye_channels = [10, 11, 8, 9]
        analog_metadata = aodata.load_ecube_metadata(data_path, 'AnalogPanel')
        analog_data = load_ecube_data(data_path, 'AnalogPanel', channels=eye_channels)

        eye_closed_mask = oculomatic.detect_noise(analog_data, analog_metadata['samplerate'], min_step=4, step_thr=2)
        plt.figure()
        plt.matshow(eye_closed_mask, aspect='auto')
        filename =  'proc_oculomatic_mask.png'
        visualization.savefig(docs_dir, filename)

class NeuropixelTests(unittest.TestCase):
    
    def test_sync_neuropixel_ecube(self):
        record_dir = '2023-03-26_Neuropixel_beignet_te8921'
        ecube_files = '2023-03-26_BMI3D_te8921'

        # For AP data
        data, metadata = load_neuropixel_data(data_dir, record_dir, 'ap', port_number=1)
        raw_timestamps = np.arange(data.sample_numbers.shape[0])/metadata['sample_rate']

        # get on_times and off_times when sync line come
        on_times_np,off_times_np = get_neuropixel_digital_input_times(data_dir, record_dir, 'ap', port_number=1)
        on_times_ecube,off_times_ecube = get_ecube_digital_input_times(data_dir, ecube_files, ch=-1)

        # get barcode on_times 
        barcode_ontimes_np, _ = extract_barcodes_from_times(on_times_np,off_times_np)
        barcode_ontimes_ecube, _ = extract_barcodes_from_times(on_times_ecube,off_times_ecube)

        # if sychronization works, the time when the first barcode turns on in openephys is converted to the time when the first barcode turns on in ecube
        sync_timestamps, _ = sync_neuropixel_ecube(raw_timestamps, on_times_np, off_times_np, on_times_ecube, off_times_ecube, bar_duration=0.017)
        raw_first_barcode_on_idx = np.where(raw_timestamps == barcode_ontimes_np[0])[0]
        raw_last_barcode_on_idx = np.where(raw_timestamps == barcode_ontimes_np[-1])[0]
        self.assertEqual(sync_timestamps[raw_first_barcode_on_idx], barcode_ontimes_ecube[0])
        self.assertEqual(sync_timestamps[raw_last_barcode_on_idx], barcode_ontimes_ecube[-1])    

        # For LFP data
        data, metadata = load_neuropixel_data(data_dir, record_dir, 'lfp', port_number=1)
        raw_timestamps = np.arange(data.sample_numbers.shape[0])/metadata['sample_rate']

        # get on_times and off_times when sync line come
        on_times_np,off_times_np = get_neuropixel_digital_input_times(data_dir, record_dir, 'lfp', port_number=1)
        on_times_ecube,off_times_ecube = get_ecube_digital_input_times(data_dir, ecube_files, ch=-1)

        # get barcode on_times 
        barcode_ontimes_np, _ = extract_barcodes_from_times(on_times_np,off_times_np)
        barcode_ontimes_ecube, _ = extract_barcodes_from_times(on_times_ecube,off_times_ecube)

        # if sychronization works, the time when the first barcode turns on in openephys is converted to the time when the first barcode turns on in ecube
        sync_timestamps, _ = sync_neuropixel_ecube(raw_timestamps, on_times_np, off_times_np, on_times_ecube, off_times_ecube, bar_duration=0.017)
        raw_first_barcode_on_idx = np.where(raw_timestamps == barcode_ontimes_np[0])[0]
        raw_last_barcode_on_idx = np.where(raw_timestamps == barcode_ontimes_np[-1])[0]
        self.assertEqual(sync_timestamps[raw_first_barcode_on_idx], barcode_ontimes_ecube[0])
        self.assertEqual(sync_timestamps[raw_last_barcode_on_idx], barcode_ontimes_ecube[-1])
        
    def test_concat_neuropixel_within_day(self):
        # test for concat_neuropixel_within_day
        date = '2023-06-27'
        subject = 'test'
        np_recorddir1 = f'{date}_Neuropixel_{subject}_te0001'
        np_recorddir2 = f'{date}_Neuropixel_{subject}_te0002'
        kilosort_dir = os.path.join(data_dir, 'kilosort')
        concat_dataname = f'{date}_Neuropixel_ks_{subject}_bottom_port1'
        ch_config_dir = os.path.join(data_dir, 'channel_config_np')
        
        # concatenate data
        _ = concat_neuropixel_within_day(data_dir, kilosort_dir, subject, date, ch_config_dir=ch_config_dir, port_number=1)
        
        # load each data
        data1, _ = load_neuropixel_data(data_dir, np_recorddir1,'ap',port_number=1)
        sample_size1 = data1.samples.shape[0]
        data2, _ = load_neuropixel_data(data_dir, np_recorddir2,'ap',port_number=1)

        # Check if the second part in con data is equal to the data2
        kilosort_dir = os.path.join(data_dir, 'kilosort')
        concat_data_dir = os.path.join(kilosort_dir, concat_dataname)
        con_data = np.memmap(os.path.join(concat_data_dir,'continuous.dat'), dtype='int16') # load continuous.dat file
        con_data = con_data.reshape(-1,384)
        self.assertTrue(np.all(con_data[:10,:] == data1.samples[:10,:]))
        self.assertTrue(np.all(con_data[sample_size1:sample_size1+10,:] == data2.samples[:10,:]))
        
        # Check removing bad task id
        #_ = concat_neuropixel_within_day(data_dir, kilosort_dir, subject, date, ch_config_dir=ch_config_dir, port_number=1, bad_taskid=['0001'])

        concat_dataname = f'{date}_Neuropixel_ks_{subject}2_bottom_port1'
        concat_data_dir = os.path.join(kilosort_dir, concat_dataname)
        con_data = np.memmap(os.path.join(concat_data_dir,'continuous.dat'), dtype='int16') # load continuous.dat file
        con_data = con_data.reshape(-1,384) 
        self.assertTrue(np.all(con_data[:10,:] == data2.samples[:10,:]))
            
    def test_parse_and_load_ksdata(self):
        date = '2023-03-26'
        subject = 'beignet'
        task_id ='8922'
        kilosort_dir = os.path.join(data_dir, 'kilosort')
        concat_data_dir = f'{date}_Neuropixel_ks_{subject}_bottom_port1'
        parsed_data_dir = f'{date}_Neuropixel_ks_{subject}_bottom_port1_{task_id}'
        
        # devide spike times and clusters into each entry data
        parse_ksdata_entries(kilosort_dir, concat_data_dir)

        # load parsed data
        spike_indices, spike_label, ks_label = load_parsed_ksdata(kilosort_dir, parsed_data_dir)
        self.assertTrue(np.all(spike_indices>0))
        self.assertTrue(spike_indices.shape == spike_label.shape)
        self.assertTrue((np.all((ks_label[:,1] == 'mua') | (ks_label[:,1] == 'good'))))
        
        spike_indices_unit = classify_ks_unit(spike_indices, spike_label)
        self.assertTrue(np.all(spike_indices_unit['0']>0))
        self.assertTrue(np.all(spike_indices_unit['1']>0))
    
    def test_sync_ts_data_timestamps(self):
        ndata = 20
        
        # Test for cropping
        data = np.random.rand(ndata)
        sync_timestamp = np.linspace(-0.1,0.5,ndata)
        sync_data, _ = sync_ts_data_timestamps(data, sync_timestamp)
        self.assertTrue(sync_data.shape[0], sum(sync_timestamp>0))

        # Test for padding
        data = np.random.rand(ndata)
        sync_timestamp = np.linspace(0.3,0.5,ndata)
        sync_data, _ = sync_ts_data_timestamps(data, sync_timestamp)
        
        dt = sync_timestamp[1] - sync_timestamp[0]
        num_padding = np.arange(sync_timestamp[0]-dt,0,-dt).shape[0]
        self.assertTrue(sync_data.shape[0], num_padding + sync_timestamp.shape[0])  
    
    def test_destripe_lfp_batch(self):
        save_path = Path(data_dir) / 'test_a'

        sample_rate = 1000
        bit_volts = 0.1
        data = np.random.randint(0,1000, (1000,384))

        destripe_lfp_batch(data, save_path, sample_rate, bit_volts, max_memory_gb = .05, dtype='int16')
        destriped_data = np.memmap(save_path, shape=data.shape, dtype='int16')

        np.testing.assert_(np.all(data.shape == destriped_data.shape))
        del destriped_data # delete memmap file to completely delete the file later
        save_path.unlink()
        
    def test_preproc_spikes(self):
        np_recorddir = '2024-08-27_Neuropixel_test_te0001'
        ecube_files = '2024-08-27_BMI3D_te0001'
        kilosort_dir = Path(data_dir)/'kilosort'
        filename = aodata.get_preprocessed_filename('test', '0001', '2024-08-27', 'spike')
        save_dir = Path(data_dir)/'test'
        result_filename = save_dir / filename

        # Preprocess data from port 1 and port 2
        port_number = 1
        for port_number in [1,2]:
            _, _, metadata = proc_neuropixel_spikes(data_dir,np_recorddir,ecube_files,kilosort_dir,port_number,result_filename)
            aodata.save_hdf(save_dir, filename, metadata, f'drive{port_number}/metadata', append=True)
            spikes, metadata = load_preproc_spike_data(data_dir, 'test', '0001', '2024-08-27', drive_number=1)
            waveforms  = load_spike_waveforms(data_dir, 'test', '0001', '2024-08-27', drive_number=1)
            self.assertIn('0', spikes)
            self.assertIn('0', waveforms)
            self.assertIn('1', spikes)
            self.assertIn('1', waveforms)
            self.assertIn('sync_timestamps', metadata)
            self.assertIn('spike_pos', metadata)

        result_filename.unlink() # delete preprocessed file
        
    def test_preproc_ts(self):
        np_recorddir = '2024-08-27_Neuropixel_test_te0001'
        ecube_files = '2024-08-27_BMI3D_te0001'
        kilosort_dir = Path(data_dir)/'kilosort'
        save_dir = Path(data_dir)/'test'
        
        # Test for ap data
        datatype = 'ap'
        filename = aodata.get_preprocessed_filename('test', '0001', '2024-08-27', 'ap')
        result_filename = save_dir / filename
        for port_number in [1,2]:
            _, metadata = proc_neuropixel_ts(data_dir,np_recorddir,ecube_files,kilosort_dir,datatype, port_number, result_filename)
            aodata.save_hdf(save_dir, filename, metadata, f'drive{port_number}/ap_metadata', append=True)
            ap_data, metadata = load_preproc_ap_data(data_dir, 'test', '0001', '2024-08-27', drive_number=port_number)
            self.assertEqual(ap_data.shape[1], metadata['n_channels'])
            self.assertIn('ap_samplerate', metadata)
            self.assertIn('sync_timestamps', metadata)
            result_filename.unlink() # delete preprocessed file

        # Test for lfp data
        datatype = 'lfp'
        filename = aodata.get_preprocessed_filename('test', '0001', '2024-08-27', 'lfp')
        result_filename = save_dir / filename
        for port_number in [1,2]:
            _, metadata = proc_neuropixel_ts(data_dir,np_recorddir,ecube_files,kilosort_dir,datatype, port_number, result_filename)
            aodata.save_hdf(save_dir, filename, metadata, f'drive{port_number}/lfp_metadata', append=True)
            lfp_data, metadata = load_preproc_lfp_data(data_dir, 'test', '0001', '2024-08-27', drive_number=port_number)
            self.assertEqual(lfp_data.shape[1], metadata['n_channels'])
            self.assertIn('lfp_samplerate', metadata)
            self.assertIn('sync_timestamps', metadata)
            result_filename.unlink() # delete preprocessed file
        
        
class LaserTests(unittest.TestCase):

    def test_calibrate_gain(self):
        gain = np.arange(0.4,1.0,0.01)
        powers = preproc.laser.calibrate_gain(gain, 12.)
        plt.figure()
        plt.plot(gain, powers)
        plt.xlabel('gain')
        plt.ylabel('power (mW)')

        visualization.savefig(docs_dir, 'calibrate_laser_gain.png')
        
    def test_calibrate_sensor(self):
        voltage = [0.12, 0.08, 0.14]
        powers = preproc.laser.calibrate_sensor(voltage, 12.)
        np.testing.assert_allclose(powers, [8.57,  5.07, 10.32], atol=1e-2)

if __name__ == "__main__":
    unittest.main()
