import unittest
from aopy.visualization import *
import aopy
import numpy as np
import os

test_dir = os.path.dirname(__file__)
data_dir = os.path.join(test_dir, 'data')
write_dir = os.path.join(test_dir, 'tmp')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

class NeuralDataPlottingTests(unittest.TestCase):

    def test_plot_timeseries(self):
        filename = 'timeseries.png'
        data = np.reshape(np.sin(np.pi*np.arange(1000)/10) + np.sin(2*np.pi*np.arange(1000)/10), (1000))
        samplerate = 1000
        plt.figure()
        plot_timeseries(data, samplerate)
        savefig(write_dir, filename)

    def test_plot_freq_domain_amplitude(self):
        filename = 'freqdomain.png'
        data = np.reshape(np.sin(np.pi*np.arange(1000)/10) + np.sin(2*np.pi*np.arange(1000)/10), (1000))
        samplerate = 1000
        plt.figure()
        plot_freq_domain_amplitude(data, samplerate) # Expect 100 and 50 Hz peaks at 1 V each
        savefig(write_dir, filename)

    def test_spatial_map(self):
        data = np.linspace(-1, 1, 100)
        x_pos, y_pos = np.meshgrid(np.arange(0.5,10.5),np.arange(0.5, 10.5))
        missing = [0, 5, 25]
        data_missing = np.delete(data, missing)
        x_missing = np.reshape(np.delete(x_pos, missing),-1)
        y_missing = np.reshape(np.delete(y_pos, missing),-1)

        filename = 'posmap.png'
        data_map = get_data_map(data_missing, x_missing, y_missing)
        self.assertEqual(data_map.shape, (10, 10))
        self.assertTrue(np.isnan(data_map[0,0]))
        plt.figure()
        plot_spatial_map(data_map, x_missing, y_missing)
        savefig(write_dir, filename)

        filename = 'posmap_interp.png'
        interp_map = calc_data_map(data_missing, x_missing, y_missing, [10, 10], threshold_dist=0.01)
        self.assertEqual(interp_map.shape, (10, 10))
        self.assertTrue(np.isnan(interp_map[0,0]))
        plt.figure()
        plot_spatial_map(interp_map, x_missing, y_missing)
        savefig(write_dir, filename)

    def test_single_spatial_map(self):
        data = 2.0
        x_pos, y_pos = np.meshgrid(1,1)

        data_map = get_data_map(data, x_pos, y_pos)
        self.assertEqual(data_map[0], 2.0)
        plt.figure()
        plot_spatial_map(data_map, x_pos, y_pos)
        plt.show()

    def test_plot_raster(self):
        filename = 'raster_plot_example.png'
        np.random.seed(0)
        data = np.random.random([50, 6])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_raster(data, cue_bin=0.2, ax=ax)
        savefig(write_dir, filename)

    def test_plot_waveforms(self):
        # load example waveform data
        data_filename = 'example_wfs.npy'
        filepath = os.path.join(data_dir, data_filename)
        with open(filepath, 'rb') as f:
            wfs = np.load(f)
        
        filename = 'waveform_plot_example.png'
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_title('Mean on')
        plot_waveforms(wfs, 40000)

        ax = fig.add_subplot(122)
        plot_waveforms(wfs, 40000, plot_mean=False)
        ax.set_title('Mean off')
        fig.tight_layout()
        savefig(write_dir, filename)
    
class CurveFittingTests(unittest.TestCase):
    def test_plot_tuning_curves(self):
        filename = 'tuning_curves_plot.png'
        nunits = 7
        targets = np.arange(0, 360, 45)
        mds_true = np.linspace(1, 3, nunits)/2
        pds_offset = np.arange(-45,270,45)
        data = np.zeros((nunits,8))*np.nan
        for ii in range(nunits):
            noise = np.random.normal(1, 0.2, size=(1,8))
            data[ii,:] = noise*mds_true[ii]*np.sin(np.deg2rad(targets)-np.deg2rad(pds_offset[ii])) + 2

        # If the mds and pds output are correct the fitting params are correct because they are required for the calculation.
        fit_params, _, _ = aopy.analysis.run_tuningcurve_fit(data, targets)
        plot_tuning_curves(fit_params, data, targets, n_subplot_cols=4)
        savefig(write_dir, filename)

class AnimationTests(unittest.TestCase):

    def test_animate_events(self):
        events = ["hello", "world", "", "!", ""]
        times = [0., 1.0, 1.5, 2.0, 2.5]
        fps = 10
        filename = os.path.join(write_dir, "animate_test.mp4")
        ani = animate_events(events, times, fps)

        from matplotlib.animation import FFMpegFileWriter
        kwargs = {'transparent': True,}
        writer = FFMpegFileWriter(fps=fps)
        ani.save(filename, dpi=300, writer=writer, savefig_kwargs=kwargs)

    def test_saveanim(self):
        events = ["hello", "world", "", "!", ""]
        times = [0., 1.0, 1.5, 2.0, 2.5]
        fps = 10
        filename = "animate_test_save.mp4"
        ani = animate_events(events, times, fps)
        saveanim(ani, write_dir, filename)

    def test_showanim(self):
        # don't know how to test this. trust me it works :)
        pass

    def test_animate_trajectory_3d(self):
        trajectory = np.zeros((10,3))
        trajectory[:,0] = np.arange(10)
        samplerate = 2
        axis_labels = ['x = Right', 'y = Forwards', 'z = Up']
        ani = animate_trajectory_3d(trajectory, samplerate, history=5, axis_labels=axis_labels)
        filename = "animate_trajectory_test.mp4"
        saveanim(ani, write_dir, filename)

    def test_animate_spatial_map(self):
        samplerate = 20
        duration = 5
        x_pos, y_pos = np.meshgrid(np.arange(0.5,10.5),np.arange(0.5, 10.5))
        data_map = []
        for frame in range(duration*samplerate):
            t = np.linspace(-1, 1, 100) + float(frame)/samplerate
            c = np.sin(t)
            data_map.append(get_data_map(c, x_pos.reshape(-1), y_pos.reshape(-1)))

        filename = 'spatial_map_animation.mp4'
        ani = animate_spatial_map(data_map, x_pos, y_pos, samplerate, cmap='bwr')
        saveanim(ani, write_dir, filename)

class OtherPlottingTests(unittest.TestCase):

    def test_plot_targets(self):

        # Draw four outer targets and one center target
        filename = 'targets.png'
        target_position = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0]
        ])
        target_radius = 0.1
        plt.figure()
        plot_targets(target_position, target_radius, (-2, 2, -2, 2))
        savefig(write_dir, filename)

        filename = 'targets_3d.png'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_targets(target_position, target_radius, (-2, 2, -2, 2, -2, 2), ax=ax)
        savefig(write_dir, filename)

    def test_plot_trajectories(self):

        # Test with two known trajectories
        filename = 'trajectories.png'
        trajectories =[
            np.array([
                [0, 0, 0],
                [1, 1, 0],
                [2, 2, 0],
                [3, 3, 0],
                [4, 2, 0]
            ]),
            np.array([
                [-1, 1, 0],
                [-2, 2, 0],
                [-3, 3, 0],
                [-3, 4, 0]
            ])
        ]
        plt.figure()
        bounds = (-5., 5., -5., 5., 0., 0.)
        plot_trajectories(trajectories, bounds)
        savefig(write_dir, filename)

        # Test 3D
        filename = 'trajectories_3d.png'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        bounds = (-5., 5., -5., 5., -5., 5.)
        plot_trajectories(trajectories, bounds, ax=ax)
        savefig(write_dir, filename)

        # Make some pretty spirals. There should be 4 spiral trajectories
        filename = 'spirals.png'
        samplerate = 60
        time = np.arange(200)/samplerate
        x = np.multiply(time, np.cos(np.pi * 10 * time))
        y = np.multiply(time, np.sin(np.pi * 10 * time))
        cursor = np.vstack((x, y)).T
        trial_times = np.array([(time[t], time[t+30]) for t in range(0, 200, 50)])
        trajectories = aopy.preproc.get_data_segments(cursor, trial_times, samplerate)
        plt.figure()
        plot_trajectories(trajectories, bounds)
        savefig(write_dir, filename)

    def test_color_trajectories(self):

        trajectories =[
                    np.array([
                        [0, 0, 0],
                        [1, 1, 0],
                        [2, 2, 0],
                        [3, 3, 0],
                        [4, 2, 0]
                    ]),
                    np.array([
                        [-1, 1, 0],
                        [-2, 2, 0],
                        [-3, 3, 0],
                        [-3, 4, 0]
                    ]),
                    np.array([
                        [2, 1, 0],
                        [2, -1, 0],
                        [3, -5, 0],
                        [5, -5, 0]
                    ])
                ]
        labels = [0, 0, 1]
        colors = ['r', 'b']
        plt.figure()
        color_trajectories(trajectories, labels, colors)
        filename = 'color_trajectories.png'
        savefig(write_dir, filename)

    def test_plot_sessions_by_date(self):
        from datetime import date, timedelta
        dates = [date.today() - timedelta(days=2), date.today() - timedelta(days=2), date.today()]
        success = [70, 65, 65]
        trials = [10, 20, 10]

        fig, ax = plt.subplots(1,1)
        plot_sessions_by_date(trials, dates, success, method='mean', labels=['success rate'], ax=ax)
        ax.set_ylabel('success (%)')

        filename = 'sessions_by_date.png'
        savefig(write_dir, filename) 
        # expect a plot of success with three days, with success rate of 
        # (70 * 10 + 65 * 20)/30 = 66.6% on the first day and 65% on the last day with a gap in between

        # Also make sure it works with dataframe columns
        df = pd.DataFrame({'trials': trials, 'dates': dates, 'success': success})
        fig, ax = plt.subplots(1,1)
        plot_sessions_by_date(df['trials'], df['dates'], df['success'], method='mean', ax=ax)

    def test_plot_sessions_by_trial(self):
        success = [70, 65, 60]
        trials = [10, 20, 10]

        fig, ax = plt.subplots(1,1)
        plot_sessions_by_trial(trials, success, labels=['success rate'], ax=ax)
        ax.set_ylabel('success (%)')
        filename = 'sessions_by_trial.png'
        savefig(write_dir, filename) 
        # expect a plot of success with 40 trials, with success rates of 70% for 10 trials,
        # 65% for 20 trials, and 60% for 10 trials

        # Also make sure it works with dataframe columns
        df = pd.DataFrame({'trials': trials, 'success': success})
        fig, ax = plt.subplots(1,1)
        plot_sessions_by_trial(df['trials'], df['success'], ax=ax)

    def test_plot_events_time(self):
        events = np.zeros(10)
        events[1::2] = 1
        event_list = [events, events[1:]]
        timestamps = np.arange(10)
        timestamps_list = [timestamps, 0.25+timestamps[1:]]
        labels_list = ['Event 1', 'Event 2']

        fig, ax = plt.subplots(1,1)
        plot_events_time(event_list, timestamps_list, labels_list, ax=ax)
        filename = 'events_time'
        savefig(write_dir,filename)

if __name__ == "__main__":
    unittest.main()