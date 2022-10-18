from pydoc import doc
import unittest
from aopy.visualization import *
import aopy
import numpy as np
import os

test_dir = os.path.dirname(__file__)
data_dir = os.path.join(test_dir, 'data')
write_dir = os.path.join(test_dir, 'tmp')
docs_dir = os.path.join(os.path.dirname(test_dir),'docs', 'source', '_images')
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

        # Fill in the missing values by using calc_data_map instead of get_data_map
        filename = 'posmap_calcmap.png'
        interp_map, xy = calc_data_map(data_missing, x_missing, y_missing, [10, 10], threshold_dist=1.5)
        self.assertEqual(interp_map.shape, (10, 10))
        self.assertFalse(np.isnan(interp_map[0,0]))
        plt.figure()
        plot_spatial_map(interp_map, xy[0], xy[1])
        savefig(write_dir, filename)

        # Use cubic interpolation to generate a high resolution map
        filename = 'posmap_calcmap_interp.png'
        interp_map, xy = calc_data_map(data_missing, x_missing, y_missing, [100, 100], threshold_dist=1.5, interp_method='cubic')
        self.assertEqual(interp_map.shape, (100, 100))
        plt.figure()
        plot_spatial_map(interp_map, xy[0], xy[1])
        savefig(write_dir, filename)

        # Test using an alpha map on top of the spatial map
        filename = 'posmap_alphamap.png'
        data_map = get_data_map(data_missing, x_missing, y_missing)
        self.assertEqual(data_map.shape, (10, 10))
        plt.figure()
        plot_spatial_map(data_map, x_missing, y_missing, alpha_map=data_map)
        savefig(docs_dir, filename)


    def test_single_spatial_map(self):
        data = 2.0
        x_pos, y_pos = np.meshgrid(1,1)

        data_map = get_data_map(data, x_pos, y_pos)
        self.assertEqual(data_map[0], 2.0)
        plt.figure()
        plot_spatial_map(data_map, x_pos, y_pos)
        filename = 'posmap_single.png'
        savefig(write_dir, filename)

    def test_plot_ECoG244_data_map(self):
        data = np.linspace(-1, 1, 256)
        missing = [0, 5, 25]
        plt.figure()
        plot_ECoG244_data_map(data, bad_elec=missing, interp=False, cmap='bwr', ax=None)
        filename = 'posmap_244ch_no_interp.png'
        savefig(write_dir, filename) # Here the missing electrodes (in addition to the ones
        # undefined by the channel mapping) should be visible in the map.

        plt.figure()
        plot_ECoG244_data_map(data, bad_elec=missing, interp=True, cmap='bwr', ax=None)
        filename = 'posmap_244ch.png'
        savefig(write_dir, filename) # Missing electrodes should be filled in with linear interp.

    def test_plot_image_by_time(self):
        time = np.array([-2, -1, 0, 1, 2, 3])
        data = np.array([[0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 0]]).T
        plt.figure()
        plot_image_by_time(time, data)
        filename = 'image_by_time.png'
        savefig(docs_dir, filename)

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

    def test_profile_data_channels(self):
        ch_list = [0,5]
        ds_factor = 25
        test_data_folder = 'fake ecube data'
        figure_dir = os.path.join(write_dir,'fake_data_ch_profile_test')
        
        test_data, test_mdata = aopy.data.load_ecube_analog(data_dir,test_data_folder)
        test_data = test_data[::ds_factor,ch_list] * test_mdata['voltsperbit']
        samplerate = test_mdata['samplerate'] // ds_factor

        profile_data_channels(test_data, samplerate, figure_dir, cmap_lim=(0,1))
        fig_out_path = os.path.join(docs_dir, 'channel_profile_example.png')
        os.replace(
            os.path.join(figure_dir,'all_ch.png'),
            fig_out_path
        )
    
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
        # Test without ax input
        fit_params, _, _ = aopy.analysis.run_tuningcurve_fit(data, targets)
        plot_tuning_curves(fit_params, data, targets, n_subplot_cols=4)

        # test with ax input
        fig, ax = plt.subplots(2,4)
        plot_tuning_curves(fit_params, data, targets, n_subplot_cols=4, ax=ax)
        
    def test_plot_boxplots(self):
        data = np.random.normal(0, 2, size=(20, 5))
        xaxis_pts = np.array([2,3,4,4.75,5.5])
        fig, ax = plt.subplots(1,1)
        plot_boxplots(data, xaxis_pts, ax=ax)
        filename = 'boxplot_example.png'
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

class KinematicPlottingTests(unittest.TestCase):

    def test_plot_random_segments(self):
        
        plot_random_segments(data, event_start_times, event_end_times, samplerate=1, num_plots=3, ax=None)
    
    def test_plot_boxplots_from_dict(self):
        
        plot_boxplots_from_dict(data, idx_entries_selected, which_event, ax)
        
    def test_plot_cursor_kinematics_around_saccade(self):
    
        plot_cursor_kinematics_around_saccade(saccade_start_times, saccade_end_times, dists, vels, 
                                            num_samples_before, num_samples_after, 
                                            segment_conditions, selected_condition, segment_results, selected_result, 
                                            which_saccade, selected_condition_name='')
        
    def test_plot_eye_vs_cursor_scatter(self):
        
        plot_eye_vs_cursor_scatter(eye_data, cursor_data, eye_samplerate=1e3, cursor_samplerate=1e3, ax=None, xlabel='', ylabel='', title='')
        
    def test_plot_event_lines(self):
        
        plot_event_lines(cur_init_time, sac_start_times, sac_end_times, event_times, event_codes, ymin, ymax, ax)
        
        
    def test_plot_error_angle(self):
        
        plot_error_angle(cursor_angle, eye_angle, cursor_init_time, saccade_start_times, saccade_end_times, 
                        event_times, event_codes, cursor_samplerate, eye_samplerate, ax)

    def test_create_zone_line(self):
        
        create_zone_line(zone, samplerate, colors, height, ax)
        

    def test_plot_zone_lines(self):
        
        plot_zone_lines(cursor_zone, eye_zone, colors, cursor_samplerate, eye_samplerate, ax)
        

    def test_plot_distance_from_target(self):
        
        plot_distance_from_target(cursor_dist, eye_dist, cursor_init_time, saccade_start_times, saccade_end_times, 
                                event_times, event_codes, cursor_samplerate, eye_samplerate, ax)
        
    def test_plot_window_around_event(self):
        
        plot_window_around_event(cursor_angle, eye_angle, event_times, which_event, num_samples_before, num_samples_after, 
                                segment_conditions, selected_condition, condition_name, result_segments, selected_result)

    def test_order_segments(self):
        
        order_segments(data_segments, idx_segments_selected, order='')

    def test_plot_zone_heatmap(self):
        
        plot_zone_heatmap(data_segments, event_times, which_event, segment_conditions, selected_condition, 
                      condition_name, result_segments, selected_result, samplerate, segment_order=[], 
                      order='', color_list=['r', 'g', 'b', 'c', 'm', 'y', 'w'], plot_cbar=True, 
                      zone_labels = ['CENTER', 'CENT<->SURR', 'SURROUND', 'NEAR_CENT', 'NEAR_SURR', 'BET_CENT_SURR', 'ELSEWHERE'],
                      with_title=True, fig=None, ax=None)

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

    def test_plot_circles(self):

            # Draw four outer targets and one center target
            filename = 'circles.png'
            target_position = np.array([
                [0, 0, 0],
                [1, 1, 0],
                [-1, 1, 0],
                [1, -1, 0],
                [-1, -1, 0]
            ])
            target_radius = 0.1
            target_color = 'b'
            plt.figure()
            plot_circles(target_position, target_radius, target_color)
            savefig(write_dir, filename)

            filename = 'circles_3d.png'
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plot_circles(target_position, target_radius, target_color, (-2, 2, -2, 2, -2, 2), ax=ax)
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

    def test_advance_plot_color(self):
        # Nothing to test here ;-)
        pass

    def test_plot_corr_over_elec_distance(self):

        duration = 0.5
        samplerate = 1000
        n_channels = 30
        frequency = 100
        amplitude = 0.5
        acq_data = aopy.utils.generate_multichannel_test_signal(duration, samplerate, n_channels, frequency, amplitude)
        acq_ch = (np.arange(n_channels)+1).astype(int)
        elec_pos = np.stack((range(n_channels), np.zeros((n_channels,))), axis=-1)
        
        plt.figure()
        plot_corr_over_elec_distance(acq_data, acq_ch, elec_pos, label='test')
        filename = 'corr_over_dist.png'
        savefig(docs_dir,filename)

if __name__ == "__main__":
    unittest.main()