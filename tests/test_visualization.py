import unittest
from aopy.visualization import *
import aopy
import numpy as np
import os
import pickle

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

    def test_gradient_timeseries(self):
        filename = 'timeseries_gradient.png'
        data = np.reshape(np.sin(np.pi*np.arange(1000)/100), (1000))
        samplerate = 1000
        plt.figure()
        gradient_timeseries(data, samplerate)
        savefig(docs_dir, filename)

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
        savefig(docs_dir, filename, transparent=False)

        # Fill in the missing values by using calc_data_map instead of get_data_map
        filename = 'posmap_calcmap.png'
        interp_map, xy = calc_data_map(data_missing, x_missing, y_missing, [10, 10], threshold_dist=1.5)
        np.testing.assert_allclose(x_pos.reshape(-1), xy[0])
        np.testing.assert_allclose(y_pos.reshape(-1), xy[1])
        self.assertEqual(interp_map.shape, (10, 10))
        self.assertFalse(np.isnan(interp_map[0,0]))
        plt.figure()
        plot_spatial_map(interp_map, xy[0], xy[1])
        savefig(docs_dir, filename, transparent=False)

        # Use cubic interpolation to generate a high resolution map
        filename = 'posmap_calcmap_interp.png'
        interp_map, xy = calc_data_map(data_missing, x_missing, y_missing, [100, 100], threshold_dist=1.5, interp_method='cubic')
        self.assertEqual(interp_map.shape, (100, 100))
        plt.figure()
        plot_spatial_map(interp_map, xy[0], xy[1])
        savefig(docs_dir, filename, transparent=False)

        # Test using an alpha map on top of the spatial map
        filename = 'posmap_alphamap.png'
        data_map = get_data_map(data_missing, x_missing, y_missing)
        self.assertEqual(data_map.shape, (10, 10))
        plt.figure()
        plot_spatial_map(data_map, x_missing, y_missing, alpha_map=data_map)
        savefig(docs_dir, filename, transparent=False)

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
        elec_pos, acq_ch, elecs = aodata.load_chmap('ECoG244')
        missing_ch = acq_ch[np.isin(elecs, missing)]-1
        data[missing_ch] = np.nan

        plt.figure()
        plot_ECoG244_data_map(data, bad_elec=missing, interp=False, cmap='bwr', ax=None)
        filename = 'posmap_244ch_no_interp.png'
        savefig(write_dir, filename) # Here the missing electrodes (in addition to the ones
        # undefined by the channel mapping) should be visible in the map.

        plt.figure()
        plot_ECoG244_data_map(data, bad_elec=missing, interp=False, cmap='bwr', ax=None, nan_color=None)
        filename = 'posmap_244ch_no_interp_transparent.png'
        savefig(write_dir, filename) # Now we make the missing electrodes transparent

        plt.figure()
        plot_ECoG244_data_map(data, bad_elec=missing, interp=True, cmap='bwr', ax=None)
        filename = 'posmap_244ch.png'
        savefig(write_dir, filename) # Missing electrodes should be filled in with linear interp.

        plt.figure()
        elec_data = np.arange(240)
        plot_ECoG244_data_map(elec_data, elec_data=True)
        filename = 'posmap_244ch_elec_data.png'
        savefig(write_dir, filename) # No missing data

    def test_plot_spatial_drive_maps(self):

        im1 = np.arange(256).astype(float)
        im2 = np.flip(im1)
        im3 = im1.copy()
        np.random.shuffle(im3)
        im4 = np.flip(im3)
        maps = [im1, im2, im3, im4]
        plot_spatial_drive_maps(maps, (1,4), (2,2), cmap='viridis', clim=(0,255), label_mode="L")
        plt.tight_layout()
        filename = 'spatial_drive_maps_1_4.png'
        savefig(docs_dir, filename, transparent=False)

        plot_spatial_drive_maps(maps, (2,2), (2,2), cmap='viridis', clim=(0,255), cbar_mode='single')
        plt.tight_layout()
        filename = 'spatial_drive_maps_2_2_single_cbar.png'
        savefig(docs_dir, filename, transparent=False)

        fig, axes, ims, cbars = plot_spatial_drive_maps(maps, (2,2), (2,2), cmap='viridis', clim=(0,255), label_mode=None, cbar_mode='each', axes_pad=(0.4,0.05))
        ims[3].set_clim(127,255)
        plt.tight_layout()
        filename = 'spatial_drive_maps_2_2.png'
        savefig(docs_dir, filename, transparent=False)

    def test_annotate_spatial_map(self):
        plt.figure()
        plot_ECoG244_data_map(np.zeros(256,), cmap='Greys')
        annotate_spatial_map_channels(drive_type='ECoG244', color='k')
        annotate_spatial_map_channels(drive_type='Opto32', color='b')
        plt.axis('off')
        filename = 'ecog244_opto32.png'
        savefig(docs_dir, filename)

        plt.figure()
        plot_ECoG244_data_map(np.zeros(256,), cmap='Greys')
        annotate_spatial_map_channels(drive_type='ECoG244', color='k')
        annotate_spatial_map_channels(drive_type='Opto32', color='b')
        annotate_spatial_map_channels(drive_type='ECoG244', color='r', theta=90)
        annotate_spatial_map_channels(drive_type='Opto32', color='g', theta=90)
        plt.axis('off')
        filename = 'ecog244_opto32_theta90.png'
        savefig(docs_dir, filename)

        plt.figure()
        plot_ECoG244_data_map(np.zeros(256,), cmap='Greys', theta=90)
        annotate_spatial_map_channels(acq_idx=[45,46,55,56], drive_type='ECoG244', color='k', theta=90)
        annotate_spatial_map_channels(acq_ch=[11], drive_type='Opto32', color='b', theta=90)
        plt.axis('off')
        filename = 'ecog244_opto32_index_subset.png'
        savefig(write_dir, filename)

        plt.figure()
        plot_spatial_drive_map(np.zeros(64,), drive_type='EMG_GR08MM1305', cmap='Greys', theta=0)
        annotate_spatial_map_channels(drive_type='EMG_GR08MM1305', color='k', theta=0)
        filename = 'emg64_gr08mm1305.png'
        savefig(docs_dir, filename, transparent=False)

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

    def test_plot_tfr(self):

        fig, ax = plt.subplots(3,1,figsize=(4,6))

        samplerate = 1000
        data_200_hz = aopy.utils.generate_multichannel_test_signal(2, samplerate, 8, 200, 2)
        nt = data_200_hz.shape[0]
        data_200_hz[:int(nt/3),:] /= 3
        data_200_hz[int(2*nt/3):,:] *= 2

        data_50_hz = aopy.utils.generate_multichannel_test_signal(2, samplerate, 8, 50, 2)
        data_50_hz[:int(nt/2),:] /= 2

        data = data_50_hz + data_200_hz
        print(data.shape)
        aopy.visualization.plot_timeseries(data, samplerate, ax=ax[0])
        aopy.visualization.plot_freq_domain_amplitude(data, samplerate, ax=ax[1])

        freqs = np.linspace(1,250,100)
        freqs, t, coef = aopy.analysis.calc_cwt_tfr(data, freqs, samplerate, fb=10, f0_norm=1, verbose=True)
        pcm = aopy.visualization.plot_tfr(abs(coef[:,:,0]), t, freqs, 'plasma', ax=ax[2])

        fig.colorbar(pcm, label='Power', orientation = 'horizontal', ax=ax[2])
        filename = 'tfr_cwt_50_200.png'
        plt.tight_layout()
        savefig(docs_dir,filename)

    def test_plot_tf_map_grid(self):
        np.random.seed(0)
        
        nfreq = 100
        nt = 3
        nch = 100
        freqs = np.linspace(1,250,nfreq)
        time = np.linspace(0, 1, nt)
        tf_data = np.random.rand(nfreq,nt,nch)
        tf_data[:,1,:] *= 2 # increase power at time 1
        tf_data[freqs > 10, :, :] *= 0.5 # decrease power in high frequencies
        bands = [(1, 10), (10, 250)]
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        elec_pos = np.zeros((100,2))
        elec_pos[:,0] = x.reshape(-1)
        elec_pos[:,1] = y.reshape(-1)
        plot_tf_map_grid(freqs, time, tf_data, bands, elec_pos, clim=(0,1), interp_grid=None, 
                     cmap='viridis')
        filename = 'tf_map_grid.png'
        savefig(docs_dir, filename, transparent=False)

    def test_plot_corr_over_elec_distance(self):

        duration = 0.5
        samplerate = 1000
        n_channels = 30
        frequency = 100
        amplitude = 0.5
        elec_data = aopy.utils.generate_multichannel_test_signal(duration, samplerate, n_channels, frequency, amplitude)
        elec_pos = np.stack((range(n_channels), np.zeros((n_channels,))), axis=-1)
        
        plt.figure()
        plot_corr_over_elec_distance(elec_data, elec_pos, label='test')
        filename = 'corr_over_dist.png'
        savefig(docs_dir,filename)

    def test_plot_corr_across_entries(self):
        subjects = ['beignet', 'beignet']
        ids = [5974, 5974]
        dates = ['2022-07-01', '2022-07-01']
        plt.figure()
        plot_corr_across_entries(data_dir, subjects, ids, dates)
        filename = 'corr_over_entries.png'
        savefig(docs_dir,filename)

    def test_plot_angles(self):

        # Test with no magnitudes
        angles = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 8, endpoint=False)
        plot_angles(angles)

        filename = 'angles_simple.png'
        savefig(docs_dir, filename, transparent=False)

        # Test with magnitudes
        angles = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 8, endpoint=False)
        magnitudes = np.arange(len(angles)) + 1

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        plot_angles(angles, magnitudes, ax)

        filename = 'angles_magnitudes.png'
        savefig(docs_dir, filename, transparent=False)

class StimPlottingWrapperTests(unittest.TestCase):

    def test_plot_annotated_stim_drive_data(self):
        np.random.seed(0)
        data = np.random.normal(0, 1, (32,))

        plt.figure()
        plot_annotated_stim_drive_data(data, 'beignet', 'lm1', 0)

        filename = 'annotated_stim_drive_data.png'
        savefig(docs_dir, filename, transparent=False)

    def test_plot_annotated_spatial_drive_map_stim(self):
        np.random.seed(0)
        data = np.random.normal(0, 1, (240,))
        stim_site = 7

        plt.figure()
        plot_annotated_spatial_drive_map_stim(data, stim_site, 'beignet', 'lm1', 0, interp_method='cubic')

        filename = 'annotated_spatial_drive_map_stim.png'
        savefig(docs_dir, filename, transparent=False)

class CurveFittingTests(unittest.TestCase):
    def test_plot_tuning_curves(self):
        filename = 'tuning_curves_plot.png'
        nunits = 7
        targets = np.arange(0, 360, 45)
        mds_true = np.linspace(1, 3, nunits)/2
        pds_offset = np.arange(-45,270,45)
        data = np.zeros((nunits,8))*np.nan
        np.random.seed(0)
        for ii in range(nunits):
            noise = np.random.normal(1, 0.2, size=(1,8))
            data[ii,:] = noise*mds_true[ii]*np.sin(np.deg2rad(targets)-np.deg2rad(pds_offset[ii])) + 2

        # If the mds and pds output are correct the fitting params are correct because they are required for the calculation.
        # Test without ax input
        fit_params, _, _ = aopy.analysis.run_tuningcurve_fit(data, targets)
        plot_tuning_curves(fit_params, data, targets, n_subplot_cols=4)
        savefig(docs_dir, filename, transparent=False)

        # test with ax input
        fig, ax = plt.subplots(2,4)
        plot_tuning_curves(fit_params, data, targets, n_subplot_cols=4, ax=ax)

    def test_plot_direction_tuning(self):
        np.random.seed(0)
        direction = [-np.pi, -np.pi/2, 0, np.pi/2]
        data = np.random.normal(0, 1, (4, 2))
        
        plt.figure()
        plot_direction_tuning(data, direction, show_var=False)
        savefig(write_dir, 'direction_tuning_simple.png', transparent=False)

        # Again with polar plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')

        plot_direction_tuning(data, direction, wrap=False, show_var=False)
        savefig(write_dir, 'direction_tuning_simple_polar.png', transparent=False)

        # Try multichannel
        direction = [-np.pi, -np.pi/2, 0, np.pi/2]
        data = np.random.normal(0, 1, (4, 2, 4))
        
        plt.figure()
        plot_direction_tuning(data, direction)
        savefig(docs_dir, 'direction_tuning.png', transparent=False)

        plt.figure()
        plot_direction_tuning(data, np.degrees(direction))
        savefig(write_dir, 'direction_tuning_degrees.png', transparent=False)

        # Again with polar plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')

        plot_direction_tuning(data, direction)
        savefig(docs_dir, 'direction_tuning_polar.png', transparent=False)

        # Test the categorical plot
        fig = plt.figure()
        plot_condition_tuning(data, np.degrees(direction))
        savefig(docs_dir, 'condition_tuning.png', transparent=False)

        # Make sure both work with a 180 degree range
        direction = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        plt.figure()
        plot_direction_tuning(data, direction)
        savefig(write_dir, 'direction_tuning_modulo.png', transparent=False)

    def test_plot_boxplots(self):
        # Rectangular array
        np.random.seed(0)
        data = np.random.normal(0, 2, size=(20, 5))
        xaxis_pts = np.array([2,3,4,4.75,5.5])
        fig, ax = plt.subplots(1,1)
        plot_boxplots(data, xaxis_pts, ax=ax)
        filename = 'boxplot_example.png'
        savefig(docs_dir, filename, transparent=False)

        # List of nonrectangular arrays
        data = [np.random.normal(0, 2, size=(10)), np.random.normal(0, 1, size=(20))]
        xaxis_pts = ['foo', 'bar']
        fig, ax = plt.subplots(1,1)
        plot_boxplots(data, xaxis_pts, ax=ax)
        filename = 'boxplot_example_nonrectangular.png'
        savefig(docs_dir, filename, transparent=False)


class AnimationTests(unittest.TestCase):

    def test_animate_events(self):
        events = ["hello", "world", "", "!", ""]
        times = [0., 1.0, 1.5, 2.0, 2.5]
        fps = 10
        filename = os.path.join(docs_dir, "test_anim_events.mp4")
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
        filename = "test_anim_trajectory.mp4"
        saveanim(ani, docs_dir, filename)

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
        saveanim(ani, docs_dir, filename)

    def test_animate_cursor_eye(self):
        cursor_trajectory = np.array([[0,0], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        eye_trajectory = np.array([[1, 0], [1, 2], [1, 2], [4, 5], [4, 5], [6, 6]])
        samplerate = 0.5
        target_positions = [np.array([0,0]), np.array([5,5])]
        bounds = (-10, 10, -10, 10, 0, 0)
        target_radius = 1.5

        ani = animate_cursor_eye(cursor_trajectory, eye_trajectory, samplerate, target_positions, target_radius, 
                        bounds)
        
        aopy.visualization.saveanim(ani, docs_dir, 'test_anim_cursor_eye.mp4')

    def test_animate_behavior(self):

        samplerate = 0.5
        cursor = np.array([[0,0], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        eye = np.array([[1, 0], [1, 2], [1, 2], [4, 5], [4, 5], [6, 6]])
        targets = [
            np.array([[np.nan, np.nan], 
                     [5, 5], 
                     [np.nan, np.nan], 
                     [np.nan, np.nan], 
                     [5, 5], 
                     [np.nan, np.nan]]),
            np.array([[np.nan, np.nan], 
                     [np.nan, np.nan], 
                     [np.nan, np.nan], 
                     [-5, 5], 
                     [-5, 5], 
                     [-5, 5]])
        ]
        
        target_radius = 2.5
        target_colors = ['orange'] * len(targets)
        cursor_radius = 0.5
        bounds = [-10, 10, -10, 10]
        
        ani = animate_behavior(targets, cursor, eye, samplerate, bounds, target_radius, target_colors, cursor_radius, 
                        cursor_color='blue', eye_radius=0.25, eye_color='purple')
        
        aopy.visualization.saveanim(ani, docs_dir, 'test_anim_behavior.mp4')
                

class KinematicsPlottingTests(unittest.TestCase):

    def test_color_targets(self):
        # Generate 8 targets at radius 6.5 from the center
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        radius = 6.5
        target_locations = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        
        # Add the center target
        target_locations = np.vstack(([0, 0], target_locations))
        target_idx = [0] + np.arange(1, 9).tolist()  # Center is index 0, peripheral are index 1 through 9

        # Choose plotting parameters
        colors = ['black'] + sns.color_palette("husl", 8)
        target_radius = 0.5
        bounds = (-8, 8, -8, 8)

        # Plot the targets
        fig, ax = plt.subplots(figsize=(8, 8))
        color_targets(target_locations, target_idx, colors, target_radius, bounds, ax)
        ax.set_aspect('equal')
        filename = 'color_targets.png'
        savefig(docs_dir, filename, transparent=False)

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

    def test_color_targets_3D(self):
        from mpl_toolkits.mplot3d import Axes3D
        import seaborn as sns

        targets = np.array([
            [0., 0., 0.],
            [0., 10., 0.],
            [7.0711, 7.0711, 0.],
            [10., 0., 0.],
            [7.0711, -7.0711, 0.],
            [0., -10., 0.],
            [-7.0711, -7.0711, 0.],
            [-10., 0., 0.],
            [-7.0711, 7.0711, 0.]
        ])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim3d([-10, 10])

        colors = sns.color_palette(n_colors=len(targets))
        aopy.visualization.color_targets_3D(targets, target_idx=np.arange(len(targets)), target_radius=1, colors=colors, ax=ax)

        filename = 'color_targets_3D.png'
        savefig(docs_dir, filename)

    def test_plot_sphere(self):
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_sphere(location=(0, 1, 2), color='blue', radius=5, resolution=30, alpha=0.5, ax=ax)

        filename = 'plot_sphere_example.png'
        savefig(docs_dir, filename)

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

        trajectories = [
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
        plt.title('Categorized trajectories')
        filename = 'color_trajectories_simple.png'
        savefig(docs_dir, filename)

        # Generate the second plot with segmented trajectories
        labels_list = [[0, 0, 1, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]]
        fig, ax = plt.subplots()
        color_trajectories(trajectories, labels_list, colors)
        plt.title('Segmented trajectories')
        filename = 'color_trajectories_segmented.png'
        savefig(docs_dir, filename)

    def test_gradient_trajectories(self):

        trajectories = [
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
        plt.figure()
        gradient_trajectories(trajectories, n_colors=4)
        plt.title('Gradient trajectories')
        filename = 'gradient_trajectories_simple.png'
        savefig(write_dir, filename)
        plt.close()

        # Test what happens when the number of colors is higher than the number of points
        plt.figure()
        gradient_trajectories(trajectories, n_colors=5)
        plt.title('Gradient trajectories')
        filename = 'gradient_trajectories_error.png'
        savefig(write_dir, filename)
        plt.close()

        # Load some test cursor data
        subject = 'beignet'
        te_id = 5974
        date = '2022-07-01'
        preproc_dir = data_dir
        traj, _ = aopy.data.get_kinematic_segments(preproc_dir, subject, te_id, date, [32], [81, 82, 83, 239], datatype='cursor')
        plt.figure()
        gradient_trajectories(traj[:3])
        filename = 'gradient_trajectories.png'
        savefig(docs_dir, filename)
        plt.close()

        # Hand data plotted in 3d
        traj, _ = aopy.data.get_kinematic_segments(preproc_dir, subject, te_id, date, [32], [81, 82, 83, 239], datatype='user_world')
        plt.figure()
        ax = plt.axes(projection='3d')
        gradient_trajectories(traj[:3], bounds=[-10,10,-10,10,-10,0], ax=ax)

        filename = 'gradient_trajectories_3d.png'
        savefig(docs_dir, filename, transparent=False)
        plt.close()

    def test_plot_plane(self):
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Example using a transformation matrix (identity)
        plane = np.eye(3)  
        plot_plane(plane, gain=1.0, color='blue', alpha=0.3, ax=ax)

        # Example using a plane equation Ax + By + Cz + D = 0
        plane_eq = np.array([1, 2, -1, 5])  # x + 2y - z + 5 = 0
        plot_plane(plane_eq, gain=1.0, color='red', alpha=0.5, ax=ax)

        filename = 'plot_plane_example.png'
        savefig(docs_dir, filename)
        
    def test_get_color_gradient_RGB(self):
        npts = 200
        x = np.linspace(0, 2*np.pi, npts)
        y = np.sin(x)
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=aopy.visualization.get_color_gradient_RGB(npts, 'g', [1,0,0]))
        filename = 'color_gradient_example.png'
        savefig(docs_dir, filename)

    def test_plot_sessions_by_date(self):
        from datetime import date, timedelta
        today = date(1991, 9, 26)
        dates = [today - timedelta(days=2), today - timedelta(days=2), today]
        success = [70, 65, 65]
        trials = [10, 20, 10]

        fig, ax = plt.subplots(1,1)
        plot_sessions_by_date(trials, dates, success, method='mean', labels=['success rate'], ax=ax)
        ax.set_ylabel('success (%)')

        filename = 'sessions_by_date.png'
        savefig(docs_dir, filename) 
        # expect a plot of success with three days, with success rate of 
        # (70 * 10 + 65 * 20)/30 = 66.6% on the first day and 65% on the last day with a gap in between

        # Also make sure it works with dataframe columns
        df = pd.DataFrame({'trials': trials, 'dates': dates, 'success': success})
        fig, ax = plt.subplots(1,1)
        plot_sessions_by_date(df['trials'], df['dates'], df['success'], method='mean', ax=ax)

    def test_plot_sessions_by_trial(self):
        success = [70, 65, 60, 70, 59, 62, 71]
        trials = [5, 12, 7, 8, 12, 4, 10]

        fig, ax = plt.subplots(1,1)
        plot_sessions_by_trial(trials, success, labels=['success rate'], ax=ax)
        ax.set_ylabel('success (%)')
        filename = 'sessions_by_trial.png'
        savefig(write_dir, filename) 
        # expect a plot of success with 40 trials, with success rates of 70% for 10 trials,
        # 65% for 20 trials, and 60% for 10 trials

        # Also make sure it works with dataframe columns
        df = pd.DataFrame({'trials': trials, 'success': success})
        fig, ax = plt.subplots(3,1, figsize=(4,6))
        plot_sessions_by_trial(df['trials'], df['success'], ax=ax[0])
        ax[0].set_ylabel('success (%)')
        ax[0].set_title('plot trials')

        # Split by date
        from datetime import date, timedelta
        today = date(1991, 9, 26)
        df['date'] = [today - timedelta(days=len(trials)-np.floor(n/2)) for n in range(len(trials))]
        plot_sessions_by_trial(df['trials'], df['success'], dates=df['date'], ax=ax[1])
        ax[1].set_ylabel('success (%)')
        ax[1].set_title('separate by date')

        # Smoothing
        plot_sessions_by_trial(df['trials'], df['success'], dates=df['date'], smoothing_window=3, ax=ax[2])
        ax[2].set_ylabel('success (%)')
        ax[2].set_title('smoothing window = 3 trials')

        plt.tight_layout()
        savefig(docs_dir, filename)

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

    def test_plot_circular_hist(self):
        fig, ax = plt.subplots(3, 2, subplot_kw=dict(projection='polar'), figsize=(12,18))
        np.random.seed(0)
        angles = np.random.normal(loc=np.pi/4, scale=np.pi/8, size=1000)

        # compare plotting the same data with vs. without allowing gaps in the bins
        plot_circular_hist(angles, bins=16, ax=ax[0,0], gaps=False, edgecolor='tab:blue', fill=False)
        plot_circular_hist(angles, bins=16, ax=ax[0,1], gaps=True, edgecolor='tab:blue', fill=False)
        ax[0,1].set_title('Bins not forced to span across entire circle')

        # compare plotting the same data with value represented by bar radius vs. area 
        plot_circular_hist(angles, bins=16, ax=ax[1,0], proportional_area=False, edgecolor='tab:blue', fill=False)
        plot_circular_hist(angles, bins=16, ax=ax[1,1], proportional_area=True, edgecolor='tab:blue', fill=False)
        ax[1,1].set_title('Value in bin represented by bar area, not radius')

        # compare plotting the same data plotted as a probability density function or normalized by the max bin value
        plot_circular_hist(angles, bins=16, ax=ax[2,0], density=True, edgecolor='tab:blue', fill=False)
        ax[2,0].set_title('Bin values represent the probability density function')
        plot_circular_hist(angles, bins=16, ax=ax[2,1], normalize=True, edgecolor='tab:blue', fill=False)
        ax[2,1].set_title('Bin values are normalized to a max value of 1')

        filename = 'circular_histograms'
        savefig(docs_dir, filename, transparent=False)

class TestPlotUtils(unittest.TestCase):

    @unittest.skip("bug in new versions of matplotlib, waiting for resolution")
    def test_advance_plot_color(self):
        plt.subplots()
        aopy.visualization.advance_plot_color(plt.gca(), 1)
        plt.plot(np.arange(10), np.arange(10))

        filename = 'advance_plot_color.png'
        savefig(docs_dir,filename)

    def test_reset_plot_color(self):
        plt.subplots()
        plt.plot(np.arange(10), np.ones(10))
        aopy.visualization.reset_plot_color(plt.gca())
        plt.plot(np.arange(10), 1 + np.ones(10))

        filename = 'reset_plot_color.png'
        savefig(docs_dir,filename)

    def test_plot_scalebar(self):
        plt.subplots()

        plt.plot(np.arange(10), np.arange(10)/10)
        aopy.visualization.plot_scalebar(plt.gca(), 1, '1 s', color='orange')
        aopy.visualization.plot_scalebar(plt.gca(), 0.1, '0.1 V', vertical=True, color='green')
        aopy.visualization.plot_xy_scalebar(plt.gca(), 1, '1 s', 0.1, '0.1 V', bbox_to_anchor=(0.8, 0.1))
        filename = 'scalebar_example.png'
        savefig(docs_dir,filename)

    def test_savefig(self):

        fig, ax = plt.subplots()
        ax.patch.set_facecolor('black')

        aopy.visualization.savefig(write_dir, "axis_transparency.png", transparent=False)

    def test_subplots_with_labels(self):

        # Test case 1: generate a figure with 2 rows and 2 columns of subplots, labeled A, B, C, D
        fig, axes = subplots_with_labels(2, 2, constrained_layout=True)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2, 2)
        assert isinstance(axes[0, 0], plt.Axes)
        assert isinstance(axes[0, 1], plt.Axes)
        assert isinstance(axes[1, 0], plt.Axes)
        assert isinstance(axes[1, 1], plt.Axes)
        aopy.visualization.savefig(docs_dir, "labeled_subplots.png")

    def test_place_subplots(self):
        fig = plt.figure(figsize=(4,6))
        positions = [[1, 2], [3, 4]]
        width = 1
        height = 1
        ax = place_subplots(fig, positions, width, height)
        ax[0].annotate('1', (0.5,0.5), ha='center', va='center', fontsize=40)
        ax[1].annotate('2', (0.5,0.5), ha='center', va='center', fontsize=40)
        aopy.visualization.savefig(docs_dir, "place_subplots_1.png", transparent=False)

        fig = plt.figure(figsize=(4,6))
        positions = [[1, 1.5], [3, 4.5]]
        width = 2
        height = 3
        ax = place_subplots(fig, positions, width, height)
        ax[0].annotate('1', (0.5,0.5), ha='center', va='center', fontsize=40)
        ax[1].annotate('2', (0.5,0.5), ha='center', va='center', fontsize=40)
        aopy.visualization.savefig(docs_dir, "place_subplots_2.png", transparent=False)

    def test_place_Opto32_subplots(self):
        fig, ax = place_Opto32_subplots()
        for i, ax in enumerate(ax):
            ax.annotate(str(i+1), (0.5,0.5), ha='center', va='center',  fontsize=40)
        aopy.visualization.savefig(docs_dir, "place_Opto32_subplots.png", transparent=False)

    def test_overlay_image_on_spatial_map(self):
        plt.figure()
        elec_pos, acq_ch, elecs = aodata.load_chmap('ECoG244')
        plot_spatial_map(np.arange(16*16).reshape((16,16)), elec_pos[:,0], elec_pos[:,1])
        overlay_sulci_on_spatial_map('beignet', 'LM1', 'ECoG244')
        filename = 'overlay_sulci_beignet.png'
        savefig(docs_dir, filename, transparent=False)

        plt.figure()
        plot_spatial_map(np.arange(16*16).reshape((16,16)), elec_pos[:,0], elec_pos[:,1])
        overlay_sulci_on_spatial_map('affi', 'LM1', 'ECoG244', theta=90)
        filename = 'overlay_sulci_affi.png'
        savefig(docs_dir, filename, transparent=False)

class TestEyePlots(unittest.TestCase):

    def test_plot_eye_calibration_result(self):
        
        subject = 'beignet'
        te_id = 5974
        date = '2022-07-01'
        preproc_dir = data_dir
        exp_data, exp_metadata = aopy.data.load_preproc_exp_data(preproc_dir, subject, te_id, date)
        eye_data, eye_metadata = aopy.data.load_preproc_eye_data(preproc_dir, subject, te_id, date)

        eye_raw = eye_data['raw_data']
        eye_samplerate = eye_metadata['samplerate']
        cursor_data = exp_data['cursor_interp']
        cursor_samplerate = exp_metadata['cursor_interp_samplerate']
        coeff, correlation_coeff, cursor_calibration_data, eye_calibration_data = aopy.preproc.calc_eye_calibration(
            cursor_data, cursor_samplerate, eye_raw, eye_samplerate,
            exp_data['events']['timestamp'],
            exp_data['events']['code'],
            align_events=list(range(81,89)),
            penalty_events=[64],
            return_datapoints=True,
        )

        aopy.visualization.eye.plot_eye_calibration_result(eye_calibration_data, cursor_calibration_data, coeff, correlation_coeff)

        filename = 'eye_calibration.png'
        savefig(docs_dir,filename, transparent=False)

class TestDecoderPlots(unittest.TestCase):

    def test_plot_decoder_summary(self):

        from aopy.data import db
        with open(os.path.join(data_dir, 'test_decoder.pkl'), 'rb') as file:
            decoder = pickle.load(file, fix_imports=False)

        bmi3d.plot_decoder_summary(decoder)
        filename = 'decoder_weights.png'
        savefig(docs_dir, filename, transparent=False)

        with open(os.path.join(data_dir, 'test_decoder_emg.pkl'), 'rb') as file:
            decoder = pickle.load(file, fix_imports=False)

        bmi3d.plot_decoder_summary(decoder, drive_type='EMG_GR08MM1305')
        filename = 'decoder_weights_emg.png'
        savefig(docs_dir, filename, transparent=False)

if __name__ == "__main__":
    unittest.main()