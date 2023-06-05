# test_data.py 
# tests of aopy.data
from aopy import visualization
from aopy.data import *
from aopy.data import peslab
from aopy.data import optitrack
from aopy.data import bmi3d
from aopy.postproc.base import get_calibrated_eye_data
from aopy.visualization import *
from aopy import preproc
import unittest
import os
import numpy as np
import pandas as pd
from matplotlib.testing.compare import compare_images

test_dir = os.path.dirname(__file__)
data_dir = os.path.join(test_dir, 'data')
write_dir = os.path.join(test_dir, 'tmp')
docs_dir = os.path.join(test_dir, '../docs/source/_images')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)
test_filepath = os.path.join(data_dir, "short headstage test")
sim_filepath = os.path.join(data_dir, "fake ecube data")

class LoadPreprocTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        files = {}
        files['hdf'] = 'fake_ecube_data_bmi3d.hdf'
        files['ecube'] = 'fake ecube data'
        cls.id = 3498
        cls.subject = 'fake_subject'
        cls.date = '2021-12-13'
        preproc.proc_single(data_dir, files, write_dir, cls.subject, cls.id, cls.date, ['exp', 'eye', 'broadband', 'lfp'], overwrite=True) # without ecube data

    def test_load_preproc_exp_data(self):
        exp_data, exp_metadata = load_preproc_exp_data(write_dir, self.subject, self.id, self.date)
        self.assertIsInstance(exp_data, dict)
        self.assertIsInstance(exp_metadata, dict)

    def test_load_preproc_eye_data(self):
        eye_data, eye_metadata = load_preproc_eye_data(write_dir, self.subject, self.id, self.date)
        self.assertIsInstance(eye_data, dict)
        self.assertIsInstance(eye_metadata, dict)
            
    def test_load_preproc_broadband_data(self):
        broadband_data, broadband_metadata = load_preproc_broadband_data(write_dir, self.subject, self.id, self.date)
        self.assertIsInstance(broadband_data, np.ndarray)
        self.assertIsInstance(broadband_metadata, dict)

    def test_load_preproc_lfp_data(self):
        lfp_data, lfp_metadata = load_preproc_lfp_data(write_dir, self.subject, self.id, self.date)
        self.assertIsInstance(lfp_data, np.ndarray)
        self.assertIsInstance(lfp_metadata, dict)

    def test_find_preproc_ids_from_day(self):
        ids = find_preproc_ids_from_day(write_dir, self.subject, self.date, 'exp')
        self.assertIn(self.id, ids)
        self.assertEqual(len(ids), 1)

    def test_proc_eye_day(self):
        self.assertRaises(ValueError, lambda:proc_eye_day(write_dir, self.subject, self.date))
        best_id, te_ids = proc_eye_day(data_dir, 'test', '2022-08-19', correlation_min=0, dry_run=True)
        self.assertIsNone(best_id)
        self.assertCountEqual(te_ids, [6581, 6577])

class OptitrackTests(unittest.TestCase):
        
    def test_load_mocap(self):
        # Data directory and filepath
        filename = 'Take 2021-03-10 17_56_55 (1039).csv'

        # Load Meta data
        mocap_metadata = optitrack.load_optitrack_metadata(data_dir, filename)
        assert mocap_metadata['samplerate'] == 240
        assert mocap_metadata['Rigid Body Name'] == 'Hand'
        assert mocap_metadata['Length Units'] == 'Meters'
        
        # Load Data
        mocap_data_pos, mocap_data_rot = optitrack.load_optitrack_data(data_dir, filename)
        assert mocap_data_pos.shape[0] == 92383
        assert mocap_data_rot.shape[0] == 92383
        assert mocap_data_pos.shape[1] == 3
        assert mocap_data_rot.shape[1] == 4

        # Load timestamps
        mocap_timestamps = optitrack.load_optitrack_time(data_dir, filename)
        assert mocap_timestamps.shape[0] == 92383
        assert mocap_timestamps.ndim == 1
        diff = np.diff(mocap_timestamps)
        assert (diff > 0).all(), 'Should be monotonically increasing'

class BMI3DTests(unittest.TestCase):

    def test_get_filenames_in_dir(self):
        test_dir = os.path.join(data_dir, 'test_filenames_dir')
        files = get_filenames_in_dir(test_dir, 1039)
        self.assertIn('foo', files)
        self.assertIn('bar', files)
        self.assertEqual(files['foo'], os.path.join('foo','1039_foo'))
        self.assertEqual(files['bar'], os.path.join('bar','1039_bar.txt'))

    def test_get_ecube_data_sources(self):
        sources = bmi3d.get_ecube_data_sources(test_filepath)
        assert len(sources) == 1
        assert sources[0] == "Headstages"

    def test_load_ecube_metadata(self):
        metadata = bmi3d.load_ecube_metadata(test_filepath, 'Headstages')
        assert metadata['n_channels'] == 64
        assert metadata['n_samples'] == 214032
        assert metadata['data_source'] == 'Headstages'
        assert metadata['samplerate'] == 25000.

    def test_process_channels(self):
        metadata = bmi3d.load_ecube_metadata(test_filepath, 'Headstages')
        data = np.zeros((0,1))
        for chunk in  bmi3d._process_channels(test_filepath, 'Headstages', [0], metadata['n_samples'], 'uint16'):
            data = np.concatenate((data, chunk), axis=0)
        assert data.shape[1] == 1
        assert data.shape[0] == 214032

    def test_load_ecube_metadata(self):
        metadata = bmi3d.load_ecube_metadata(test_filepath, 'Headstages')
        assert metadata['samplerate'] == 25000.
        assert metadata['n_channels'] == 64

    def test_load_ecube_data(self):
        # Make a 'ground truth' figure from simulated data
        sim_filename = 'Headstages_8_Channels_int16_2021-05-06_11-47-02.bin'
        filepath = os.path.join(sim_filepath, sim_filename)
        with open(filepath, 'rb') as f:
            f.seek(8) # first uint64 is timestamp
            databuf = bytes(f.read())
            flatarray = np.frombuffer(databuf, dtype='<i2')
            shapedarray = flatarray.reshape(-1, 8).swapaxes(0,1)
        data = shapedarray.T
        self.assertEqual(data.shape[1], 8)
        self.assertEqual(data.shape[0], 25000)
        figname = 'load_ecube_data_groundtruth.png'
        plt.figure()
        plot_timeseries(data, 25000)
        savefig(write_dir, figname)

        # Compare to using the load_ecube_data() function
        data = bmi3d.load_ecube_data(sim_filepath, 'Headstages')
        self.assertEqual(data.shape[1], 8)
        self.assertEqual(data.shape[0], 25000)
        plt.figure()
        plot_timeseries(data, 25000)
        savefig(write_dir, 'load_ecube_data.png')

        fig1 = os.path.join(write_dir, figname)
        fig2 = os.path.join(write_dir, 'load_ecube_data.png')
        str = compare_images(fig1, fig2, 0.001)
        self.assertIsNone(str)

        # Load real data
        data = bmi3d.load_ecube_data(test_filepath, 'Headstages')
        assert data.shape[1] == 64
        assert data.shape[0] == 214032

        # Test that channels work
        data = bmi3d.load_ecube_data(test_filepath, 'Headstages', channels=[4])
        assert data.shape[1] == 1
        assert data.shape[0] == 214032

    def test_load_ecube_data_chunked(self):
        # Load 738 samples at once
        gen = bmi3d.load_ecube_data_chunked(test_filepath, 'Headstages')
        data = next(gen)
        assert data.shape[1] == 64
        assert data.shape[0] == 728

        # Load the rest
        for chunk in gen:
            data = np.concatenate((data, chunk), axis=0)

        assert data.shape[1] == 64
        assert data.shape[0] == 214032

        # Test that channels work
        gen = bmi3d.load_ecube_data_chunked(test_filepath, 'Headstages', channels=[4])
        data = next(gen)
        assert data.shape[1] == 1
        assert data.shape[0] == 728

        # Make a figure to test that the data is intact
        data = np.zeros((0,8))
        for chunk in bmi3d.load_ecube_data_chunked(sim_filepath, 'Headstages'):
            data = np.concatenate((data, chunk), axis=0)
        self.assertEqual(data.shape[0], 25000)
        plt.figure()
        plot_timeseries(data, 25000)
        savefig(write_dir, 'load_ecube_data_chunked.png') # should be the same as 'load_ecube_data.png'

    def test_proc_ecube_data(self):
        import os
        import h5py
        hdf_filepath = os.path.join(write_dir, "preprocessed_ecube_data.hdf")
        if os.path.exists(hdf_filepath):
            os.remove(hdf_filepath)
        dset, metadata = bmi3d.proc_ecube_data(test_filepath, 'Headstages', hdf_filepath)
        assert os.path.exists(hdf_filepath)
        hdf = h5py.File(hdf_filepath, 'r')
        assert 'broadband_data' in hdf
        assert hdf['broadband_data'].shape[1] == 64
        assert hdf['broadband_data'].shape[0] == 214032
        assert metadata['samplerate'] == 25000

    def test_load_bmi3d_hdf_table(self):
        testfile = 'test20210330_12_te1254.hdf'
        data, metadata = bmi3d.load_bmi3d_hdf_table(data_dir, testfile, 'task')
        self.assertEqual(len(data), 534)
        self.assertEqual(len(metadata.keys()), 35)

        data, metadata = bmi3d.load_bmi3d_hdf_table(data_dir, testfile, 'sync_clock')
        self.assertEqual(len(data), 534)
        data, metadata = bmi3d.load_bmi3d_hdf_table(data_dir, testfile, 'sync_events')
        self.assertEqual(len(data), 6)

        self.assertRaises(ValueError, lambda: bmi3d.load_bmi3d_hdf_table(data_dir, testfile, 'nonexistent_table'))

    def test_get_neuropixel_digital_input_times(self):
        ecube_files = '2023-03-26_BMI3D_te8921'
        on_times,off_times = get_ecube_digital_input_times(data_dir, ecube_files, -1)
        self.assertTrue(all(np.diff(on_times)>0))
        self.assertTrue(all(off_times - on_times)>0)
        self.assertTrue(any(np.diff(on_times)>30))

class NeuropixelTest(unittest.TestCase):
    
    def test_load_neuropixel_data(self):
        record_dir = '2023-03-26_Neuropixel_te8921'
        # AP data
        data,metadata = load_neuropixel_data(data_dir, record_dir, 'ap', port_number=1)
        self.assertEqual(data.samples.shape[1], metadata['num_channels'])
        self.assertEqual(metadata['sample_rate'], 30000)
        self.assertEqual(metadata['xpos'].shape[0], metadata['ypos'].shape[0])
        # LFP data
        data,metadata = load_neuropixel_data(data_dir, record_dir, 'lfp', port_number=1)
        self.assertEqual(data.samples.shape[1], metadata['num_channels'])
        self.assertEqual(metadata['sample_rate'], 2500)
        self.assertEqual(metadata['xpos'].shape[0], metadata['ypos'].shape[0])
           
    def test_load_neuropixel_configuration(self):
        record_dir = '2023-03-26_Neuropixel_te8921'
        port_num = 1
        config = load_neuropixel_configuration(data_dir, record_dir, port_number=port_num)
        nch = config['channel'].shape[0]
        self.assertEqual(config['xpos'].shape[0], nch)
        self.assertEqual(config['xpos'].shape[0], config['ypos'].shape[0])
        self.assertEqual(config['port'], str(port_num))
        
    def test_load_neuropixel_event(self):
        record_dir = '2023-03-26_Neuropixel_te8921'
        # AP data
        events = load_neuropixel_event(data_dir, record_dir, 'ap', port_number=1)
        self.assertTrue(all(np.diff(events['sample_number'])>0)) # sample numbers should increaseb monotonically
        self.assertTrue(all(events['stream_name'] == b'ProbeA-AP'))
        # LFP data
        events = load_neuropixel_event(data_dir, record_dir, 'lfp', port_number=1)
        self.assertTrue(all(np.diff(events['sample_number'])>0)) # sample numbers should increaseb monotonically
        self.assertTrue(all(events['stream_name'] == b'ProbeA-LFP'))
           
    def test_get_neuropixel_digital_input_times(self):
        record_dir = '2023-03-26_Neuropixel_te8921'
        on_times,off_times = get_neuropixel_digital_input_times(data_dir, record_dir, 'ap', port_number=1)
        self.assertTrue(all(np.diff(on_times)>0)) # on_times should increaseb monotonically
        self.assertTrue(all(off_times - on_times)>0) # on_times precede off_times
        self.assertTrue(any(np.diff(on_times)>30)) # whether there is a 30s inteval between on_times
           
class HDFTests(unittest.TestCase):

    def test_save_hdf(self):
        import os
        import h5py
        testfile = 'save_hdf_test.hdf'
        testfile_comp = 'save_hdf_test_comp.hdf'
        testpath = os.path.join(write_dir, testfile)
        testpath_comp = os.path.join(write_dir, testfile_comp)
        if os.path.exists(testpath):
            os.remove(testpath)
        if os.path.exists(testpath_comp):
            os.remove(testpath_comp)
        data = {'test_data_array': np.arange(1000)}
        params = {'key1': 'value1', 'key2': 2}
        compression = 3
        save_hdf(write_dir, testfile, data_dict=data, data_group="/", append=False)
        save_hdf(write_dir, testfile, data_dict=data, data_group="/test_data", append=True)
        save_hdf(write_dir, testfile, data_dict=params, data_group="/test_metadata", append=True)
        save_hdf(write_dir, testfile_comp, data_dict=data, data_group="/", append=False, compression=compression)
        f = h5py.File(testpath, 'r')
        self.assertIn('test_data_array', f)
        self.assertIn('test_data', f)
        self.assertIn('test_metadata', f)
        f_comp = h5py.File(testpath_comp,'r')
        self.assertTrue(f_comp['/test_data_array'].compression == 'gzip')
        self.assertTrue(f_comp['/test_data_array'].compression_opts == compression)
        self.assertTrue(np.allclose(f_comp['/test_data_array'][()],data['test_data_array']))
        # add check to assert data is compressed in this record
        test_data = f['test_data']
        test_metadata = f['test_metadata']
        self.assertEqual(test_metadata['key1'][()], b'value1') # note that the hdf doesn't save unicode strings
        self.assertEqual(test_metadata['key2'][()], 2)
        self.assertTrue(np.allclose(test_data['test_data_array'], np.arange(1000)))
        self.assertRaises(FileExistsError, lambda: save_hdf(write_dir, testfile, data, "/", append=False))

    def test_get_hdf_dictionary(self):
        testfile = 'load_hdf_contents_test.hdf'
        testpath = os.path.join(write_dir, testfile)
        if os.path.exists(testpath):
            os.remove(testpath)
        data_dict = {'test_data': np.arange(1000)}
        save_hdf(write_dir, testfile, data_dict=data_dict, data_group="/", append=False)
        group_data_dict = {'group_data': np.arange(1000)}
        save_hdf(write_dir, testfile, data_dict=group_data_dict, data_group="/group1", append=True)
        result = get_hdf_dictionary(write_dir, testfile, show_tree=True)
        self.assertIn('test_data', result)
        self.assertIn('group1', result)
        self.assertIn('group_data', result['group1'])
        print(result)

    def test_load_hdf_data(self):
        testfile = 'load_hdf_test.hdf'
        testpath = os.path.join(write_dir, testfile)
        if os.path.exists(testpath):
            os.remove(testpath)
        data_dict = {'test_data': np.arange(1000)}
        save_hdf(write_dir, testfile, data_dict=data_dict, append=False)
        self.assertRaises(ValueError, lambda: load_hdf_data(write_dir, testfile, 'not_valid_data'))
        data = load_hdf_data(write_dir, testfile, 'test_data')
        self.assertEqual(len(data), len(data_dict['test_data']))
        self.assertTupleEqual(data.shape, data_dict['test_data'].shape)
        self.assertTrue(np.allclose(data, data_dict['test_data']))
        save_hdf(write_dir, testfile, data_dict=data_dict, data_group="/test_group", append=True)
        data = load_hdf_data(write_dir, testfile, 'test_data', "/test_group")
        self.assertTrue(np.allclose(data, data_dict['test_data']))

    def test_load_hdf_group(self):
        import os
        import h5py
        testfile = 'load_hdf_group_test.hdf'
        testpath = os.path.join(write_dir, testfile)
        if os.path.exists(testpath):
            os.remove(testpath)
        data_dict = {'test_data_1': np.arange(1000), 'test_data_2': np.arange(100)}
        params_dict = {'key1': 'value1', 'key2': 2, 'key3': 3.3}
        save_hdf(write_dir, testfile, data_dict=data_dict, append=False)
        save_hdf(write_dir, testfile, data_dict=params_dict, data_group="/params", append=True)
        self.assertRaises(ValueError, lambda: load_hdf_group(write_dir, testfile, 'not_valid_group'))
        everything = load_hdf_group(write_dir, testfile)
        params_only = load_hdf_group(write_dir, testfile, "/params")
        self.assertIn('test_data_1', everything)
        self.assertIn('test_data_2', everything)
        self.assertIn('params', everything)
        self.assertEqual(len(data_dict['test_data_1']), len(data_dict['test_data_1']))
        self.assertEqual(len(data_dict['test_data_2']), len(data_dict['test_data_2']))
        self.assertDictEqual(params_only, params_dict)

CENTER_TARGET_ON = 16
CURSOR_ENTER_CENTER_TARGET = 80
CURSOR_ENTER_PERIPHERAL_TARGET = list(range(81,89))
PERIPHERAL_TARGET_ON = list(range(17,25))
CENTER_TARGET_OFF = 32
REWARD = 48
DELAY_PENALTY = 66
TIMEOUT_PENALTY = 65
HOLD_PENALTY = 64
TRIAL_END = 239

class TestGetPreprocDataFuncs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        files = {}
        files['hdf'] = 'fake_ecube_data_bmi3d.hdf'
        files['ecube'] = 'fake ecube data'
        cls.subject = 'test'
        cls.te_id = 3498
        cls.date = '2021-12-13'
        preproc_dir = os.path.join(write_dir, cls.subject)
        preproc.proc_single(data_dir, files, preproc_dir, cls.subject, cls.te_id, cls.date, ['exp', 'eye', 'lfp'], overwrite=True)

        # For eye data we need to additionally make a 'calibrated_data' entry
        coeff = np.repeat([[0,0]],4,axis=1)
        eye_data, eye_metadata = load_preproc_eye_data(write_dir, cls.subject, cls.te_id, cls.date)
        eye_data['calibrated_data'] = get_calibrated_eye_data(eye_data['raw_data'], coeff)
        eye_data['coefficients'] = coeff
        eye_metadata['external_calibration'] = True
        preproc_file = get_preprocessed_filename(cls.subject, cls.te_id, cls.date, 'eye')
        save_hdf(preproc_dir, preproc_file, eye_data, "/eye_data", append=True)
        save_hdf(preproc_dir, preproc_file, eye_metadata, "/eye_metadata", append=True)

    def test_get_interp_kinematics(self):
        exp_data, exp_metadata = load_preproc_exp_data(write_dir, self.subject, self.te_id, self.date)
        cursor_interp = get_interp_kinematics(exp_data, datatype='cursor', samplerate=100)
        hand_interp = get_interp_kinematics(exp_data, datatype='hand', samplerate=100)

        self.assertEqual(cursor_interp.shape[1], 2)
        self.assertEqual(hand_interp.shape[1], 3)

        self.assertEqual(len(cursor_interp), len(hand_interp))

        plt.figure()
        visualization.plot_trajectories([cursor_interp], [-10, 10, -10, 10])
        filename = 'get_interp_cursor.png'
        visualization.savefig(docs_dir, filename)

        plt.figure()
        ax = plt.axes(projection='3d')
        visualization.plot_trajectories([hand_interp], [-10, 10, -10, 10, -10, 10])
        filename = 'get_interp_hand.png'
        visualization.savefig(docs_dir, filename)

    def test_get_kinematic_segments(self):

        # Plot cursor trajectories - expect 9 trials
        trial_start_codes = [CURSOR_ENTER_CENTER_TARGET]
        trial_end_codes = [REWARD, TRIAL_END]
        trajs, segs = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes)
        self.assertEqual(len(trajs), 9)
        self.assertEqual(trajs[1].shape[1], 2) # x z
        bounds = [-10, 10, -10, 10]
        plt.figure()
        visualization.plot_trajectories(trajs, bounds=bounds)
        figname = 'get_trial_aligned_trajectories.png'
        visualization.savefig(write_dir, figname)
        plt.close()

        # Plot eye trajectories - expect same 9 trials but no eye pos to plot
        trajs, segs = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, datatype='eye')
        self.assertEqual(len(trajs), 9)
        self.assertEqual(trajs[1].shape[1], 4) # two eyes x and y
        plt.figure()
        visualization.plot_trajectories(trajs[:2], bounds=bounds)
        figname = 'get_eye_trajectories.png'
        visualization.savefig(write_dir, figname) # expect zeros
        plt.close()

        # Plot hand trajectories - expect same 9 trials but hand kinematics.
        hand_trajs, segs = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, datatype='hand')
        self.assertEqual(len(hand_trajs), 9)
        self.assertEqual(hand_trajs[1].shape[1], 3)
        plt.figure()
        visualization.plot_trajectories(hand_trajs, bounds=bounds)
        figname = 'get_hand_trajectories.png' # since these were test data generated with a cursor, it should look the same as the cursor data.
        visualization.savefig(write_dir, figname)
        plt.close()

        # Try cursor velocity
        # Test normalized output
        vel, _ = get_velocity_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, norm=True)
        self.assertEqual(len(vel), 9)
        self.assertEqual(vel[1].ndim, 1)
        plt.figure()
        plt.plot(vel[1])
        figname = 'get_trial_velocities.png'
        visualization.savefig(write_dir, figname)
        plt.close()

        # Test component wise velocity output
        vel, _ = get_velocity_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, norm=False)
        self.assertEqual(len(vel), 9)
        self.assertEqual(vel[1].shape[1], 2)

        # Use a trial filter to only get rewarded trials
        trial_filter = lambda t: TRIAL_END not in t
        trajs, segs = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, trial_filter=trial_filter)
        self.assertEqual(len(trajs), 7)

        # Test the samplerate return option
        trajs, segs, samplerate = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, return_samplerate=True)
        self.assertEqual(samplerate, 1000)

        trajs, segs, samplerate = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, datatype='hand', return_samplerate=True)
        self.assertEqual(samplerate, 1000)

        trajs, segs, samplerate = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, datatype='eye', return_samplerate=True)
        self.assertEqual(samplerate, 1000)

    def test_get_lfp_segments(self):
        trial_start_codes = [CURSOR_ENTER_CENTER_TARGET]
        trial_end_codes = [REWARD, TRIAL_END]
        lfp_segs, segs = get_lfp_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes)
        self.assertEqual(len(lfp_segs), 9)
        self.assertEqual(lfp_segs[0].shape, (0, 8)) # fake lfp data has 8 channels and 0 samples

    def test_get_lfp_aligned(self):
        trial_start_codes = [CURSOR_ENTER_CENTER_TARGET]
        trial_end_codes = [REWARD, TRIAL_END]
        time_before = 0.1
        time_after = 0.4
        lfp_aligned = get_lfp_aligned(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, time_before, time_after)
        self.assertEqual(lfp_aligned.shape, (9, (time_before+time_after)*1000, 8))

    def test_get_target_locations(self):
        target_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        locs = get_target_locations(write_dir, self.subject, self.te_id, self.date, target_indices)
        self.assertEqual(locs.shape, (9, 3))
        self.assertEqual(len(str(locs[1][0])), 6)

        # If you supply an invalid target index it should raise an error
        target_indices = np.array([10])
        self.assertRaises(ValueError, lambda: get_target_locations(write_dir, self.subject, self.te_id, self.date, target_indices))

    def test_get_source_files(self):
        subject = 'beignet'
        te_id = 5974
        date = '2022-07-01'
        preproc_dir = data_dir

        files, raw_data_dir = get_source_files(preproc_dir, subject, te_id, date)
        self.assertEqual(files['hdf'], 'hdf/beig20220701_04_te5974.hdf')
        self.assertEqual(files['ecube'], 'ecube/2022-07-01_BMI3D_te5974')
        self.assertEqual(raw_data_dir, '/data/raw')

    def test_tabulate_behavior_data(self):

        subjects = [self.subject, self.subject]
        ids = [self.te_id, self.te_id]
        dates = [self.date, self.date]
        trial_start_codes = [80]
        trial_end_codes = [239]
        target_codes = range(81,89)
        reward_codes = [48]
        penalty_codes = [64]
        df = tabulate_behavior_data(
            write_dir, subjects, ids, dates, trial_start_codes, trial_end_codes,
            target_codes, reward_codes, penalty_codes, df=None, 
            include_handdata=False, include_eyedata=False)
        self.assertEqual(len(df), 18)
        self.assertTrue(np.all(df['target_idx'] < 9))
        expected_reward = np.ones(len(df))
        expected_reward[-2:] = 0
        expected_reward[-11:-9] = 0
        np.testing.assert_allclose(df['reward'], expected_reward)
        
        plt.figure()
        bounds = [-10, 10, -10, 10]
        visualization.plot_trajectories(df['cursor_traj'].to_numpy(), bounds=bounds)
        figname = 'concat_trials.png' # should look very similar to get_trial_aligned_trajectories.png
        visualization.savefig(write_dir, figname)

        df = tabulate_behavior_data(
            write_dir, subjects, ids, dates, trial_start_codes, trial_end_codes,
            target_codes, reward_codes, penalty_codes, df=None, 
            include_handdata=True, include_eyedata=True)
        self.assertEqual(len(df), 18)
        self.assertEqual(df['cursor_traj'].iloc[0].shape[1], 2) # should have 2 cursor axes     
        self.assertEqual(df['hand_traj'].iloc[0].shape[1], 3) # should have 3 hand axes 
        self.assertEqual(df['eye_traj'].iloc[0].shape[1], 4) # should have 4 eye axes

    def test_tabulate_behavior_data_center_out(self):

        subjects = [self.subject, self.subject]
        ids = [self.te_id, self.te_id]
        dates = [self.date, self.date]

        df = tabulate_behavior_data_center_out(write_dir, subjects, ids, dates, df=None, 
                                               include_center_target=True,
                                               include_handdata=False, include_eyedata=False)
        self.assertEqual(len(df), 18) # should be the same df as above

class TestMatlab(unittest.TestCase):
    
    def test_load_matlab_cell_strings(self):
        testfile = 'matlab_cell_str.mat'
        strings = load_matlab_cell_strings(data_dir, testfile, 'bmiSessions')
        expected_strings = ['jeev070412j', 'jeev070512g', 'jeev070612d', 'jeev070712e', 'jeev070812d']
        self.assertListEqual(strings[:5], expected_strings)

    def test_parse_str_list(self):
        str_list = ['sig001i_wf', 'sig001i_wf_ts', 'sig002a_wf', 'sig002a_wf_ts', 'sig002b_wf', 'sig002b_wf_ts', 'sig002i_wf', 'sig002i_wf_ts']
        
        # Check case where both str_include and str_avoid are used
        parsed_strs1 = parse_str_list(str_list, str_include=['sig002', 'wf'], str_avoid=['b_wf', 'i_wf'])
        expected_parsed_strs1 = ['sig002a_wf', 'sig002a_wf_ts']
        self.assertListEqual(parsed_strs1, expected_parsed_strs1)
        
        # Check case where only str_include is used
        parsed_strs2 = parse_str_list(str_list, str_include=['sig001'])
        expected_parsed_strs2 = ['sig001i_wf', 'sig001i_wf_ts']
        self.assertListEqual(parsed_strs2, expected_parsed_strs2)
        
        # Check case where only str_avoid is used
        parsed_strs3 = parse_str_list(str_list, str_avoid=['sig002'])
        expected_parsed_strs3 = ['sig001i_wf', 'sig001i_wf_ts']
        self.assertListEqual(parsed_strs3, expected_parsed_strs3)

        # Check case where neither str_include or str_avoid are used
        parsed_strs4 = parse_str_list(str_list)
        self.assertListEqual(parsed_strs4, str_list)


class TestPickle(unittest.TestCase):

    def test_pkl_fn(self):
        test_dir = os.path.dirname(__file__)
        tmp_dir = os.path.join(test_dir, 'tmp')

         # Testing pkl_write
        val = np.random.rand(10,10)
        pkl_write('pickle_write_test.dat', val, tmp_dir)

        # Testing pkl_read
        dat_1 = pkl_read('pickle_write_test.dat', tmp_dir)

        self.assertEqual(np.shape(val), np.shape(dat_1))

class TestYaml(unittest.TestCase):

    def test_yaml_fn(self):
        test_dir = os.path.dirname(__file__)
        tmp_dir = os.path.join(test_dir, 'tmp')
        params_file = os.path.join(tmp_dir, 'task_codes.yaml')

         # Testing yaml_write
        params = [{'CENTER_TARGET_ON': 16,
                   'CURSOR_ENTER_CENTER_TARGET': 80,
                   'CURSOR_ENTER_PERIPHERAL_TARGET': list(range(81, 89)),
                   'PERIPHERAL_TARGET_ON': list(range(17, 25)),
                   'CENTER_TARGET_OFF': 32,
                   'REWARD': 48,
                   'DELAY_PENALTY': 66,
                   'TIMEOUT_PENALTY': 65,
                   'HOLD_PENALTY': 64,
                   'PAUSE': 254,
                   'TIME_ZERO': 238,
                   'TRIAL_END': 239}]
        yaml_write(params_file, params)

        # Testing pkl_read
        task_codes = yaml_read(params_file)

        self.assertEqual(params,task_codes)

class SignalPathTests(unittest.TestCase):

    def test_lookup_excel_value(self):
        testfile = '210118_ecog_channel_map.xls'
        self.assertEqual(lookup_excel_value(data_dir, testfile, 'acq', 'electrode', 119), 1)
        self.assertEqual(lookup_excel_value(data_dir, testfile, 'acq', 'zif61GroupID', 119), '_A')

    def test_lookup_acq2elec(self):
        testfile = '210118_ecog_channel_map.xls'
        self.assertEqual(lookup_acq2elec(data_dir, testfile, 118), 0)
        self.assertEqual(lookup_acq2elec(data_dir, testfile, 63), -1)
        self.assertEqual(lookup_acq2elec(data_dir, testfile, 119, zero_index=False), 1)
        self.assertEqual(lookup_acq2elec(data_dir, testfile, 64, zero_index=False), 0)

    def test_lookup_elec2acq(self):
        testfile = '210118_ecog_channel_map.xls'
        self.assertEqual(lookup_elec2acq(data_dir, testfile, 0), 118)
        self.assertEqual(lookup_elec2acq(data_dir, testfile, 321), -1)
        self.assertEqual(lookup_elec2acq(data_dir, testfile, 1, zero_index=False), 119)
        self.assertEqual(lookup_elec2acq(data_dir, testfile, 320, zero_index=False), 0)

    def test_load_electrode_pos(self):
        testfile = '244ch_viventi_ecog_elec_to_pos.xls'
        x, y = load_electrode_pos(data_dir, testfile)
        self.assertEqual(len(x), 244)
        self.assertEqual(len(y), 244)

    def test_map_acq2pos(self):
        # Note, this also tests map_acq2elec
        test_signalpathfile = '210910_ecog_signal_path.xlsx'
        test_layoutfile = '244ch_viventi_ecog_elec_to_pos.xls'
        test_signalpath_table = pd.read_excel(os.path.join(data_dir, test_signalpathfile))
        test_eleclayout_table = pd.read_excel(os.path.join(data_dir, test_layoutfile))

        acq_ch_position, acq_chs, connected_elecs = map_acq2pos(test_signalpath_table, test_eleclayout_table, xpos_name='topdown_x', ypos_name='topdown_y')
        
        np.testing.assert_array_equal(test_signalpath_table['acq'].to_numpy()[:240], acq_chs)
        np.testing.assert_array_equal(test_signalpath_table['electrode'].to_numpy()[:240], connected_elecs)
        
        # Manually test a few electrode positions and output array shape
        self.assertEqual(acq_ch_position.shape[0], 240)
        self.assertEqual(acq_ch_position.shape[1], 2)

        self.assertEqual(2.25, acq_ch_position[0,0])
        self.assertEqual(9, acq_ch_position[0,1])

        self.assertEqual(7.5, acq_ch_position[100,0])
        self.assertEqual(5.25, acq_ch_position[100,1])
        
        self.assertEqual(3.75, acq_ch_position[200,0])
        self.assertEqual(4.5, acq_ch_position[200,1])

    def test_load_chmap(self):
        test_signalpathfile = '221021_opto_signal_path.xlsx'
        test_layoutfile = '32ch_fiber_optic_assy_elec_to_pos.xlsx'
        test_signalpath_table = pd.read_excel(os.path.join(data_dir, test_signalpathfile))

        acq_ch_position, acq_chs, connected_elecs = load_chmap(drive_type='Opto32')
        
        np.testing.assert_array_equal(test_signalpath_table['acq'].to_numpy()[:32], acq_chs)
        np.testing.assert_array_equal(test_signalpath_table['electrode'].to_numpy()[:32], connected_elecs)
        self.assertEqual(acq_ch_position.shape[0], 32)
        self.assertEqual(acq_ch_position.shape[1], 2)

        test_signalpathfile = '210910_ecog_signal_path.xlsx'
        test_layoutfile = '244ch_viventi_ecog_elec_to_pos.xls'
        test_signalpath_table = pd.read_excel(os.path.join(data_dir, test_signalpathfile))

        acq_ch_position, acq_chs, connected_elecs = load_chmap(drive_type='ECoG244')
        
        np.testing.assert_array_equal(test_signalpath_table['acq'].to_numpy()[:240], acq_chs)
        np.testing.assert_array_equal(test_signalpath_table['electrode'].to_numpy()[:240], connected_elecs)
        self.assertEqual(acq_ch_position.shape[0], 240)
        self.assertEqual(acq_ch_position.shape[1], 2)

    def test_map_data2elec(self):
        test_signalpathfile = '210910_ecog_signal_path.xlsx'
        test_signalpath_table = pd.read_excel(os.path.join(data_dir, test_signalpathfile))
        datain = np.zeros((10, 256))
        for i in range(256):
            datain[:,i] = i+1

        dataout, acq_chs, connected_elecs = map_data2elec(datain, test_signalpath_table)

        self.assertEqual(dataout.shape[0], 10)
        self.assertEqual(dataout.shape[1], 240)
        np.testing.assert_allclose(dataout[0,:].flatten(), acq_chs)

        # Check zero_indexing flag
        datain = datain - 1
        test_signalpath_table['acq'] = test_signalpath_table['acq'] - 1
        dataout, acq_chs, connected_elecs = map_data2elec(datain, test_signalpath_table, zero_indexing=True)
        np.testing.assert_allclose(dataout[0,:].flatten(), acq_chs)

    def test_map_data2elecandpos(self):
        test_signalpathfile = '210910_ecog_signal_path.xlsx'
        test_layoutfile = '244ch_viventi_ecog_elec_to_pos.xls'
        test_signalpath_table = pd.read_excel(os.path.join(data_dir, test_signalpathfile))
        test_eleclayout_table = pd.read_excel(os.path.join(data_dir, test_layoutfile))
        datain = np.zeros((10, 256))
        for i in range(256):
            datain[:,i] = i+1

        dataout, acq_ch_position, acq_chs, connected_elecs = map_data2elecandpos(datain, test_signalpath_table, test_eleclayout_table,  xpos_name='topdown_x', ypos_name='topdown_y')

        self.assertEqual(dataout.shape[0], 10)
        self.assertEqual(dataout.shape[1], 240)
        np.testing.assert_allclose(dataout[0,:].flatten(), acq_chs)

        # Test acquisition channel subset selection
        acq_ch_subset = np.array([1,3,5,8,10])
        expected_acq_ch_pos = np.array([[2.25, 9], [5.25, 6.75],[3.75, 9]])
        dataout, acq_ch_position, acq_chs, connected_elecs = map_data2elecandpos(datain, test_signalpath_table, test_eleclayout_table, acq_ch_subset=acq_ch_subset)
        np.testing.assert_allclose(dataout[0,:].flatten(), np.array([1,5,10]))
        np.testing.assert_allclose(acq_chs, np.array([1,5,10]))
        np.testing.assert_allclose(connected_elecs, np.array([54,52,42]))
        np.testing.assert_allclose(acq_ch_position, expected_acq_ch_pos)

        # Test zero_indexing flag
        datain = datain - 1
        test_signalpath_table['acq'] = test_signalpath_table['acq'] - 1
        dataout, acq_ch_position, acq_chs, connected_elecs = map_data2elecandpos(datain, test_signalpath_table, test_eleclayout_table, zero_indexing=True)
        np.testing.assert_allclose(dataout[0,:].flatten(), acq_chs)

    def test_map_acq2elec(self):
        test_signalpathfile = '210910_ecog_signal_path.xlsx'
        test_signalpath_table = pd.read_excel(os.path.join(data_dir, test_signalpathfile))
        elecs = np.array((1,100,150,200))
        expected_acq_chs = np.array((58, 87, 158, 244))

        acq_chs_subset = map_elec2acq(test_signalpath_table, elecs)
        np.testing.assert_allclose(expected_acq_chs, acq_chs_subset)

        # Test if electrodes requested are unconnected
        elecs = np.array((1,100,33,155))
        expected_acq_chs = np.array((58, 87, np.nan, np.nan))

        acq_chs_subset = map_elec2acq(test_signalpath_table, elecs)
        np.testing.assert_allclose(expected_acq_chs, acq_chs_subset)

class E3vFrameTests(unittest.TestCase):

    def test_get_pulse_times(self):
        test_03         = [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]
        test_03_trigger = [0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0]
        test_03_data    = np.stack((test_03,test_03_trigger)).T
        test_03_times = np.array([2, 6, 10, 14])
        test_03_dc = np.array([0.5, 0.5, 0.5, 0.5])
        sync_ch_idx = 0
        trig_ch_idx = 1
        samplerate = 1
        pulse_times, pulse_dc = get_e3v_video_frame_data(test_03_data,sync_ch_idx,trig_ch_idx,samplerate)
        self.assertTrue(np.all(test_03_times == pulse_times))
        self.assertTrue(np.all(test_03_dc == pulse_dc))

class PesaranLabTests(unittest.TestCase):
        
    def test_read_lfp(self):
        pass

    def test_load_ecog_clfp_data(self):
        # this only works if every other peslab function works, with the exception of the lfp function tested above (eventually)
        test_ecog_ds250_data_file = os.path.join(data_dir,'peslab_test_data','recTEST.LM1_ECOG_3.clfp_ds250.dat')
        test_data, test_exp, test_mask = peslab.load_ecog_clfp_data(test_ecog_ds250_data_file)
        self.assertEqual(test_data.shape,(10000,62))

class EyeTests(unittest.TestCase):

    def test_apply_eye_calibration(self):

        # Create a test hdf file
        subject = 'test'
        te_id = 1
        date = 'calibration'
        data_source = 'eye'
        filename = get_preprocessed_filename(subject, te_id, date, data_source)
        filepath = os.path.join(write_dir, subject, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        data_dict = {
            'raw_data': np.array([[0,0], [1,1]])
        }
        metadata_dict = {}
        preproc_dir = os.path.join(write_dir, subject)
        if not os.path.exists(preproc_dir):
            os.mkdir(preproc_dir)
        save_hdf(preproc_dir, filename, data_dict, data_group='/eye_data')
        save_hdf(preproc_dir, filename, metadata_dict, '/eye_metadata', append=True)

        # Apply a calibration
        coeff = np.array([[3,4], [5,6]])
        apply_eye_calibration(coeff, write_dir, subject, te_id, date)

        # Check the result
        eye_data, eye_metadata = load_preproc_eye_data(write_dir, subject, te_id, date)
        self.assertIn('calibrated_data', eye_data)
        self.assertIn('coefficients', eye_data)
        self.assertIn('external_calibration', eye_metadata)    

if __name__ == "__main__":
    unittest.main()
