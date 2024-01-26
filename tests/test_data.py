# test_data.py 
# tests of aopy.data
from aopy import visualization
from aopy.data import *
from aopy.data import peslab
from aopy.data import optitrack
from aopy.data import bmi3d
from aopy.data import db
from aopy.postproc.base import get_calibrated_eye_data
from aopy.visualization import *
from aopy import preproc
import unittest
import os
import numpy as np
import pandas as pd
from matplotlib.testing.compare import compare_images
import datetime
import json

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
        record_dir = '2023-03-26_Neuropixel_beignet_te8921'
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
        record_dir = '2023-03-26_Neuropixel_beignet_te8921'
        port_num = 1
        config = load_neuropixel_configuration(data_dir, record_dir, port_number=port_num)
        nch = config['channel'].shape[0]
        self.assertEqual(config['xpos'].shape[0], nch)
        self.assertEqual(config['xpos'].shape[0], config['ypos'].shape[0])
        self.assertEqual(config['port'], str(port_num))
        
    def test_load_neuropixel_event(self):
        record_dir = '2023-03-26_Neuropixel_beignet_te8921'
        # AP data
        events = load_neuropixel_event(data_dir, record_dir, 'ap', port_number=1)
        self.assertTrue(all(np.diff(events['sample_number'])>0)) # sample numbers should increaseb monotonically
        self.assertTrue(all(events['stream_name'] == b'ProbeA-AP'))
        # LFP data
        events = load_neuropixel_event(data_dir, record_dir, 'lfp', port_number=1)
        self.assertTrue(all(np.diff(events['sample_number'])>0)) # sample numbers should increaseb monotonically
        self.assertTrue(all(events['stream_name'] == b'ProbeA-LFP'))
           
    def test_get_neuropixel_digital_input_times(self):
        record_dir = '2023-03-26_Neuropixel_beignet_te8921'
        on_times,off_times = get_neuropixel_digital_input_times(data_dir, record_dir, 'ap', port_number=1)
        self.assertTrue(all(np.diff(on_times)>0)) # on_times should increaseb monotonically
        self.assertTrue(all(off_times - on_times)>0) # on_times precede off_times
        self.assertTrue(any(np.diff(on_times)>30)) # whether there is a 30s inteval between on_times
        
    def test_load_ks_output(self):
        date = '2023-03-26'
        subject = 'beignet'
        kilosort_dir = os.path.join(data_dir, 'kilosort')
        concat_data_dir = f'{date}_Neuropixel_ks_{subject}_bottom_port1'
        ks_output = load_ks_output(kilosort_dir, concat_data_dir, flag='spike')
        self.assertTrue('spike_indices' in list(ks_output.keys()))
        self.assertTrue('spike_clusters' in list(ks_output.keys()))
    
    def test_chanel_bank_name(self):
        record_dir = '2023-03-26_Neuropixel_beignet_te8921'
        ch_config_dir = os.path.join(data_dir, 'channel_config_np')
        _,metadata = load_neuropixel_data(data_dir, record_dir, 'ap', port_number=1)
        ch_name = get_channel_bank_name(metadata['ch_bank'], ch_config_dir =ch_config_dir)
        self.assertEqual(ch_name, 'bottom')
                
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
        # Test with center out data
        exp_data, exp_metadata = load_preproc_exp_data(write_dir, self.subject, self.te_id, self.date)
        cursor_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='cursor', samplerate=100)
        hand_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='hand', samplerate=100)
        targets_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='targets', samplerate=100)

        self.assertEqual(cursor_interp.shape[1], 2)
        self.assertEqual(hand_interp.shape[1], 3)
        self.assertEqual(targets_interp.shape[1], 9) # 9 targets including center

        self.assertEqual(len(cursor_interp), len(hand_interp))

        plt.figure()
        visualization.plot_trajectories([cursor_interp], [-10, 10, -10, 10])
        filename = 'get_interp_cursor_centerout.png'
        visualization.savefig(docs_dir, filename)

        plt.figure()
        ax = plt.axes(projection='3d')
        visualization.plot_trajectories([hand_interp], [-10, 10, -10, 10, -10, 10])
        filename = 'get_interp_hand_centerout.png'
        visualization.savefig(docs_dir, filename)

        plt.figure()
        time = np.arange(len(targets_interp))/100
        plt.plot(time, targets_interp[:,:,0]) # plot just the x coordinate
        plt.xlim(10, 20)
        plt.xlabel('time (s)')
        plt.ylabel('x position (cm)')
        filename = 'get_interp_targets_centerout.png'
        visualization.savefig(docs_dir, filename)

        # Test with tracking task data (rig1)
        exp_data, exp_metadata = load_preproc_exp_data(data_dir, 'test', 8461, '2023-02-25')
        # check this is an experiment with reference & disturbance
        assert exp_metadata['trajectory_amplitude'] > 0
        assert exp_metadata['disturbance_amplitude'] > 0 & json.loads(exp_metadata['sequence_params'])['disturbance']

        cursor_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='cursor', samplerate=exp_metadata['fps']) # should equal user + dis
        ref_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='reference', samplerate=exp_metadata['fps'])
        dis_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='disturbance', samplerate=exp_metadata['fps']) # should be non-0s
        user_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='user', samplerate=exp_metadata['fps']) # should equal cursor - dis
        hand_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='hand', samplerate=exp_metadata['fps'])

        self.assertEqual(cursor_interp.shape[1], 2)
        self.assertEqual(ref_interp.shape[1], 2)
        self.assertEqual(dis_interp.shape[1], 2)
        self.assertEqual(user_interp.shape[1], 2)
        self.assertEqual(hand_interp.shape[1], 3)
        self.assertEqual(len(cursor_interp), len(ref_interp))
        self.assertEqual(len(ref_interp), len(dis_interp))
        self.assertAlmostEqual(sum(ref_interp[:,0]), 0)
        self.assertAlmostEqual(sum(dis_interp[:,0]), 0)

        n_sec = 120
        time = np.arange(exp_metadata['fps']*n_sec)/exp_metadata['fps']
        plt.figure()
        plt.plot(time, cursor_interp[:int(exp_metadata['fps']*n_sec),1], color='blueviolet', label='cursor')
        plt.plot(time, ref_interp[:int(exp_metadata['fps']*n_sec),1], color='darkorange', label='ref')
        plt.xlabel('time (s)')
        plt.ylabel('y position (cm)'); plt.ylim(-10,10)
        plt.legend()
        filename = 'get_interp_cursor_tracking.png'
        visualization.savefig(docs_dir, filename)

        plt.figure()
        plt.plot(time, user_interp[:int(exp_metadata['fps']*n_sec),1], color='darkturquoise', label='user')
        plt.plot(time, ref_interp[:int(exp_metadata['fps']*n_sec),1], color='darkorange', label='ref')
        plt.plot(time, dis_interp[:int(exp_metadata['fps']*n_sec),1], color='tab:red', linestyle='--', label='dis')
        plt.xlabel('time (s)')
        plt.ylabel('y position (cm)'); plt.ylim(-10,10)
        plt.legend()
        filename = 'get_interp_user_tracking.png'
        visualization.savefig(docs_dir, filename)
        
        # Test with tracking task data (tablet rig)
        exp_data, exp_metadata = load_preproc_exp_data(data_dir, 'churro', 375, '2023-10-02')
        # check this is an experiment with reference & NO disturbance
        assert exp_metadata['trajectory_amplitude'] > 0
        assert not json.loads(exp_metadata['sequence_params'])['disturbance']
        
        cursor_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='cursor', samplerate=exp_metadata['fps']) # should equal user
        ref_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='reference', samplerate=exp_metadata['fps'])
        dis_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='disturbance', samplerate=exp_metadata['fps']) # should be 0s
        user_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='user', samplerate=exp_metadata['fps']) # should equal cursor
        hand_interp = get_interp_kinematics(exp_data, exp_metadata, datatype='hand', samplerate=exp_metadata['fps']) # x dim (out of screen) should be 0s

        self.assertEqual(cursor_interp.shape[1], 2)
        self.assertEqual(ref_interp.shape[1], 2)
        self.assertEqual(dis_interp.shape[1], 2)
        self.assertEqual(user_interp.shape[1], 2)
        self.assertEqual(hand_interp.shape[1], 3)
        self.assertEqual(len(cursor_interp), len(ref_interp))
        self.assertEqual(len(ref_interp), len(dis_interp))
        self.assertAlmostEqual(sum(ref_interp[:,0]), 0)
        self.assertAlmostEqual(sum(dis_interp[:,0]), 0)

        plt.figure()
        plt.plot(time, cursor_interp[:int(exp_metadata['fps']*n_sec),1], color='blueviolet', label='cursor')
        plt.plot(time, ref_interp[:int(exp_metadata['fps']*n_sec),1], color='darkorange', label='ref')
        plt.xlabel('time (s)')
        plt.ylabel('y position (cm)'); plt.ylim(-10,10)
        plt.legend()
        filename = 'get_interp_cursor_tracking_tablet.png'
        visualization.savefig(docs_dir, filename)

        plt.figure()
        plt.plot(time, user_interp[:int(exp_metadata['fps']*n_sec),1], color='darkturquoise', label='user')
        plt.plot(time, ref_interp[:int(exp_metadata['fps']*n_sec),1], color='darkorange', label='ref')
        plt.plot(time, dis_interp[:int(exp_metadata['fps']*n_sec),1], color='tab:red', linestyle='--', label='dis')
        plt.xlabel('time (s)')
        plt.ylabel('y position (cm)'); plt.ylim(-10,10)
        plt.legend()
        filename = 'get_interp_user_tracking_tablet.png'
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
        self.assertEqual(lfp_aligned.shape, ((time_before+time_after)*1000, 8, 9))

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
        self.assertEqual(raw_data_dir, '/media/moor-data/raw')

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
            reward_codes, penalty_codes, metadata=['target_radius', 'rand_delay'], df=None)
        self.assertEqual(len(df), 18)
        np.testing.assert_allclose(df['target_radius'], 2.)
        for delay in df['rand_delay']:
             np.testing.assert_allclose(delay, [0.23, 0.3])
        expected_reward = np.ones(len(df))
        expected_reward[-2:] = 0
        expected_reward[-11:-9] = 0
        np.testing.assert_allclose(df['reward'], expected_reward)

    def test_tabulate_behavior_data_center_out(self):

        subjects = [self.subject, self.subject]
        ids = [self.te_id, self.te_id]
        dates = [self.date, self.date]

        df = tabulate_behavior_data_center_out(write_dir, subjects, ids, dates, df=None)
        self.assertEqual(len(df), 20) # 10 total trials, duplicated
        self.assertTrue(np.all(df['target_idx'] < 9))
        self.assertTrue(np.all(df['target_idx'] >= 0))
        self.assertTrue(np.all(df['target_idx'][df['reward']] > 0))
        for loc in df['target_location']:
            self.assertEqual(loc.shape[0], 3)
            self.assertLess(np.linalg.norm(loc), 7)

        # Check that reaches are completed
        self.assertTrue(np.all(df['hold_completed'][df['reward']]))
        self.assertTrue(np.all(df['delay_completed'][df['reward']]))
        self.assertTrue(np.all(df['reach_completed'][df['reward']]))

        # Check a couple interesting trials
        trial = df.iloc[0] # a successful trial
        self.assertTrue(trial['reward'])
        np.testing.assert_allclose(trial['event_codes'], [16, 80, 18, 32, 82, 48, 239])
        np.testing.assert_allclose(trial['target_location'], [0., 6.5, 0.])
        self.assertTrue(trial['trial_initiated'])
        self.assertTrue(trial['hold_completed'])
        self.assertTrue(trial['delay_completed'])
        self.assertTrue(trial['reach_completed'])
        events = [trial['trial_start_time'], trial['hold_start_time'], trial['delay_start_time'], 
                  trial['go_cue_time'], trial['reach_end_time'], trial['reward_start_time'], trial['reward_end_time']]
        np.testing.assert_allclose(events, sorted(events)) # events should occur in order

        trial = df.iloc[7] # a timeout penalty before anything happens
        self.assertFalse(trial['reward'])
        self.assertTrue(trial['penalty'])
        np.testing.assert_allclose(trial['event_codes'], [16, 65, 239])
        np.testing.assert_allclose(trial['target_location'], [0., 0., 0.])
        self.assertFalse(trial['trial_initiated'])
        self.assertFalse(trial['hold_completed'])
        self.assertFalse(trial['delay_completed'])
        self.assertFalse(trial['reach_completed'])
        self.assertTrue(~np.isnan(trial['penalty_start_time']))
        self.assertEqual(trial['penalty_event'], 65) # timeout penalty

        trial = df.iloc[8] # a hold penalty on the center target
        self.assertFalse(trial['reward'])
        self.assertTrue(trial['penalty'])
        np.testing.assert_allclose(trial['event_codes'], [16, 80, 64, 239])
        np.testing.assert_allclose(trial['target_location'], [0., 0., 0.])
        self.assertTrue(trial['trial_initiated'])
        self.assertFalse(trial['hold_completed'])
        self.assertFalse(trial['delay_completed'])
        self.assertFalse(trial['reach_completed'])
        self.assertTrue(~np.isnan(trial['penalty_start_time']))
        self.assertEqual(trial['penalty_event'], 64) # hold penalty

    def test_tabulate_behavior_data_out(self):

        subjects = [self.subject, self.subject]
        ids = [self.te_id, self.te_id]
        dates = [self.date, self.date]

        df = tabulate_behavior_data_out(write_dir, subjects, ids, dates, df=None)
        self.assertEqual(len(df), 16) # 8 total trials, duplicated (center target hold and timeout penalty trials are excluded)
        self.assertTrue(np.all(df['target_idx'] < 9))
        self.assertTrue(np.all(df['target_idx'] >= 0))
        self.assertTrue(np.all(df['target_idx'][df['reward']] > 0))
        for loc in df['target_location']:
            self.assertEqual(loc.shape[0], 3)
            self.assertLess(np.linalg.norm(loc), 7)

        # Check that reaches are completed
        self.assertTrue(np.all(df['reach_completed'][df['reward']]))

        # Check a couple interesting trials
        trial = df.iloc[0] # a successful trial
        self.assertTrue(trial['reward'])
        np.testing.assert_allclose(trial['event_codes'], [18, 32, 82, 48, 239])
        np.testing.assert_allclose(trial['target_location'], [0., 6.5, 0.])
        self.assertTrue(trial['reach_completed'])
        events = [trial['trial_start_time'], trial['reach_end_time'], trial['reward_start_time'], trial['reward_end_time']]
        np.testing.assert_allclose(events, sorted(events)) # events should occur in order

        trial = df.iloc[7] # a hold penalty on the peripheral target
        self.assertFalse(trial['reward'])
        self.assertTrue(trial['penalty'])
        np.testing.assert_allclose(trial['event_codes'], [21, 32, 85, 64, 239])
        np.testing.assert_allclose(trial['target_location'], [-4.5962, -4.5962, 0.])
        self.assertTrue(trial['reach_completed'])
        self.assertTrue(~np.isnan(trial['penalty_start_time']))
        self.assertEqual(trial['penalty_event'], 64) # hold penalty

    def test_tabulate_behavior_data_tracking_task(self):
        subjects = ['test', 'test']
        ids = [8461, 8461]
        dates = ['2023-02-25', '2023-02-25']
        df = tabulate_behavior_data_tracking_task(data_dir, subjects, ids, dates)  # no penalties in this session
        self.assertEqual(len(df), 42) # 21 total trials, duplicated
        self.assertTrue(np.all(df['reward']))
        self.assertFalse(np.all(df['penalty']))
        self.assertTrue(np.all(df['trial_initiated']))
        self.assertTrue(np.all(df['hold_completed']))

        # Check sequence params
        self.assertTrue(np.all([json.loads(params)['ramp']>0 for params in df['sequence_params']]))
        self.assertTrue(np.all([json.loads(params)['ramp_down']>0 for params in df['sequence_params']]))

        # Check that rewarded trials are complete
        self.assertTrue(np.all(df['trial_initiated'][df['reward']]))
        self.assertTrue(np.all(df['hold_completed'][df['reward']]))

        # Check that trial segments occur in the correct order
        trial_lengths, traj_lengths = [], []
        for i in range(len(df)):
            self.assertLess(df.loc[i,'hold_start_time'], df.loc[i,'tracking_start_time'])
            self.assertLess(df.loc[i,'tracking_start_time'], df.loc[i,'tracking_end_time'])
            self.assertLess(df.loc[i,'trajectory_start_time'], df.loc[i,'trajectory_end_time'])
            self.assertLess(df.loc[i,'tracking_start_time'], df.loc[i,'trajectory_start_time']) # ramp period
            self.assertLess(df.loc[i,'trajectory_end_time'], df.loc[i,'tracking_end_time'])
            trial_lengths.append(df.loc[i,'tracking_end_time'] - df.loc[i,'tracking_start_time'])
            traj_lengths.append(df.loc[i,'trajectory_end_time'] - df.loc[i,'trajectory_start_time'])

        # Check that trajectory timing doesn't include ramp periods
        plt.figure()
        plt.plot(trial_lengths, label='total tracking'); plt.plot(traj_lengths, label='trajectory (no ramps)')
        plt.xlabel('Reward trial #'); plt.ylabel('Time (sec)'); 
        plt.ylim(15,25); plt.legend()
        figname = 'tabulate_tracking_trial_segment_lengths_test.png'
        visualization.savefig(write_dir, figname)

        subjects = ['churro', 'churro']
        ids = [375, 375]
        dates = ['2023-10-02', '2023-10-02']
        df = tabulate_behavior_data_tracking_task(data_dir, subjects, ids, dates)
        self.assertEqual(len(df), 212)

        # Check sequence params
        self.assertTrue(np.all([json.loads(params)['ramp']==0 for params in df['sequence_params']]))
        self.assertTrue(np.all([json.loads(params)['ramp_down']==0 for params in df['sequence_params']]))

        # Check that rewarded trials are complete
        self.assertTrue(np.all(df['trial_initiated'][df['reward']]))
        self.assertTrue(np.all(df['hold_completed'][df['reward']]))

        # Check that trial segments occur in the correct order
        trial_lengths, traj_lengths = [], []
        for i in df[df['hold_completed']].index:
            self.assertLess(df.loc[i,'hold_start_time'], df.loc[i,'tracking_start_time'])
            self.assertLess(df.loc[i,'tracking_start_time'], df.loc[i,'tracking_end_time'])
            self.assertLess(df.loc[i,'trajectory_start_time'], df.loc[i,'trajectory_end_time'])
            self.assertEqual(df.loc[i,'tracking_start_time'], df.loc[i,'trajectory_start_time']) # no ramp period
            self.assertEqual(df.loc[i,'tracking_end_time'], df.loc[i,'trajectory_end_time'])
            if df.loc[i,'reward']:
                trial_lengths.append(df.loc[i,'tracking_end_time'] - df.loc[i,'tracking_start_time'])
                traj_lengths.append(df.loc[i,'trajectory_end_time'] - df.loc[i,'trajectory_start_time'])

        # Check that trajectory timing matches total tracking
        plt.figure()
        plt.plot(trial_lengths, label='total tracking'); plt.plot(traj_lengths, label='trajectory (no ramps)')
        plt.xlabel('Reward trial #'); plt.ylabel('Time (sec)')
        plt.ylim(15,25); plt.legend()
        figname = 'tabulate_tracking_trial_segment_lengths_churro.png'
        visualization.savefig(write_dir, figname)       

    def test_tabulate_kinematic_data(self):
        subjects = [self.subject, self.subject]
        ids = [self.te_id, self.te_id]
        dates = [self.date, self.date]

        df = tabulate_behavior_data_center_out(write_dir, subjects, ids, dates, df=None)
        
        # Only consider completed reaches
        df = df[df['reach_completed']]
        kin = tabulate_kinematic_data(write_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                            preproc=lambda x,fs : (x,fs), datatype='cursor', samplerate=1000)

        self.assertEqual(len(df), len(kin))

        plt.figure()
        bounds = [-10, 10, -10, 10]
        visualization.plot_trajectories(kin, bounds=bounds)
        figname = 'tabulate_kinematics.png' # should look very similar to get_trial_aligned_trajectories.png
        visualization.savefig(write_dir, figname)

    def test_tabulate_ts_data(self):

        subjects = [self.subject]
        ids = [self.te_id]
        dates = [self.date]

        df = tabulate_behavior_data_center_out(write_dir, subjects, ids, dates, df=None)

        # Only consider initiated trials
        df = df[df['trial_initiated']]

        trigger_times = df['hold_start_time']
        time_before = 0.5
        time_after = 0.5

        # Note: the data we're reading is only 1s long, so mostly these will be nans
        ts_data, samplerate = tabulate_ts_data(write_dir, df['subject'], df['te_id'], df['date'], 
                               trigger_times, time_before, time_after, datatype='lfp')
        
        trial_start_codes = [CURSOR_ENTER_CENTER_TARGET]
        trial_end_codes = [TRIAL_END]
        ts_data_single_file = get_lfp_aligned(write_dir, self.subject, self.te_id, self.date, 
                                              trial_start_codes, trial_end_codes, time_before, time_after)
     
        print(ts_data_single_file.shape)

        self.assertEqual(ts_data_single_file.shape, ts_data.shape)

    def test_tabulate_behavior_data_flash(self):
        files = {}
        files['hdf'] = 'test20220311_07_te4298.hdf'
        files['ecube'] = '2022-03-11_BMI3D_te4298'
        subject = 'test'
        te_id = 4298
        date = '2022-03-11'
        preproc_dir = os.path.join(write_dir, subject)
        preproc.proc_single(data_dir, files, preproc_dir, subject, te_id, date, ['exp'], overwrite=True)

        df = tabulate_behavior_data_flash(write_dir, [subject], [te_id], [date], df=None)
        self.assertEqual(len(df), 13) # 13 total trials

        # Check that flash times are in the correct order
        self.assertTrue(np.all(df['flash_end_time'] - df['flash_start_time'] > 0))

    def test_tabulate_stim_data(self):
        subjects = ['test']
        ids = [6577]
        dates = ['2022-08-19']
        df = tabulate_stim_data(data_dir, subjects, ids, dates, debug=True, df=None, laser_trigger='laser_trigger', 
            laser_sensor='laser_sensor') # note in this old file the laser_trigger is not called qwalor_trigger

        figname = 'tabulate_stim_data.png' # should be the same as laser_aligned_sensor_debug.png
        visualization.savefig(write_dir, figname)

        self.assertEqual(len(df), 51)
        for trial in range(len(df)):
            self.assertLessEqual(df['trial_width'][trial], 0.1)
            self.assertGreater(df['trial_power'][trial], 0.)
            self.assertLessEqual(df['trial_power'][trial], 1.0)
            self.assertGreater(df['trial_time'][trial], 0.)
            self.assertLessEqual(df['trial_time'][trial], 100.)
            self.assertGreater(df['trial_power_watts'][trial], 0.)
            self.assertEqual(df['peak_power_watts'][trial], 1.5)
            self.assertTrue(df['trial_found'][trial])

        self.assertEqual(np.sum(df['width_above_thr']), 0)
        self.assertEqual(np.sum(df['power_above_thr']), 4)

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
        params = {'CENTER_TARGET_ON': 16,
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
                   'TRIAL_END': 239,
                   'TRIAL_START': 2,
                   'CURSOR_ENTER_TARGET': 80,
                   'CURSOR_LEAVE_TARGET': 96,
                   'OTHER_PENALTY': 79}
        yaml_write(params_file, params)

        # Testing pkl_read
        task_codes = yaml_read(params_file)

        self.assertEqual(params,task_codes)

        task_codes_file = load_bmi3d_task_codes('task_codes.yaml')

        self.assertDictEqual(params, task_codes_file)

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

class DatabaseTests(unittest.TestCase):

    # Create some tests - only has to be run once and saved in db/tes
    @classmethod
    def setUpClass(cls):
        db.BMI3D_DBNAME = 'default'
        db.DB_TYPE = 'bmi3d'

        '''
        This database contains one subject and one experimenter:
        - Subject(name="test_subject")
        - Experimenter(name="experimenter_1")
        
        Tasks:
        - Task(name="manual control")
        - Task(name="tracking")

        Features:
        - Feature(name="feat_1")

        Systems:
        - System(name="test_system", path="", archive="")

        Decoders:
        - Decoder(name="test_decoder", entry_id=3) # the bmi control entry

        Generators:
        - Generator(name="test_gen")

        Sequences:
        - Sequence(name="test_seq", generator_name="test_gen", params='{"seq_param_1": 1}')
        
        Entries:
        - Tracking task entry from 2023-06-26
        - Manual control entry from 2023-06-26
            - project = "test project"
            - session = "test session"
            - entry_name = "task_desc" 
            - te.report = '{"runtime": 3.0, "n_trials": 2, "n_success_trials": 1}'
            - feats = [feat_1]
            - params = '{"task_param_1": 1}'
        - Flash entry (manual control task) from 2023-06-26
            - entry_name = "flash"
            - te.report = '{"runtime": 3.0, "n_trials": 2, "n_success_trials": 0}'
        - BMI entry (bmi control task) from 2023-06-26
            - params='{"bmi": 0}'
        ''' 

    def test_lookup_sessions(self):

        # Most basic lookup
        sessions = db.lookup_sessions(id=1)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].id, 1)
        sessions = db.lookup_sessions(id=[1,2])
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[1].id, 2)

        # Other sanity tests
        total_sessions = 4
        self.assertEqual(len(db.lookup_sessions()), total_sessions)
        self.assertEqual(len(db.lookup_mc_sessions()), 1)
        self.assertEqual(len(db.lookup_flash_sessions()), 1)
        self.assertEqual(len(db.lookup_tracking_sessions()), 1)
        self.assertEqual(len(db.lookup_bmi_sessions()), 1)

        # Test filtering
        self.assertEqual(len(db.lookup_sessions(subject="non_existent")), 0)
        self.assertEqual(len(db.lookup_sessions(subject="test_subject")), total_sessions)
        sessions = db.lookup_sessions(subject="test_subject", date="2023-06-26", task_name="manual control",
                                      task_desc="task_desc", session="test session", project="test project",
                                      experimenter="experimenter_1")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].task_name, "manual control")
        self.assertEqual(sessions[0].task_desc, "task_desc")
        self.assertEqual(sessions[0].subject, "test_subject")
        self.assertEqual(sessions[0].session, "test session")
        self.assertEqual(sessions[0].project, "test project")
        self.assertEqual(sessions[0].experimenter, "experimenter_1")
        self.assertEqual(str(sessions[0].date), "2023-06-26")

        # Special case - filter by id
        sessions = db.lookup_sessions(exclude_ids=[2,3])
        self.assertEqual(len(sessions), 2)

        # Special case - arbitrary filter fn
        sessions = db.lookup_sessions(filter_fn=lambda x:x.duration > 0)
        self.assertEqual(len(sessions), 2)

        # Check that changing the db name works
        db.DB_TYPE = 'unknown'
        self.assertEqual(len(db.lookup_sessions()), 0)
        db.DB_TYPE = 'bmi3d'
        db.BMI3D_DBNAME = 'rig2'
        self.assertRaises(Exception, db.lookup_sessions)
        db.BMI3D_DBNAME = 'default'

    def test_filter_functions(self):
        
        # Filter by features
        filter_fn = db.filter_has_features("feat_1")
        sessions = db.lookup_sessions(filter_fn=filter_fn)
        self.assertEqual(len(sessions), 1)
        filter_fn = db.filter_has_features(["feat_1"])
        sessions = db.lookup_sessions(filter_fn=filter_fn)
        self.assertEqual(len(sessions), 1)

        # Filter neural data
        filter_fn = db.filter_has_neural_data("ecog")
        sessions = db.lookup_sessions(filter_fn=filter_fn)
        self.assertEqual(len(sessions), 0)

    def test_BMI3DTaskEntry(self):

        # Test that all the fields work as they should
        te = db.lookup_sessions(task_desc='task_desc')[0]
        self.assertEqual(te.subject, 'test_subject')
        self.assertEqual(te.experimenter, 'experimenter_1')
        self.assertEqual(te.id, 2)
        self.assertEqual(str(te.date), "2023-06-26")
        self.assertEqual(type(te.datetime), datetime.datetime)
        self.assertEqual(te.session, 'test session')
        self.assertEqual(te.project, 'test project')
        self.assertEqual(te.task_name, 'manual control')
        self.assertEqual(te.task_desc, 'task_desc')
        self.assertEqual(te.notes, '')
        self.assertEqual(te.duration, 3.0)
        self.assertEqual(te.n_trials, 1)
        self.assertEqual(te.features[0], 'feat_1')
        decoder = te.get_decoder_record()
        self.assertEqual(decoder, None)
        self.assertCountEqual(te.task_params.keys(), ['task_param_1'])
        self.assertEqual(te.get_task_param('task_param_1'), 1)
        self.assertCountEqual(te.sequence_params.keys(), ['seq_param_1'])
        self.assertEqual(te.get_sequence_param('seq_param_1'), 1)
        self.assertCountEqual(te.get_preprocessed_sources(), ['exp', 'eye'])
        self.assertEqual(len(te.get_raw_files()), 0)
        raw = te.get_db_object()
        self.assertIsNotNone(raw)

        # Test a bmi session and decoder
        te = db.lookup_sessions(task_name="bmi control")[0]
        decoder = te.get_decoder_record()
        self.assertEqual(decoder.name, "test_decoder")
        self.assertRaises(Exception, te.get_decoder) # No decoder file present

    def test_list_entry_details(self):
        sessions = db.lookup_sessions(task_desc='task_desc')
        subject, te_id, date = db.list_entry_details(sessions)
        self.assertCountEqual(subject, ['test_subject'])
        self.assertCountEqual(te_id, [2])
        self.assertCountEqual([str(d) for d in date], ['2023-06-26'])
        
    def test_group_entries(self):

        sessions = db.lookup_sessions()
        grouped = db.group_entries(sessions) # by date
        self.assertEqual(len(grouped), 1)
        self.assertEqual(len(grouped[0]), 4)

        grouped = db.group_entries(sessions, lambda x: x.duration) # by duration
        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(grouped[0]), 2) # duration = 0.0
        self.assertEqual(len(grouped[1]), 2) # duration = 3.0


if __name__ == "__main__":
    unittest.main()
