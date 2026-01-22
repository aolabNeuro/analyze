# test_data.py 
# tests of aopy.data
import time
from aopy import visualization
from aopy.data import *
from aopy.data import peslab
from aopy.data import optitrack
from aopy.data import bmi3d
from aopy.data import db
from aopy.postproc.base import get_calibrated_eye_data
from aopy import visualization
from aopy import preproc
import unittest
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.testing.compare import compare_images
import datetime
import json
import pickle
from pathlib import Path

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
        files['hdf'] = 'beig20220701_04_te5974.hdf'
        files['ecube'] = '2022-07-01_BMI3D_te5974'

        cls.subject = 'fake_subject'
        cls.id = 5974
        cls.date = '2022-07-01'
        
        cls.id2 = '0000'
        cls.subject2 = 'test'
        cls.date2 = '2024-11-12'
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
        lfp_data, lfp_metadata = load_preproc_lfp_data(write_dir, self.subject, self.id, self.date, drive_number=1)
        self.assertIsInstance(lfp_data, np.ndarray)
        self.assertIsInstance(lfp_metadata, dict)
        
    def test_find_preproc_ids_from_day(self):
        ids = find_preproc_ids_from_day(write_dir, self.subject, self.date, 'exp')
        self.assertIn(self.id, ids)
        self.assertEqual(len(ids), 1)

    def test_proc_eye_day(self):
        #self.assertRaises(ValueError, lambda:proc_eye_day(write_dir, self.subject, self.date))
        best_id, te_ids = proc_eye_day(data_dir, 'test', '2022-08-19', correlation_min=0, dry_run=True)
        self.assertIsNone(best_id)
        self.assertCountEqual(te_ids, [6581, 6577])
    
    # Test for loading functions when data has multiple drive data
    def test_load_preproc_lfp_data_multidrive(self):
        with self.assertRaises(ValueError):
            lfp_data, lfp_metadata = load_preproc_lfp_data(data_dir, self.subject2, self.id2, self.date2, drive_number=None)
        
        lfp_data, lfp_metadata = load_preproc_lfp_data(data_dir, self.subject2, self.id2, self.date2, drive_number=1)
        self.assertIsInstance(lfp_data, np.ndarray)
        self.assertIsInstance(lfp_metadata, dict)
        lfp_data, lfp_metadata = load_preproc_lfp_data(data_dir, self.subject2, self.id2, self.date2, drive_number=2)        
        self.assertIsInstance(lfp_data, np.ndarray)
        self.assertIsInstance(lfp_metadata, dict)
        
    def test_load_preproc_ap_data_multidrive(self):
        with self.assertRaises(ValueError):
            ap_data, ap_metadata = load_preproc_ap_data(data_dir, self.subject2, self.id2, self.date2, drive_number=None)
            
        ap_data, ap_metadata = load_preproc_ap_data(data_dir, self.subject2, self.id2, self.date2, drive_number=1)
        self.assertIsInstance(ap_data, np.ndarray)
        self.assertIsInstance(ap_metadata, dict)
        ap_data, ap_metadata = load_preproc_ap_data(data_dir, self.subject2, self.id2, self.date2, drive_number=2)        
        self.assertIsInstance(ap_data, np.ndarray)
        self.assertIsInstance(ap_metadata, dict)
        
    def test_load_preproc_spike_data_multidrive(self):
        spike, metadata = load_preproc_spike_data(data_dir, self.subject2, self.id2, self.date2, drive_number=1)
        self.assertIsInstance(spike, dict)
        self.assertIsInstance(metadata, dict)
        spike, metadata = load_preproc_spike_data(data_dir, self.subject2, self.id2, self.date2, drive_number=2)        
        self.assertIsInstance(spike, dict)
        self.assertIsInstance(metadata, dict)
        
    def test_load_spike_waveforms_multidrive(self):
        wfs = load_spike_waveforms(data_dir, self.subject2, self.id2, self.date2, drive_number=1)
        self.assertIsInstance(wfs, dict)
        wfs = load_spike_waveforms(data_dir, self.subject2, self.id2, self.date2, drive_number=2)        
        self.assertIsInstance(wfs, dict)     
                
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
        visualization.plot_timeseries(data, 25000)
        visualization.savefig(write_dir, figname)

        # Compare to using the load_ecube_data() function
        data = bmi3d.load_ecube_data(sim_filepath, 'Headstages')
        self.assertEqual(data.shape[1], 8)
        self.assertEqual(data.shape[0], 25000)
        plt.figure()
        visualization.plot_timeseries(data, 25000)
        visualization.savefig(write_dir, 'load_ecube_data.png')

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
        visualization.plot_timeseries(data, 25000)
        visualization.savefig(write_dir, 'load_ecube_data_chunked.png') # should be the same as 'load_ecube_data.png'

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

    def test_load_emg_data(self):
        emg_data_dir = os.path.join(data_dir, 'quatt_emg')
        filename = 'aj20250319_05_te2540_emg.hdf'
        
        emg_data, emg_metadata = load_emg_data(emg_data_dir, filename)
        self.assertEqual(emg_data.shape[1], 64)
        self.assertEqual(emg_data.shape[0], 14720)

        self.assertIn('samplerate', emg_metadata)
        self.assertEqual(emg_metadata['samplerate'], 2048)
        self.assertEqual(emg_metadata['n_channels'], 64)

    def test_load_emg_analog(self):
        emg_data_dir = os.path.join(data_dir, 'quatt_emg')
        filename = 'aj20250319_05_te2540_emg.hdf'
        
        analog_data, analog_metadata = load_emg_analog(emg_data_dir, filename)
        self.assertEqual(analog_data.shape[1], 16)
        self.assertEqual(analog_data.shape[0], 14720)

        self.assertIn('samplerate', analog_metadata)
        self.assertEqual(analog_metadata['samplerate'], 2048)
        self.assertEqual(analog_metadata['n_channels'], 16)

    def test_load_emg_digital(self):
        emg_data_dir = os.path.join(data_dir, 'quatt_emg')
        filename = 'aj20250319_05_te2540_emg.hdf'
        
        digital_data, digital_metadata = load_emg_digital(emg_data_dir, filename)
        self.assertEqual(digital_data.shape[0], 14720)
        self.assertEqual(digital_data.ndim, 1)

        self.assertIn('samplerate', digital_metadata)
        self.assertEqual(digital_metadata['samplerate'], 2048)
        self.assertEqual(digital_metadata['n_channels'], 16)


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

    def test_list_root_groups(self):
        testfile = 'load_hdf_contents_test.hdf'
        group_data_dict = {'group_data': np.arange(1000)}
        save_hdf(write_dir, testfile, data_dict=group_data_dict, data_group="/group1", append=True)
        group_data_dict = {'group_data': np.arange(1000)}
        save_hdf(write_dir, testfile, data_dict=group_data_dict, data_group="/group2", append=True)
        group_names = list_root_groups(write_dir, testfile)
        self.assertIn('group1', group_names)
        self.assertIn('group2', group_names)
        
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
        files['hdf'] = 'beig20220701_04_te5974.hdf'
        files['ecube'] = '2022-07-01_BMI3D_te5974'

        # Reduce the file size so we can upload it to github
        # headstage_data, metadata = load_ecube_headstages(data_dir, files['ecube'])
        # headstage_data = headstage_data[:,:16] * metadata['voltsperbit'] # reduce to 16 channels
        # filename = utils.save_test_signal_ecube(headstage_data, data_dir, 1, datasource='Headstages')

        cls.subject = 'beignet'
        cls.te_id = 5974
        cls.date = '2022-07-01'
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

    def test_get_interp_task_data(self):
        # Test with center out data
        exp_data, exp_metadata = load_preproc_exp_data(write_dir, self.subject, self.te_id, self.date)
        cursor_interp = get_interp_task_data(exp_data, exp_metadata, datatype='cursor', samplerate=100)
        hand_interp = get_interp_task_data(exp_data, exp_metadata, datatype='hand', samplerate=100)
        targets_interp = get_interp_task_data(exp_data, exp_metadata, datatype='targets', samplerate=100)
        user_interp = get_interp_task_data(exp_data, exp_metadata, datatype='user_world', samplerate=100)
        screen_interp = get_interp_task_data(exp_data, exp_metadata, datatype='user_screen', samplerate=100)

        self.assertEqual(cursor_interp.shape[1], 3)
        self.assertEqual(hand_interp.shape[1], 3)
        self.assertEqual(targets_interp.shape[1], 9) # 9 targets including center
        self.assertEqual(user_interp.shape[1], 3)

        self.assertEqual(len(cursor_interp), len(hand_interp))

        plt.figure()
        visualization.plot_trajectories([cursor_interp], [-10, 10, -10, 10])
        filename = 'get_interp_cursor_centerout.png'
        visualization.savefig(docs_dir, filename, transparent=False)

        plt.figure()
        ax = plt.axes(projection='3d')
        visualization.plot_trajectories([hand_interp]) #, [-10, 10, -10, 10, -10, 10])
        filename = 'get_interp_hand_centerout.png'
        visualization.savefig(docs_dir, filename, transparent=False)

        plt.figure()
        ax = plt.axes(projection='3d')
        visualization.plot_trajectories([user_interp]) #, [-10, 10, -10, 10, -10, 10])
        filename = 'get_user_world.png'
        visualization.savefig(docs_dir, filename, transparent=False)

        plt.figure()
        ax = plt.axes(projection='3d')
        visualization.plot_trajectories([user_interp]) #, [-10, 10, -10, 10, -10, 10])
        filename = 'get_user_screen.png'
        visualization.savefig(docs_dir, filename, transparent=False)

        plt.figure()
        time = np.arange(len(targets_interp))/100
        plt.plot(time, targets_interp[:,:,0]) # plot just the x coordinate
        plt.xlim(10, 20)
        plt.xlabel('time (s)')
        plt.ylabel('x position (cm)')
        filename = 'get_interp_targets_centerout.png'
        visualization.savefig(docs_dir, filename, transparent=False)

        # Test with tracking task data (rig1)
        exp_data, exp_metadata = load_preproc_exp_data(data_dir, 'test', 8461, '2023-02-25')
        # check this is an experiment with reference & disturbance
        assert exp_metadata['trajectory_amplitude'] > 0
        assert exp_metadata['disturbance_amplitude'] > 0 & json.loads(exp_metadata['sequence_params'])['disturbance']

        cursor_interp = get_interp_task_data(exp_data, exp_metadata, datatype='cursor', samplerate=exp_metadata['fps']) # should equal user + dis
        ref_interp = get_interp_task_data(exp_data, exp_metadata, datatype='reference', samplerate=exp_metadata['fps'])
        dis_interp = get_interp_task_data(exp_data, exp_metadata, datatype='disturbance', samplerate=exp_metadata['fps']) # should be non-0s
        user_interp = get_interp_task_data(exp_data, exp_metadata, datatype='user', samplerate=exp_metadata['fps']) # should equal cursor - dis
        hand_interp = get_interp_task_data(exp_data, exp_metadata, datatype='manual_input', samplerate=exp_metadata['fps'])

        self.assertEqual(cursor_interp.shape[1], 3)
        self.assertEqual(ref_interp.shape[1], 3)
        self.assertEqual(dis_interp.shape[1], 3)
        self.assertEqual(user_interp.shape[1], 3)
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
        visualization.savefig(docs_dir, filename, transparent=False)

        plt.figure()
        plt.plot(time, user_interp[:int(exp_metadata['fps']*n_sec),1], color='darkturquoise', label='user')
        plt.plot(time, ref_interp[:int(exp_metadata['fps']*n_sec),1], color='darkorange', label='ref')
        plt.plot(time, dis_interp[:int(exp_metadata['fps']*n_sec),1], color='tab:red', linestyle='--', label='dis')
        plt.xlabel('time (s)')
        plt.ylabel('y position (cm)'); plt.ylim(-10,10)
        plt.legend()
        filename = 'get_interp_user_tracking.png'
        visualization.savefig(docs_dir, filename, transparent=False)
        
        # Test with tracking task data (tablet rig)
        exp_data, exp_metadata = load_preproc_exp_data(data_dir, 'churro', 375, '2023-10-02')
        # check this is an experiment with reference & NO disturbance
        assert exp_metadata['trajectory_amplitude'] > 0
        assert not json.loads(exp_metadata['sequence_params'])['disturbance']
        
        cursor_interp = get_interp_task_data(exp_data, exp_metadata, datatype='cursor', samplerate=exp_metadata['fps']) # should equal user
        ref_interp = get_interp_task_data(exp_data, exp_metadata, datatype='reference', samplerate=exp_metadata['fps'])
        dis_interp = get_interp_task_data(exp_data, exp_metadata, datatype='disturbance', samplerate=exp_metadata['fps']) # should be 0s
        user_interp = get_interp_task_data(exp_data, exp_metadata, datatype='user', samplerate=exp_metadata['fps']) # should equal cursor
        hand_interp = get_interp_task_data(exp_data, exp_metadata, datatype='manual_input', samplerate=exp_metadata['fps']) # x dim (out of screen) should be 0s

        self.assertEqual(cursor_interp.shape[1], 3)
        self.assertEqual(ref_interp.shape[1], 3)
        self.assertEqual(dis_interp.shape[1], 3)
        self.assertEqual(user_interp.shape[1], 3)
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
        visualization.savefig(docs_dir, filename, transparent=False)

        plt.figure()
        plt.plot(time, user_interp[:int(exp_metadata['fps']*n_sec),1], color='darkturquoise', label='user')
        plt.plot(time, ref_interp[:int(exp_metadata['fps']*n_sec),1], color='darkorange', label='ref')
        plt.plot(time, dis_interp[:int(exp_metadata['fps']*n_sec),1], color='tab:red', linestyle='--', label='dis')
        plt.xlabel('time (s)')
        plt.ylabel('y position (cm)'); plt.ylim(-10,10)
        plt.legend()
        filename = 'get_interp_user_tracking_tablet.png'
        visualization.savefig(docs_dir, filename, transparent=False)

    def test_get_task_data(self):

        # Plot cycle count
        ts_data, samplerate = get_task_data(write_dir, self.subject, self.te_id, self.date, 'cycle')
        self.assertEqual(len(ts_data), 7985)
        self.assertEqual(samplerate, 120)
        time = np.arange(len(ts_data))/samplerate
        plt.figure()
        plt.plot(time[1:], 1/np.diff(ts_data), 'ko')
        plt.xlabel('time (s)')
        plt.ylabel('cycle step')
        plt.ylim(0, 2)
        figname = 'get_cycle_data.png'
        visualization.savefig(docs_dir, figname, transparent=False)
        plt.close()

    def test_extract_lfp_features(self):
        with open(os.path.join(data_dir, 'test_decoder.pkl'), 'rb') as file:
            decoder = pickle.load(file)

        # Test with no LFP data
        def test_fn():
            extract_lfp_features(write_dir, self.subject, self.te_id, self.date, 
                                             decoder)
        self.assertRaises(ValueError, test_fn)

        # Test with LFP data
        subject = 'affi'
        te_id = 17269
        date = '2024-05-03'
        preproc_dir = data_dir
        start_time = 10
        end_time = 30

        # Reduce the size of the LFP data
        # lfp_data, lfp_metadata = load_preproc_lfp_data(data_dir, subject, te_id, date)
        # lfp_data = lfp_data[:1000*30] # only keep first 30 seconds
        # os.remove(os.path.join(data_dir, 'affi/preproc_2024-05-03_affi_17269_lfp.hdf'))
        # print('lfp data', lfp_data.nbytes)
        # save_hdf(data_dir, 'affi/preproc_2024-05-03_affi_17269_lfp.hdf', {'lfp_data': lfp_data})
        # save_hdf(data_dir, 'affi/preproc_2024-05-03_affi_17269_lfp.hdf', lfp_metadata, "/lfp_metadata", append=True)

        features_offline, samplerate_offline = extract_lfp_features(
            preproc_dir, subject, te_id, date, decoder, 
            start_time=start_time, end_time=end_time)

        features_online, samplerate_online = get_extracted_features(
            preproc_dir, subject, te_id, date, decoder,
            start_time=start_time, end_time=end_time)

        time_offline = np.arange(len(features_offline))/samplerate_offline + start_time
        time_online = np.arange(len(features_online))/samplerate_online + start_time

        plt.figure(figsize=(8,3))
        plt.plot(time_offline, features_offline[:,1], alpha=0.8, label='offline')
        plt.plot(time_online, features_online[:,1], alpha=0.8, label='online')
        plt.xlabel('time (s)')
        plt.ylabel('power')
        plt.legend()
        plt.title('readout 1')
        
        filename = 'extract_decoder_features.png'
        visualization.savefig(docs_dir, filename, transparent=False)

    def test_get_kinematic_segments(self):

        # Plot cursor trajectories - expect 9 trials
        trial_start_codes = [CURSOR_ENTER_CENTER_TARGET]
        trial_end_codes = [REWARD, TRIAL_END]
        trajs, segs = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes)
        self.assertEqual(len(trajs), 13)
        self.assertEqual(trajs[1].shape[1], 3)
        bounds = [-10, 10, -10, 10]
        plt.figure()
        visualization.plot_trajectories(trajs, bounds=bounds)
        figname = 'get_trial_aligned_trajectories.png'
        visualization.savefig(write_dir, figname)
        plt.close()

        # Plot eye trajectories - expect same 9 trials but no eye pos to plot
        trajs, segs = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, datatype='eye')
        self.assertEqual(len(trajs), 13)
        self.assertEqual(trajs[1].shape[1], 4) # two eyes x and y
        plt.figure()
        visualization.plot_trajectories(trajs[:2], bounds=bounds)
        figname = 'get_eye_trajectories.png'
        visualization.savefig(write_dir, figname) # expect zeros
        plt.close()

        # Plot hand trajectories - expect same 9 trials but hand kinematics.
        hand_trajs, segs = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, datatype='manual_input')
        self.assertEqual(len(hand_trajs), 13)
        self.assertEqual(hand_trajs[1].shape[1], 3)
        plt.figure()
        visualization.plot_trajectories(hand_trajs, bounds=bounds)
        figname = 'get_hand_trajectories.png' # since these were test data generated with a cursor, it should look the same as the cursor data.
        visualization.savefig(write_dir, figname)
        plt.close()

        # Try cursor velocity
        # Test normalized output
        vel, _ = get_velocity_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, norm=True)
        self.assertEqual(len(vel), 13)
        self.assertEqual(vel[1].ndim, 1)
        plt.figure()
        plt.plot(vel[1])
        figname = 'get_trial_velocities.png'
        visualization.savefig(write_dir, figname)
        plt.close()

        # Test component wise velocity output
        vel, _ = get_velocity_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, norm=False)
        self.assertEqual(len(vel), 13)
        self.assertEqual(vel[1].shape[1], 3)

        # Use a trial filter to only get rewarded trials
        trial_filter = lambda t: TRIAL_END not in t
        trajs, segs = get_kinematic_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, trial_end_codes, trial_filter=trial_filter)
        self.assertEqual(len(trajs), 10)

    def test_get_lfp_segments(self):
        trial_start_codes = [CURSOR_ENTER_CENTER_TARGET]
        trial_end_codes = [REWARD, TRIAL_END]
        lfp_segs, segs = get_lfp_segments(write_dir, self.subject, self.te_id, self.date, 
                                          trial_start_codes, trial_end_codes, drive_number=1)
        self.assertEqual(len(lfp_segs), 13)
        self.assertEqual(lfp_segs[0].shape, (1395, 16)) # fake lfp data has 8 channels and 0 samples

    def test_get_lfp_aligned(self):
        trial_start_codes = [CURSOR_ENTER_CENTER_TARGET]
        trial_end_codes = [REWARD, TRIAL_END]
        time_before = 0.1
        time_after = 0.4
        lfp_aligned = get_lfp_aligned(write_dir, self.subject, self.te_id, self.date, 
                                      trial_start_codes, trial_end_codes, time_before, time_after, drive_number=1)
        self.assertEqual(lfp_aligned.shape, ((time_before+time_after)*1000, 16, 13))

    def test_get_target_locations(self):
        target_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        locs = get_target_locations(write_dir, self.subject, self.te_id, self.date, target_indices)
        self.assertEqual(locs.shape, (9, 3))
        self.assertEqual(len(str(locs[1][0])), 3)

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
        self.assertEqual(len(df), 26)
        np.testing.assert_allclose(df['target_radius'], 1.3)
        for delay in df['rand_delay']:
             np.testing.assert_allclose(delay, [0.1, 0.6])
        expected_reward = np.ones(len(df))
        expected_reward[[4,6,8,17,19,21]] = 0
        np.testing.assert_allclose(df['reward'], expected_reward)

    def test_tabulate_behavior_data_center_out(self):

        subjects = [self.subject, self.subject]
        ids = [self.te_id, self.te_id]
        dates = [self.date, self.date]

        t0 = time.perf_counter()
        df = tabulate_behavior_data_center_out(write_dir, subjects, ids, dates, df=None)
        t1 = time.perf_counter()
        print(f"tabulate_behavior_data_center_out took {t1-t0:0.3f} seconds")
        
        self.assertEqual(len(df), 26) # 10 total trials, duplicated
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
        np.testing.assert_allclose(trial['event_codes'], [16, 80, 24, 32, 88, 48, 239])
        np.testing.assert_allclose(trial['target_location'], [-4.5962, 4.5962, 0.])
        self.assertTrue(trial['trial_initiated'])
        self.assertTrue(trial['hold_completed'])
        self.assertTrue(trial['delay_completed'])
        self.assertTrue(trial['reach_completed'])
        events = [trial['prev_trial_end_time'], trial['center_target_on_time'], trial['hold_start_time'], trial['delay_start_time'], 
                  trial['go_cue_time'], trial['reach_end_time'], trial['reward_start_time'], trial['trial_end_time']]
        np.testing.assert_allclose(events, sorted(events)) # events should occur in order
        self.assertEqual(trial['prev_trial_end_time'], 0.)
        self.assertGreater(trial['trial_end_time'], trial['reward_start_time'])

        trial = df.iloc[7] # a timeout penalty before anything happens
        self.assertTrue(trial['reward'])
        self.assertFalse(trial['penalty'])
        np.testing.assert_allclose(trial['event_codes'], [16, 80, 22, 32, 86, 48, 239])
        np.testing.assert_allclose(trial['target_location'], [-4.5962, -4.5962, 0.])
        self.assertTrue(trial['trial_initiated'])
        self.assertTrue(trial['hold_completed'])
        self.assertTrue(trial['delay_completed'])
        self.assertTrue(trial['reach_completed'])
        self.assertTrue(np.isnan(trial['penalty_start_time']))
        self.assertTrue(np.isnan(trial['penalty_event']))# timeout penalty
        self.assertGreater(trial['prev_trial_end_time'], 0.)

        trial = df.iloc[8] # a hold penalty on the center target
        self.assertFalse(trial['reward'])
        self.assertTrue(trial['penalty'])
        np.testing.assert_allclose(trial['event_codes'], [16, 80, 20, 32, 84, 64, 239])
        np.testing.assert_allclose(trial['target_location'], [4.5962, -4.5962, 0.])
        self.assertTrue(trial['trial_initiated'])
        self.assertTrue(trial['hold_completed'])
        self.assertTrue(trial['delay_completed'])
        self.assertTrue(trial['reach_completed'])
        self.assertTrue(~np.isnan(trial['penalty_start_time']))
        np.testing.assert_allclose(trial['penalty_start_time'], 41.38588)
        self.assertEqual(trial['penalty_event'], 64) # hold penalty
        self.assertGreater(trial['prev_trial_end_time'], 0.)
        self.assertGreater(trial['trial_end_time'], trial['penalty_start_time'])

        trial = df.iloc[10] # first trial of the second session
        self.assertEqual(trial['prev_trial_end_time'], 49.4024)

    def test_tabulate_behavior_data_out(self):

        subjects = [self.subject, self.subject]
        ids = [self.te_id, self.te_id]
        dates = [self.date, self.date]

        df = tabulate_behavior_data_out(write_dir, subjects, ids, dates, df=None)
        self.assertEqual(len(df), 26) # 8 total trials, duplicated (center target hold and timeout penalty trials are excluded)
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
        np.testing.assert_allclose(trial['event_codes'], [24, 32, 88, 48, 239])
        np.testing.assert_allclose(trial['target_location'], [-4.5962, 4.5962, 0.])
        self.assertTrue(trial['reach_completed'])
        events = [trial['prev_trial_end_time'], trial['target_on_time'], trial['reach_end_time'], trial['reward_start_time'], trial['trial_end_time']]
        np.testing.assert_allclose(events, sorted(events)) # events should occur in order

        trial = df.iloc[7] # a hold penalty on the peripheral target
        self.assertTrue(trial['reward'])
        self.assertFalse(trial['penalty'])
        np.testing.assert_allclose(trial['event_codes'], [22, 32, 86, 48, 239])
        np.testing.assert_allclose(trial['target_location'], [-4.5962, -4.5962, 0.])
        self.assertTrue(trial['reach_completed'])
        self.assertFalse(~np.isnan(trial['penalty_start_time']))
        self.assertTrue(np.isnan(trial['penalty_event']))# hold penalty

    def test_tabulate_behavior_data_corners(self):
        task_codes = load_bmi3d_task_codes()
        subjects = ['test', 'test']
        ids = [19005, 19054]
        dates = ['2024-12-31', '2025-01-21'] # first entry is pre-pause state, second entry has pause state
        df = tabulate_behavior_data_corners(data_dir, subjects, ids, dates, metadata=['target_radius', 'cursor_radius', 'rand_delay'])
        self.assertEqual(len(df), 55)
        self.assertEqual(len(df.columns), 8+3+20) # no. columns in base tabulate func + no. of user-inputted metadata fields + no. columns in tabulate wrapper

        # Check chain length (sequence param)
        self.assertTrue(np.all(df['chain_length'] == 2))

        # Check that rewarded trials are complete
        self.assertTrue(np.all(df['trial_initiated'][df['reward']]))
        self.assertTrue(np.all(df['hold_completed'][df['reward']]))

        # Check that reach completed trials have two target indicies & locations
        self.assertTrue(np.all([len(idx)==2 for idx in df[df['reach_completed']].target_idx]))
        self.assertTrue(np.all([loc.shape==(2,3) for loc in df[df['reach_completed']].target_location]))

        # Check that hold completed trials have two target indicies & locations
        self.assertTrue(np.all([len(idx)==2 for idx in df[df['hold_completed']].target_idx]))
        self.assertTrue(np.all([loc.shape==(2,3) for loc in df[df['hold_completed']].target_location]))

        # Check that hold penalty trials have one target idx & location
        self.assertTrue(np.all([len(idx)==1 for idx in df[df['hold_completed']==False].target_idx]))
        self.assertTrue(np.all([loc.shape==(3,) for loc in df[df['hold_completed']==False].target_location]))

        # Check that trial segments occur in the correct order
        reward_df = df[df['reward']]
        for i in range(len(reward_df)):
            trial = reward_df.iloc[i]
            self.assertTrue(trial['first_target_on_time'] < trial['hold_start_time'])
            self.assertTrue(trial['hold_start_time'] < trial['delay_start_time'])
            self.assertTrue(trial['delay_start_time'] < trial['go_cue_time'])
            self.assertTrue(trial['go_cue_time'] < trial['reach_end_time'])
            self.assertTrue(trial['reach_end_time'] < trial['reward_start_time'])
            self.assertTrue(trial['reward_start_time'] < trial['trial_end_time'])
            
        penalty_df = df[df['penalty']]
        for i in range(len(penalty_df)):
            trial = penalty_df.iloc[i]
            self.assertTrue(trial['first_target_on_time'] < trial['penalty_start_time'])
            self.assertTrue(trial['penalty_start_time'] < trial['trial_end_time'])
            
        # Check pause events
        pause_df = df[~np.isnan(df['pause_event'])]
        self.assertTrue(np.all(pause_df['pause_event'] == task_codes['PAUSE_START']))
        self.assertTrue(np.all(~np.isnan(pause_df['pause_start_time'])))
        for i in range(len(pause_df)):
            trial = pause_df.iloc[i]
            self.assertTrue(trial['pause_start_time'] == trial['trial_end_time'])
        
    def test_tabulate_behavior_data_tracking_task(self):
        subjects = ['test', 'test']
        ids = [8461, 8461]
        dates = ['2023-02-25', '2023-02-25']
        df = tabulate_behavior_data_tracking_task(data_dir, subjects, ids, dates)  # no penalties in this session
        self.assertEqual(len(df), 42) # 21 total trials, duplicated
        self.assertEqual(len(df.columns), 8+0+18) # no. columns in base tabulate func + no. of user-inputted metadata fields + no. columns in tabulate wrapper

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
        self.assertEqual(len(df.columns), 8+0+18) # no. columns in base tabulate func + no. of user-inputted metadata fields + no. columns in tabulate wrapper

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

    def test_tabulate_behavior_data_random_targets(self):

        subjects = ['Leo', 'Leo']
        ids = [1957, 1959]
        dates = ['2025-02-13', '2025-02-13']

        df = tabulate_behavior_data_random_targets(data_dir, subjects, ids, dates, metadata = ['sequence_params'])
        self.assertEqual(len(df), 66) #check correct length 
        self.assertEqual(len(df.columns), 18+1)  #check correct number of columns
        for loc in df['target_location']:
            self.assertEqual(loc.shape[0], 3) #3 coordinates per target location 
            self.assertLess(np.linalg.norm(loc), 10) #values in target location should be less than 10 
        
        # Visualization check 
        example_reaches = df[-5:] #last 5 reaches in the earlier dataframe
        example_traj = tabulate_kinematic_data(data_dir, example_reaches['subject'], example_reaches['te_id'],
                                               example_reaches['date'], example_reaches['target_on'], 
                                               example_reaches['cursor_enter_target'], datatype = 'cursor')
        ex_targets = example_reaches['target_location'].to_numpy()
        bounds = [-5,5,-5,5,-5,5] #equal bounds to make visualization appear as spheres
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = default_colors[:len(ex_targets)] #match colors from the trajectories
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        for idx, path in enumerate(example_traj):
            ax.plot(*path.T)
            visualization.plot_sphere(ex_targets[idx], color = colors[idx], radius = 0.5, 
                                      bounds = bounds, ax = ax)
        figname = 'tabulate_behavior_random_targets.png' 
        visualization.savefig(docs_dir, figname, transparent = False)
    
    def test_tabulate_readyset_data(self):

        subjects = ['churro']
        ids = [20778,]
        dates = ['2025-06-12']

        df = tabulate_behavior_data_readyset(data_dir, subjects, ids, dates, metadata = ['target_radius' , 'pertubation_rotation'], version = "v1")
        self.assertEqual(len(df), 212) #check correct length 
        self.assertEqual(len(df.columns), 38)  #check correct number of columns

        # Visualization Check 
        df_hc = df[df['hold_completed']].reset_index()
        example_reaches = df_hc[0:6].copy()
        example_traj = tabulate_kinematic_data(data_dir, example_reaches['subject'], example_reaches['te_id'],
                                               example_reaches['date'], example_reaches['ready_start_time'], 
                                               example_reaches['trial_end_time'], datatype = 'cursor')
        
        example_reaches['cursor_traj'] = example_traj

        fig, ax = plt.subplots(figsize=(8, 8))
        color_dict = {'ready': 'red', 'set': 'orange', 'go': 'green'}
        style_option = {'reward': '-', 'penalty': '--'}
        tone_space = 500 #space between tones is 0.5 seconds or 500 samples 
        tarcir = []

        for idx, traj in enumerate(example_traj):

            if example_reaches.loc[idx, 'reward']:
                style = style_option['reward']
            else:
                style = style_option['penalty']

            tarloc = example_reaches.loc[idx, 'target_location']
            tarcir.append(tarloc)
            n = traj.shape[0]

            # Plot first 500 samples (the ready)
            if n > 0:
                ax.plot(traj[:min(tone_space, n), 0], traj[:min(tone_space, n), 1], color=color_dict['ready'], linestyle =style, linewidth = 3)
            # Plot next 500 samples (the set)
            if n > tone_space:
                ax.plot(traj[tone_space:min(2*tone_space, n), 0], traj[tone_space:min(2*tone_space, n), 1], color=color_dict['set'], linestyle =style, linewidth = 3)
            # Plot last 500 samples (the go)
            if n > 2*tone_space:
                ax.plot(traj[2*tone_space:min(3*tone_space, n), 0], traj[2*tone_space:min(3*tone_space, n), 1], color=color_dict['go'], linestyle =style, linewidth = 3)

        tarcir.append([0,0,0]) #add center target circle

        visualization.plot_circles(tarcir, circle_radius = 1.0, circle_color = 'm', bounds = [-10,10,-10,10], ax = ax)

        ax.set_aspect('equal', adjustable='box')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Example Cursor Trajectories')
        reward_line = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label='Reward')
        penalty_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Failed')

        # Legend for colors
        ready_label = mlines.Line2D([], [], color='red', linewidth=3, label='Ready')
        set_label = mlines.Line2D([], [], color='orange', linewidth=3, label='Set')
        go_label = mlines.Line2D([], [], color='green', linewidth=3, label='Go')

        # Add legends separately (so they don’t merge)
        first_legend = ax.legend(handles=[reward_line, penalty_line], loc='upper left', title='Reward Outcome')
        ax.add_artist(first_legend)  
        ax.legend(handles=[ready_label, set_label, go_label], loc='upper right', title='Part of Trial')

        figname = 'tabulate_behavior_readyset.png' 
        visualization.savefig(docs_dir, figname, transparent = False)

        #test for newer version
        subjects = ['GE021']
        ids = [4859,]
        dates = ['2025-12-18']

        df = tabulate_behavior_data_readyset(data_dir, subjects, ids, dates, metadata = ['target_radius' , 'pertubation_rotation'], version = "v2")
        self.assertEqual(len(df), 102) #check correct length 
        self.assertEqual(len(df.columns), 37)  #check correct number of columns


    def test_tabulate_kinematic_data(self):
        subjects = [self.subject, self.subject]
        ids = [self.te_id, self.te_id]
        dates = [self.date, self.date]

        df = tabulate_behavior_data_center_out(write_dir, subjects, ids, dates, df=None)
        
        # Only consider completed reaches
        df = df[df['reach_completed']]
        kin = tabulate_kinematic_data(write_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                      datatype='cursor', samplerate=1000)

        self.assertEqual(len(df), len(kin))

        plt.figure()
        bounds = [-10, 10, -10, 10]
        visualization.plot_trajectories(kin, bounds=bounds)
        figname = 'tabulate_kinematics.png' # should look very similar to get_trial_aligned_trajectories.png
        visualization.savefig(docs_dir, figname, transparent=False)

        # Test speed and acceleration
        dst = tabulate_kinematic_data(write_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                      deriv=0, norm=True, datatype='cursor', samplerate=1000, filter_kinematics=True)
        spd = tabulate_kinematic_data(write_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                      deriv=1, norm=True, datatype='cursor', samplerate=1000, filter_kinematics=True)
        acc = tabulate_kinematic_data(write_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                      deriv=2, norm=True, datatype='cursor', samplerate=1000, filter_kinematics=True)
        plt.figure()
        visualization.plot_timeseries(dst[0], 1000)
        visualization.plot_timeseries(spd[0], 1000)
        visualization.plot_timeseries(acc[0], 1000)
        plt.legend(['distance', 'speed', 'acceleration'])
        plt.xlabel('time from go cue (s)')
        plt.ylabel('kinematics (cm)')
        figname = 'tabulate_kinematics_derivative.png' # should look very similar to get_trial_aligned_trajectories.png
        visualization.savefig(docs_dir, figname, transparent=False)

        def plot_kin(df, start_event, end_event, filter_kinematics=False):            
            dst = tabulate_kinematic_data(data_dir, df['subject'], df['te_id'], df['date'], df[start_event], df[end_event], 
                                        deriv=0, norm=True, datatype='cursor', samplerate=1000, filter_kinematics=filter_kinematics)
            spd = tabulate_kinematic_data(data_dir, df['subject'], df['te_id'], df['date'], df[start_event], df[end_event],
                                        deriv=1, norm=True, datatype='cursor', samplerate=1000, filter_kinematics=filter_kinematics)
            acc = tabulate_kinematic_data(data_dir, df['subject'], df['te_id'], df['date'], df[start_event], df[end_event],
                                        deriv=2, norm=True, datatype='cursor', samplerate=1000, filter_kinematics=filter_kinematics)
            plt.figure()
            plt.subplot(3,1,1) # position
            for i in range(len(dst)):
                visualization.plot_timeseries(dst[i], 1000, alpha=0.1, color='k')
            plt.ylabel('distance (cm)')
            plt.subplot(3,1,2) # speed
            for i in range(len(spd)):
                visualization.plot_timeseries(spd[i], 1000, alpha=0.1, color='k')
            plt.ylabel('speed (cm/s)')
            plt.subplot(3,1,3) # acceleration
            for i in range(len(acc)):
                visualization.plot_timeseries(acc[i], 1000, alpha=0.1, color='k')
            plt.ylabel('acceleration (cm/s^2)')
            plt.xlabel('time from go cue (s)')
            plt.tight_layout()

        # Plot all trials together
        plot_kin(df, 'go_cue_time', 'reach_end_time')
        figname = 'tabulate_kinematics_beignet.png'
        visualization.savefig(docs_dir, figname, transparent=False)

        # Test return_nan arg
        df = tabulate_behavior_data_center_out(write_dir, subjects, ids, dates, df=None)
        df['te_id'] = 0 
        kin_nan = tabulate_kinematic_data(write_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                            datatype='cursor', samplerate=1000, return_nan=True)
        self.assertTrue(np.isnan(kin_nan[0]))

        # Test data from the human rig
        subject = 'CES003'
        te_id = 2234
        date = '2025-03-04'
        df = tabulate_behavior_data_center_out(data_dir, [subject], [te_id], [date])
        df = df[df['reach_completed']]
        plot_kin(df, 'go_cue_time', 'reach_end_time')
        figname = 'tabulate_kinematics_ces.png'
        visualization.savefig(docs_dir, figname, transparent=False)

        raw = tabulate_kinematic_data(data_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                 datatype='cursor', samplerate=1000)
        raw_filt = tabulate_kinematic_data(data_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                 datatype='cursor', samplerate=1000, filter_kinematics=True, low_cut=5, buttord=2)
        nan = tabulate_kinematic_data(data_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                 datatype='user_screen', samplerate=1000, remove_nan=False)
        nan_filt = tabulate_kinematic_data(data_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                 datatype='user_screen', samplerate=1000, low_cut=5, buttord=2, filter_kinematics=True, remove_nan=False)        
        pos = tabulate_kinematic_data(data_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                 datatype='user_screen', samplerate=1000)
        pos_filt = tabulate_kinematic_data(data_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                 datatype='user_screen', samplerate=1000, filter_kinematics=True, low_cut=5, buttord=2)        
        spd = tabulate_kinematic_data(data_dir, df['subject'], df['te_id'], df['date'], df['go_cue_time'], df['reach_end_time'], 
                                       deriv=1, norm=True, datatype='cursor', samplerate=1000, filter_kinematics=True)
        weird_trials = np.where([np.any(s > 500) for s in spd])[0]
        plt.figure(figsize=(5,6))
        plt.subplot(3,1,1)
        for i in weird_trials:
            visualization.plot_timeseries(raw[i][:,0], 1000)
            visualization.plot_timeseries(raw_filt[i][:,0], 1000, color='k', alpha=0.5)
        plt.ylabel('x position (cm)')
        plt.xlabel('')
        plt.title('cursor')
        plt.legend(['raw', 'filtered'])
        plt.subplot(3,1,2)
        for i in weird_trials:
            visualization.plot_timeseries(nan[i][:,0], 1000)
            visualization.plot_timeseries(nan_filt[i][:,0], 1000, color='k', alpha=0.5)
        plt.ylabel('x position (cm)')
        plt.xlabel('time from go cue (s)')
        plt.title('user_screen')
        plt.subplot(3,1,3)
        for i in weird_trials:
            visualization.plot_timeseries(pos[i][:,0], 1000)
            visualization.plot_timeseries(pos_filt[i][:,0], 1000, color='k', alpha=0.5)
        plt.ylabel('x position (cm)')
        plt.xlabel('time from go cue (s)')
        plt.title('user_screen interp')
        plt.tight_layout()

        figname = 'kinematics_interpolation.png'
        visualization.savefig(docs_dir, figname, transparent=False)


    def test_tabulate_features(self):
        preproc_dir = data_dir
        subject = 'affi'
        te_id = 17269
        date = '2024-05-03'
        subjects = [subject, subject, subject]
        te_ids = [te_id, te_id, te_id]
        dates = [date, date, date]
        start_time = 10
        end_time = 30
        start_times = [10, 15, 20]
        end_times = [14, 18, 28]
        with open(os.path.join(data_dir, 'test_decoder.pkl'), 'rb') as file:
            decoder = pickle.load(file)

        # Load the full features and state data for comparison
        features_offline, samplerate_offline = extract_lfp_features(
            preproc_dir, subject, te_id, date, decoder, 
            start_time=start_time, end_time=end_time)
        features_online, samplerate_online = get_extracted_features(
            preproc_dir, subject, te_id, date, decoder,
            start_time=start_time, end_time=end_time)
        time_offline = np.arange(len(features_offline))/samplerate_offline + start_time
        time_online = np.arange(len(features_online))/samplerate_online + start_time

        plt.figure(figsize=(8,3))
        plt.plot(time_offline, features_offline[:,1], alpha=0.8, label='offline')
        plt.plot(time_online, features_online[:,1], alpha=0.8, label='online')
        plt.xlabel('time (s)')
        plt.ylabel('power')
        plt.title('readout 1')
        
        plt.tight_layout()

        # Tabulate the segments
        features_offline, samplerate_offline = tabulate_lfp_features(
            preproc_dir, subjects, te_ids, dates, start_times, end_times, decoder)
        features_online, samplerate_online = tabulate_feature_data(
            preproc_dir, subjects, te_ids, dates, start_times, end_times, decoder)

        for idx in range(len(start_times)):
            time_offline = np.arange(len(features_offline[idx]))/samplerate_offline + start_times[idx]
            time_online = np.arange(len(features_online[idx]))/samplerate_online + start_times[idx]

            plt.plot(time_offline, features_offline[idx][:,1], 'k--')
            plt.plot(time_online, features_online[idx][:,1], 'k--')
        
        # Add legends
        plt.plot([], [], 'k--', label='segments')
        plt.legend()

        filename = 'tabulate_decoder_features.png'
        visualization.savefig(docs_dir, filename, transparent=False)

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
                               trigger_times, time_before, time_after, drive_number=1, datatype='lfp')
        
        trial_start_codes = [CURSOR_ENTER_CENTER_TARGET]
        trial_end_codes = [TRIAL_END]
        ts_data_single_file = get_lfp_aligned(write_dir, self.subject, self.te_id, self.date, 
                                              trial_start_codes, trial_end_codes, time_before, time_after, drive_number=1)
     
        print(ts_data_single_file.shape)

        self.assertEqual(ts_data_single_file.shape, ts_data.shape)

        # Test getting a single channel
        ts_data, samplerate = tabulate_ts_data(write_dir, df['subject'], df['te_id'], df['date'],
                                 trigger_times, time_before, time_after, drive_number=1, datatype='lfp', channels=[0])

        self.assertEqual(ts_data.shape[1], 1)


    def test_tabulate_ts_segments(self):

        subjects = [self.subject]
        ids = [self.te_id]
        dates = [self.date]

        df = tabulate_behavior_data_center_out(write_dir, subjects, ids, dates, df=None)

        # Only consider completed trials
        df = df[df['reach_completed']]
        ts_seg, samplerate = tabulate_ts_segments(write_dir, df['subject'], df['te_id'], df['date'], 
                                                  df['go_cue_time'], df['reach_end_time'], drive_number=1)

        self.assertEqual(len(df), len(ts_seg))
        
        trial_start_codes = [CENTER_TARGET_OFF]
        trial_end_codes = CURSOR_ENTER_PERIPHERAL_TARGET + [TRIAL_END]
        ts_seg_single_file, _ = get_lfp_segments(write_dir, self.subject, self.te_id, self.date, trial_start_codes, 
                                                 trial_end_codes, drive_number=1, trial_filter=lambda t: TRIAL_END not in t)

        self.assertEqual(len(ts_seg_single_file), len(ts_seg))
        for i in range(len(ts_seg)):
            self.assertEqual(ts_seg[i].shape, ts_seg_single_file[i].shape)

    def test_get_spike_data_segment(self):
        # Check basic functionality
        start_time = 0.1
        end_time = 0.15
        bin_width = 0.01
        spike_segments_port1, bins = get_spike_data_segment(data_dir, 'affi', 18378, datetime.date(2024, 9, 23), start_time, end_time, 1, bin_width=bin_width)
        spike_segments_port2, _ = get_spike_data_segment(data_dir, 'affi', 18378, datetime.date(2024, 9, 23), start_time, end_time, 2, bin_width=bin_width)
        
        port1_keys = list(spike_segments_port1.keys())
        port2_keys = list(spike_segments_port2.keys())
        port1_segment_lens = [len(spike_segments_port1[key]) for key in port1_keys]
        port2_segment_lens = [len(spike_segments_port2[key]) for key in port2_keys]
        self.assertEqual(port1_segment_lens[0], np.round((end_time-start_time)/bin_width).astype(int)) # Check that the data segment is the expected length
        self.assertEqual(len(np.unique(port1_segment_lens)), 1) # Check that the data segments for all units are the same length
        self.assertFalse(len(spike_segments_port1) == len(spike_segments_port2)) # Check that different data (units) is loaded for different ports

        # Check a different bin width
        bin_width=0.005
        spike_segments_port1, bins = get_spike_data_segment(data_dir, 'affi', 18378, datetime.date(2024, 9, 23), start_time, end_time, 1, bin_width=bin_width)
        port1_keys = list(spike_segments_port1.keys())
        port1_segment_lens = [len(spike_segments_port1[key]) for key in port1_keys]
        self.assertEqual(port1_segment_lens[0], np.round((end_time-start_time)/bin_width).astype(int)) # Check that the data segment is the expected length
        self.assertEqual(len(np.unique(port1_segment_lens)), 1) # Check that the data segments for all units are the same length

        # Check unbinned spike segments
        end_time=1
        spike_segments_port1, bins = get_spike_data_segment(data_dir, 'affi', 18378, datetime.date(2024, 9, 23), start_time, end_time, 1, bin_width=None)
        spike_times = spike_segments_port1['24']
        self.assertTrue(np.logical_and(spike_times>=start_time, spike_times<=end_time).all()) # Check that all times are between the start and end
        self.assertEqual(np.sum(np.diff(spike_times)>0), len(spike_times)-1) # Check that spike times are monotonic.

    def test_get_spike_data_aligned(self):
        time_before = 0.1
        time_after = 0.4
        bin_width = 0.01
        flash_df = bmi3d.tabulate_behavior_data_flash(data_dir, ['affi'], [18378], [datetime.date(2024, 9, 23)])
        trigger_times = np.array(flash_df['flash_start_time'])
        spike_aligned, unit_labels, bins = get_spike_data_aligned(data_dir, 'affi', 18378, datetime.date(2024, 9, 23), trigger_times, time_before, time_after, 1, bin_width=bin_width)

        self.assertEqual(spike_aligned.shape[1], len(unit_labels))  # Assert that the correct number of units are in the aligned data. Plot will check other axis.

        # Plot for example figure 
        spike_aligned1, unit_labels, bins1 = get_spike_data_aligned(data_dir,'affi', 18378, datetime.date(2024, 9, 23), trigger_times, time_before,time_after, 1, bin_width=0.01 )
        spike_aligned2, unit_labels, bins2 = get_spike_data_aligned(data_dir,'affi', 18378, datetime.date(2024, 9, 23), trigger_times, time_before,time_after, 1, bin_width=0.001 )
        iunit = 24
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].pcolor(bins1, np.arange(len(trigger_times)), spike_aligned1[:,iunit,:].T, cmap='Grays') #24
        ax[1].pcolor(bins2, np.arange(len(trigger_times)), spike_aligned2[:,iunit,:].T, cmap='Grays') #24


        ax[0].set(title=f"Unit label: {unit_labels[iunit]} - Bin width: 10ms", xlabel='Time [s]', ylabel='Trial')
        ax[1].set(title=f"Unit label: {unit_labels[iunit]} - Bin width: 100ms", xlabel='Time [s]', ylabel='Trial')
        ax[0].set_xticks(bins1[::10], np.round(bins1[::10],2))
        ax[1].set_xticks(bins2[::100], np.round(bins2[::100],2))
        fig.tight_layout()
    
        filename = 'spike_align_example.png'
        visualization.savefig(docs_dir, filename)
   
    def test_tabulate_spike_data_segments(self):
        subjects = ['affi', 'affi']
        te_ids = [18378, 18378]
        dates = [datetime.date(2024, 9, 23), datetime.date(2024, 9, 23)]
        drives = [1,2]
        start_times = [.1,.1]
        end_times = [.15,.15]
        bin_width = 0.01
        segments, bins = tabulate_spike_data_segments(data_dir, subjects, te_ids, dates, start_times, end_times, drives, bin_width)

        self.assertEqual(len(segments), len(subjects))
        self.assertEqual(len(segments[0]['0']), 5)
        self.assertEqual(len(segments[1]['0']), 5)

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

        df = tabulate_behavior_data_flash(write_dir, [subject, subject], [te_id, te_id], [date, date], df=None)
        self.assertEqual(len(df), 26) # 13 total trials, duplicated
        trial = df.iloc[12] # last trial of the first session
        self.assertGreater(trial['trial_end_time'], 0.)

        trial = df.iloc[13] # first trial of the second session
        self.assertEqual(trial['prev_trial_end_time'], 0.)

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
            self.assertGreater(df['trial_gain'][trial], 0.)
            self.assertLessEqual(df['trial_gain'][trial], 1.0)
            self.assertGreater(df['trial_time'][trial], 0.)
            self.assertLessEqual(df['trial_time'][trial], 100.)
            self.assertGreater(df['trial_power'][trial], 0.)
            self.assertLessEqual(df['trial_power'][trial], 25.0)

    def test_tabulate_poisson_trial_times(self):
        subjects = ['test']
        ids = [6577]
        dates = ['2022-08-19']
        df = tabulate_poisson_trial_times(data_dir, subjects, ids, dates)

        self.assertIn('trial_time', df.columns)
        self.assertTrue(len(df) > 0)

    def test_get_kilosort_foldername(self):
        subject='affi'
        te_id = 1000
        date = datetime.date(2024,9,23)
        data_source = 'Neuropixel'
        foldername = get_kilosort_foldername(subject, te_id, date, data_source)
        self.assertEqual(foldername, "2024-09-23_Neuropixel_affi_te1000")

        # Test multiple TEs
        te_ids = [1000,1001]
        foldername = get_kilosort_foldername(subject, te_ids, date, data_source)
        self.assertEqual(foldername, "2024-09-23_Neuropixel_affi_te1000_te1001")

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
                   'CURSOR_ENTER_CORNER_TARGET': list(range(81, 85)),
                   'CORNER_TARGET_ON': list(range(17, 21)),
                   'CORNER_TARGET_OFF': list(range(33, 37)),
                   'REWARD': 48,
                   'DELAY_PENALTY': 66,
                   'TIMEOUT_PENALTY': 65,
                   'HOLD_PENALTY': 64,
                   'PAUSE': 254,
                   'TIME_ZERO': 238,
                   'TRIAL_END': 239,
                   'TRIAL_START': 2,
                   'CURSOR_ENTER_TARGET': 80,
                   'CURSOR_ENTER_TARGET_RAMP_UP': 81,
                   'CURSOR_ENTER_TARGET_RAMP_DOWN': 82,
                   'CURSOR_LEAVE_TARGET': 96,
                   'CURSOR_LEAVE_TARGET_RAMP_UP': 97,
                   'CURSOR_LEAVE_TARGET_RAMP_DOWN': 98,
                   'OTHER_PENALTY': 79,
                   'PAUSE_START': 128,
                   'PAUSE_END': 129, 
                   'CUE': 112}
        yaml_write(params_file, params)

        # Testing pkl_read
        task_codes = yaml_read(params_file)

        self.assertEqual(params,task_codes)

        task_codes_file = load_bmi3d_task_codes('task_codes.yaml')

        self.assertDictEqual(params, task_codes_file)

    def test_load_lasers(self):
        lasers = load_bmi3d_lasers()
        for l in lasers:
            self.assertIn('name', l.keys())
            self.assertIn('stimulation_site', l.keys())
            self.assertIn('trigger', l.keys())
            self.assertIn('trigger_dch', l.keys())
            self.assertIn('sensor', l.keys())
            self.assertIn('sensor_ach', l.keys())

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

        acq_ch_position, acq_chs, connected_elecs = load_chmap(drive_type='NP_Insert137')

    def test_align_neuropixel_recoring_drive(self):
        # Create plots for documentation with affi and biegnet
        fig, ax = plt.subplots(1,2)
        subjects = ['affi', 'beignet']
        neuropixel_drive='NP_Insert137'
        drive2 = 'ECoG244'
        for iax, subject in enumerate(subjects):
            if subject == 'affi':
                theta=90
            else:
                theta=0
            aligned_np_drive_coordinates, drive2_coordinates, recording_sites, acq_elecs = align_neuropixel_recoring_drive(neuropixel_drive, drive2, subject, theta=theta)
            [ax[iax].annotate(str(acq_elecs[ipt]), (drive2_coordinates[ipt,0], drive2_coordinates[ipt,1]), ha='center', va='center', color='k',fontsize=4) for ipt in range(len(acq_elecs))]
            [ax[iax].annotate(str(recording_sites[ipt]), (aligned_np_drive_coordinates[ipt,0], aligned_np_drive_coordinates[ipt,1]), ha='center', va='center', color='r',fontsize=4) for ipt in range(len(recording_sites))]
            visualization.base.overlay_sulci_on_spatial_map(subject, 'LM1', drive2, theta=theta, ax=ax[iax])
            ax[iax].set(xlim=(-8,8), ylim=(-8,8), title=f'{subject}')
        visualization.savefig(docs_dir, f'{neuropixel_drive}_{drive2}_alignment.png', transparent=False)

        fig, ax = plt.subplots(1,2)
        neuropixel_drive='NP_Insert72'
        for iax, subject in enumerate(subjects):
            if subject == 'affi':
                theta=90
            else:
                theta=0
            center = (5,5)
            aligned_np_drive_coordinates, drive2_coordinates, recording_sites, acq_elecs = align_neuropixel_recoring_drive(neuropixel_drive, drive2, subject, theta=theta, center=center)
            [ax[iax].annotate(str(acq_elecs[ipt]), (drive2_coordinates[ipt,0], drive2_coordinates[ipt,1]), ha='center', va='center', color='k',fontsize=4) for ipt in range(len(acq_elecs))]
            [ax[iax].annotate(str(recording_sites[ipt]), (aligned_np_drive_coordinates[ipt,0], aligned_np_drive_coordinates[ipt,1]), ha='center', va='center', color='r',fontsize=4) for ipt in range(len(recording_sites))]
            visualization.base.overlay_sulci_on_spatial_map(subject, 'LM1', drive2, theta=theta, center=center, ax=ax[iax])
            ax[iax].set(xlim=(-3,13), ylim=(-3,13), title=f'{subject}')
        visualization.savefig(docs_dir, f'{neuropixel_drive}_{drive2}_alignment.png', transparent=False)

        # Test the rest of the combinations
        for subject in subjects:
            neuropixel_drive = 'NP_Insert72'
            drive2 = 'ECoG244'
            aligned_np_drive_coordinates, drive2_coordinates ,  recording_sites, acq_elecs = align_neuropixel_recoring_drive(neuropixel_drive, drive2, subject)
            fig, ax = plt.subplots(1,1)
            [ax.annotate(str(acq_elecs[ipt]), (drive2_coordinates[ipt,0], drive2_coordinates[ipt,1]), ha='center', va='center', color='k',fontsize=4) for ipt in range(len(acq_elecs))]
            [ax.annotate(str(recording_sites[ipt]), (aligned_np_drive_coordinates[ipt,0], aligned_np_drive_coordinates[ipt,1]), ha='center', va='center', color='r', fontsize=4) for ipt in range(len(recording_sites))]
            visualization.base.overlay_sulci_on_spatial_map(subject, 'LM1', drive2, theta=0)
            ax.set(xlim=(-8,8), ylim=(-8,8), title=f'{subject}')
            visualization.savefig(write_dir, f'{neuropixel_drive}_{drive2}_alignment_{subject}.png', transparent=False)

            neuropixel_drive='NP_Insert137'
            aligned_np_drive_coordinates, drive2_coordinates , recording_sites, acq_elecs = align_neuropixel_recoring_drive(neuropixel_drive, drive2, subject)
            fig, ax = plt.subplots(1,1)
            [ax.annotate(str(acq_elecs[ipt]), (drive2_coordinates[ipt,0], drive2_coordinates[ipt,1]), ha='center', va='center', color='k',fontsize=4) for ipt in range(len(acq_elecs))]
            [ax.annotate(str(recording_sites[ipt]), (aligned_np_drive_coordinates[ipt,0], aligned_np_drive_coordinates[ipt,1]), ha='center', va='center', color='r', fontsize=4) for ipt in range(len(recording_sites))]
            visualization.base.overlay_sulci_on_spatial_map(subject, 'LM1', drive2, theta=0)
            ax.set(xlim=(-8,8), ylim=(-8,8), title=f'{subject}')
            visualization.savefig(write_dir, f'{neuropixel_drive}_{drive2}_alignment_{subject}.png', transparent=False)

            ## Test with opto drive
            drive2 = 'Opto32'
            neuropixel_drive = 'NP_Insert72'
            # acq_ch_position, acq_chs, connected_elecs = load_chmap(drive_type=drive2)
            aligned_np_drive_coordinates, drive2_coordinates , recording_sites, acq_elecs = align_neuropixel_recoring_drive(neuropixel_drive, drive2, subject)
            fig, ax = plt.subplots(1,1)
            [ax.annotate(str(acq_elecs[ipt]), (drive2_coordinates[ipt,0], drive2_coordinates[ipt,1]), ha='center', va='center', color='k',fontsize=5) for ipt in range(len(acq_elecs))]
            [ax.annotate(str(recording_sites[ipt]), (aligned_np_drive_coordinates[ipt,0], aligned_np_drive_coordinates[ipt,1]), ha='center', va='center', color='r', fontsize=5) for ipt in range(len(recording_sites))]
            ax.set(xlim=(-8,8), ylim=(-8,8), title=f'{subject}')
            visualization.base.overlay_sulci_on_spatial_map(subject, 'LM1', 'ECoG244', theta=0)
            ax.set_aspect('equal')
            visualization.savefig(write_dir, f'{neuropixel_drive}_{drive2}_alignment_{subject}.png', transparent=False)

            neuropixel_drive='NP_Insert137'
            aligned_np_drive_coordinates, drive2_coordinates , recording_sites, acq_elecs = align_neuropixel_recoring_drive(neuropixel_drive, drive2, subject)
            fig, ax = plt.subplots(1,1)
            [ax.annotate(str(acq_elecs[ipt]), (drive2_coordinates[ipt,0], drive2_coordinates[ipt,1]), ha='center', va='center', color='k',fontsize=5) for ipt in range(len(acq_elecs))]
            [ax.annotate(str(recording_sites[ipt]), (aligned_np_drive_coordinates[ipt,0], aligned_np_drive_coordinates[ipt,1]), ha='center', va='center', color='r', fontsize=5) for ipt in range(len(recording_sites))]
            visualization.base.overlay_sulci_on_spatial_map(subject, 'LM1', 'ECoG244', theta=0)
            ax.set_aspect('equal')
            ax.set(xlim=(-8,8), ylim=(-8,8), title=f'{subject}')
            visualization.savefig(write_dir, f'{neuropixel_drive}_{drive2}_alignment_{subject}.png', transparent=False)

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

class DatabaseTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        db.BMI3D_DBNAME = 'test_aopy'
        db.DB_TYPE = 'bmi3d'
        from db.tracker import models
        
        # Clear the database
        models.Decoder.objects.using('test_aopy').all().delete()
        models.TaskEntry.objects.using('test_aopy').all().delete()
        models.Subject.objects.using('test_aopy').all().delete()
        models.Experimenter.objects.using('test_aopy').all().delete()
        models.Sequence.objects.using('test_aopy').all().delete()
        models.Task.objects.using('test_aopy').all().delete()
        models.Feature.objects.using('test_aopy').all().delete()
        models.Generator.objects.using('test_aopy').all().delete()
        models.System.objects.using('test_aopy').all().delete()

        # Make some test entries for subject, experimenter, and task 
        subj = models.Subject(name="test")
        subj.save(using='test_aopy')
        expm = models.Experimenter(name="experimenter_1")
        expm.save(using='test_aopy')
        task = models.Task(name="nothing")
        task.save(using='test_aopy')
        task = models.Task(name="manual control")
        task.save(using='test_aopy')
        task = models.Task(name="tracking")
        task.save(using='test_aopy')
        feat = models.Feature(name="feat_1")
        feat.save(using='test_aopy')
        gen = models.Generator(name="test_gen", static=False)
        gen.save(using='test_aopy')
        seq = models.Sequence(generator_id=gen.id, task_id=task.id, name="test_seq", params='{"seq_param_1": 1}')
        seq.save(using='test_aopy')

        # Make a basic task entry
        subj = models.Subject.objects.get(name="test")
        task = models.Task.objects.get(name="tracking")
        te = models.TaskEntry(subject_id=subj.id, task_id=task.id)
        te.save(using='test_aopy')

        # Make a manual control task entry
        task = models.Task.objects.get(name="manual control")
        expm = models.Experimenter.objects.get(name="experimenter_1")
        te = models.TaskEntry(subject_id=subj.id, task_id=task.id, experimenter_id=expm.id, entry_name="task_desc",
                            session="test session", project="test project", params='{"task_param_1": 1}', sequence_id=seq.id)
        te.report = '{"runtime": 3.0, "n_trials": 2, "n_success_trials": 1}'
        te.save(using='test_aopy')
        te.feats.set([feat])
        te.save(using='test_aopy')

        system = models.System(name="bmi", path=write_dir, archive="")
        system.save(using='test_aopy')

        # Add a decoder entry that was "trained" on a parent task entry
        from riglib.bmi.state_space_models import StateSpaceEndptVel2D
        from riglib.bmi.bmi import Decoder, MachineOnlyFilter
        ssm = StateSpaceEndptVel2D()
        A, B, W = ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        decoder = Decoder(filt, units, ssm, binlen=0.1)
        parent = db.create_decoder_parent('project', 'session')
        db.save_decoder(parent, decoder, 'test_decoder')
        decoder = db.lookup_decoders()[0]

        # And a flash task entry
        task = models.Task.objects.get(name="manual control")
        expm = models.Experimenter.objects.get(name="experimenter_1")
        te = models.TaskEntry(subject_id=subj.id, task_id=task.id, experimenter_id=expm.id, entry_name="flash")
        te.report = '{"runtime": 3.0, "n_trials": 2, "n_success_trials": 0}'
        te.save(using='test_aopy')

        # Add a bmi task entry
        task = models.Task(name="bmi control")
        task.save(using='test_aopy')
        subj = models.Subject.objects.get(name="test")
        expm = models.Experimenter.objects.get(name="experimenter_1")
        te = models.TaskEntry(subject_id=subj.id, task_id=task.id, 
                              experimenter_id=expm.id, params='{"bmi": '+str(decoder.id)+'}')
        te.save(using='test_aopy')

        # Add a task entry from a different rig
        subj = models.Subject.objects.get(name="test")
        expm = models.Experimenter.objects.get(name="experimenter_1")
        te = models.TaskEntry(subject_id=subj.id, task_id=task.id, experimenter_id=expm.id, rig_name="siberut-bmi")
        te.save(using='test_aopy')

        # Add a perturbation manual control session
        task = models.Task.objects.get(name="manual control")
        expm = models.Experimenter.objects.get(name="experimenter_1")
        te = models.TaskEntry(subject_id=subj.id, task_id=task.id, experimenter_id=expm.id, entry_name="task_desc",
                            session="test session", project="test project", params='{"perturbation_rotation_x": 90}', sequence_id=seq.id)
        te.save(using='test_aopy')

        # And a washout session
        te = models.TaskEntry(subject_id=subj.id, task_id=task.id, experimenter_id=expm.id, entry_name="task_desc",
                            session="test session", project="test project", sequence_id=seq.id)
        te.save(using='test_aopy')


    def test_lookup_sessions(self):
        db.BMI3D_DBNAME = 'test_aopy'

        # Most basic lookup
        all_sessions = db.lookup_sessions()
        sessions = db.lookup_sessions(id=all_sessions[0].id)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].id, sessions[0].id)
        sessions = db.lookup_sessions(id=[all_sessions[0].id, all_sessions[1].id])
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[1].id, all_sessions[1].id)

        # Other sanity tests
        total_sessions = len(db.lookup_sessions())
        self.assertGreater(len(db.lookup_mc_sessions()), 0)
        self.assertGreater(len(db.lookup_flash_sessions()), 0)
        self.assertGreater(len(db.lookup_tracking_sessions()), 0)
        self.assertGreater(len(db.lookup_bmi_sessions()), 0)
        self.assertGreater(len(db.lookup_decoder_parent()), 0)

        # Test filtering
        self.assertEqual(len(db.lookup_sessions(subject="non_existent")), 0)
        self.assertEqual(len(db.lookup_sessions(subject="test")), total_sessions)
        sessions = db.lookup_sessions(subject="test", task_name="manual control",
                                      task_desc="task_desc", session="test session", project="test project",
                                      experimenter="experimenter_1")
        self.assertEqual(sessions[0].task_name, "manual control")
        self.assertEqual(sessions[0].task_desc, "task_desc")
        self.assertEqual(sessions[0].subject, "test")
        self.assertEqual(sessions[0].session, "test session")
        self.assertEqual(sessions[0].project, "test project")
        self.assertEqual(sessions[0].experimenter, "experimenter_1")
        self.assertEqual(str(sessions[0].date), str(datetime.datetime.today().date()))

        # Special case - filter by id
        sessions = db.lookup_sessions(exclude_ids=[all_sessions[0].id,all_sessions[1].id])
        self.assertEqual(len(sessions), total_sessions-2)

        # Special case - arbitrary filter fn
        sessions = db.lookup_sessions(filter_fn=lambda x:x.duration > 0)
        self.assertEqual(len(sessions), 2)

        # Check that changing the db name works
        db.DB_TYPE = 'unknown'
        self.assertEqual(len(db.lookup_sessions()), 0)
        db.DB_TYPE = 'bmi3d'
        db.BMI3D_DBNAME = 'rig2'
        self.assertRaises(Exception, db.lookup_sessions)
        db.BMI3D_DBNAME = 'test_aopy'

        # And the rig name
        sessions = db.lookup_bmi_sessions(rig_name='siberut-bmi')
        self.assertEqual(len(sessions), 1)

    def test_lookup_decoders(self):
        db.BMI3D_DBNAME = 'test_aopy'

        # Most basic lookup
        all_decoders = db.lookup_decoders()
        decoders = db.lookup_decoders(id=all_decoders[0].id)
        self.assertEqual(len(decoders), 1)
        self.assertEqual(decoders[0].id, decoders[0].id)

        # Other sanity tests
        total_decoders = 1
        self.assertEqual(len(db.lookup_decoders()), total_decoders)
        self.assertEqual(len(db.lookup_decoders(name="project_session_test_decoder")), total_decoders)

        # Test filtering
        self.assertEqual(len(db.lookup_decoders(name="non_existent")), 0)
        self.assertEqual(len(db.lookup_decoders(name="project_session_test_decoder")), total_decoders)
        decoders = db.lookup_decoders(parent_id=db.lookup_decoder_parent()[0].id)
        self.assertEqual(len(decoders), 1)
        self.assertEqual(decoders[0].name, "project_session_test_decoder")

    def test_filter_functions(self):
        db.BMI3D_DBNAME = 'test_aopy'

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

        # Test filtering in lookup_mc_sessions and lookup_flash_sessions
        filter_fn = db.filter_has_features("feat_1")
        sessions = db.lookup_mc_sessions(filter_fn=filter_fn)
        self.assertEqual(len(sessions), 1)
        sessions = db.lookup_flash_sessions(filter_fn=filter_fn)
        self.assertEqual(len(sessions), 0)

    def test_BMI3DTaskEntry(self):
        db.BMI3D_DBNAME = 'test_aopy'

        # Test that all the fields work as they should
        te = db.lookup_sessions(task_desc='task_desc')[0]
        self.assertEqual(te.subject, 'test')
        self.assertEqual(te.experimenter, 'experimenter_1')
        self.assertEqual(str(te.date), str(datetime.datetime.today().date()))
        self.assertEqual(type(te.datetime), datetime.datetime)
        self.assertEqual(te.session, 'test session')
        self.assertEqual(te.project, 'test project')
        self.assertEqual(te.task_name, 'manual control')
        self.assertEqual(te.task_desc, 'task_desc')
        self.assertEqual(te.notes, '')
        self.assertEqual(te.duration, 3.0)
        self.assertEqual(te.n_trials, 2)
        self.assertEqual(te.n_rewards, 1)
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
        np.testing.assert_allclose(te.get_exp_mapping(), np.eye(3))
        self.assertEqual(te.has_exp_perturbation(), False)

        # Test a bmi session and decoder
        te = db.lookup_sessions(task_name="bmi control")[0]
        decoder = te.get_decoder_record()
        self.assertEqual(decoder.name, "project_session_test_decoder")
        self.assertRaises(Exception, te.get_decoder) # No decoder file present

        # Test preprocess function
        te = db.lookup_sessions(task_desc='task_desc')[0]
        error = te.preprocess(data_dir, write_dir)
        self.assertEqual(error, None)

    def test_list_entry_details(self):
        db.BMI3D_DBNAME = 'test_aopy'
        sessions = db.lookup_sessions(task_desc='task_desc')
        subject, te_id, date = db.list_entry_details(sessions)
        self.assertEqual(len(subject), len(sessions))
        for s in subject:
            self.assertEqual(s, 'test')
        
    def test_group_entries(self):
        db.BMI3D_DBNAME = 'test_aopy'

        sessions = db.lookup_sessions()
        grouped = db.group_entries(sessions) # by date
        self.assertEqual(len(grouped), 1)
        self.assertEqual(len(grouped[0]), len(sessions))

        grouped = db.group_entries(sessions, lambda x: x.duration) # by duration
        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(grouped[0]), len(sessions) - 2) # duration = 0.0
        self.assertEqual(len(grouped[1]), 2) # duration = 3.0

    def test_summarize_entries(self):
            
        sessions = db.lookup_sessions()
        summary = db.summarize_entries(sessions)
        self.assertEqual(len(summary), len(sessions))

    def test_encode_onehot_sequence_name(self):

        sessions = db.lookup_mc_sessions()
        df = db.encode_onehot_sequence_name(sessions, sequence_types=['centerout_2D'])
        self.assertEqual(df.shape[1], 4) 

    def test_add_metadata_columns(self):

        sessions = db.lookup_sessions()
        df = pd.DataFrame({
            'te_id': [e.id for e in sessions],
        })
        db.add_metadata_columns(df, sessions, ['id_copy', 'test'], [lambda e: e.id, lambda e: 'test'])
        self.assertEqual(len(df), len(sessions))
        self.assertEqual(df['id_copy'].sum(), df['te_id'].sum())
        self.assertTrue(all(df['test'] == 'test'))

    def test_get_aba_perturbation_sessions(self):
        sessions = db.lookup_mc_sessions()
        names = db.get_aba_perturbation_sessions(sessions)
        self.assertCountEqual(names, ['a', 'b', 'aprime'])

    def test_get_aba_perturbation_days(self):
        sessions = db.lookup_mc_sessions()
        days, sessions = db.get_aba_perturbation_days(sessions)
        self.assertEqual(len(days), 1)
        self.assertCountEqual(sessions[0], ['a', 'b', 'aprime'])

if __name__ == "__main__":
    unittest.main()
