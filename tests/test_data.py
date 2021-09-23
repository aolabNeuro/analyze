# test_data.py 
# tests of aopy.data
from aopy.data import _process_channels
from aopy.data import *
from aopy.visualization import *
import unittest
import os
import numpy as np
import pandas as pd
from matplotlib.testing.compare import compare_images

test_dir = os.path.dirname(__file__)
data_dir = os.path.join(test_dir, 'data')
write_dir = os.path.join(test_dir, 'tmp')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)
test_filepath = os.path.join(data_dir, "short headstage test")
sim_filepath = os.path.join(data_dir, "fake ecube data")

class LoadDataTests(unittest.TestCase):

    def test_get_filenames_in_dir(self):
        test_dir = os.path.join(data_dir, 'test_filenames_dir')
        files = get_filenames_in_dir(test_dir, 1039)
        self.assertIn('foo', files)
        self.assertIn('bar', files)
        self.assertEqual(files['foo'], os.path.join('foo','1039_foo'))
        self.assertEqual(files['bar'], os.path.join('bar','1039_bar.txt'))

    def test_load_mocap(self):
        # Data directory and filepath
        filename = 'Take 2021-03-10 17_56_55 (1039).csv'

        # Load Meta data
        mocap_metadata = load_optitrack_metadata(data_dir, filename)
        assert mocap_metadata['samplerate'] == 240
        assert mocap_metadata['Rigid Body Name'] == 'Hand'
        assert mocap_metadata['Length Units'] == 'Meters'
        
        # Load Data
        mocap_data_pos, mocap_data_rot = load_optitrack_data(data_dir, filename)
        assert mocap_data_pos.shape[0] == 92383
        assert mocap_data_rot.shape[0] == 92383
        assert mocap_data_pos.shape[1] == 3
        assert mocap_data_rot.shape[1] == 4

        # Load timestamps
        mocap_timestamps = load_optitrack_time(data_dir, filename)
        assert mocap_timestamps.shape[0] == 92383
        assert mocap_timestamps.ndim == 1
        diff = np.diff(mocap_timestamps)
        assert (diff > 0).all(), 'Should be monotonically increasing'

    def test_get_ecube_data_sources(self):
        sources = get_ecube_data_sources(test_filepath)
        assert len(sources) == 1
        assert sources[0] == "Headstages"

    def test_load_ecube_metadata(self):
        metadata = load_ecube_metadata(test_filepath, 'Headstages')
        assert metadata['n_channels'] == 64
        assert metadata['n_samples'] == 214032
        assert metadata['data_source'] == 'Headstages'
        assert metadata['samplerate'] == 25000.

    def test_process_channels(self):
        metadata = load_ecube_metadata(test_filepath, 'Headstages')
        data = _process_channels(test_filepath, 'Headstages', [0], metadata['n_samples'], 'uint16')
        assert data.shape[1] == 1
        assert data.shape[0] == 214032

    def test_load_ecube_metadata(self):
        metadata = load_ecube_metadata(test_filepath, 'Headstages')
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
        data = load_ecube_data(sim_filepath, 'Headstages')
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
        data = load_ecube_data(test_filepath, 'Headstages')
        assert data.shape[1] == 64
        assert data.shape[0] == 214032

        # Test that channels work
        data = load_ecube_data(test_filepath, 'Headstages', channels=[4])
        assert data.shape[1] == 1
        assert data.shape[0] == 214032

    def test_proc_ecube_data(self):
        import os
        import h5py
        hdf_filepath = os.path.join(write_dir, "preprocessed_ecube_data.hdf")
        if os.path.exists(hdf_filepath):
            os.remove(hdf_filepath)
        proc_ecube_data(test_filepath, 'Headstages', hdf_filepath)
        assert os.path.exists(hdf_filepath)
        hdf = h5py.File(hdf_filepath, 'r')
        assert 'Headstages' in hdf
        assert hdf['Headstages'].attrs['samplerate'] == 25000.
        assert hdf['Headstages'].shape[1] == 64
        assert hdf['Headstages'].shape[0] == 214032
    
    def test_save_hdf(self):
        import os
        import h5py
        testfile = 'save_hdf_test.hdf'
        testpath = os.path.join(write_dir, testfile)
        if os.path.exists(testpath):
            os.remove(testpath)
        data = {'test_data_array': np.arange(1000)}
        params = {'key1': 'value1', 'key2': 2}
        save_hdf(write_dir, testfile, data_dict=data, data_group="/", append=False)
        save_hdf(write_dir, testfile, data_dict=data, data_group="/test_data", append=True)
        save_hdf(write_dir, testfile, data_dict=params, data_group="/test_metadata", append=True)
        f = h5py.File(testpath, 'r')
        self.assertIn('test_data_array', f)
        self.assertIn('test_data', f)
        self.assertIn('test_metadata', f)
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

    def test_load_bmi3d_hdf_table(self):
        testfile = 'test20210330_12_te1254.hdf'
        data, metadata = load_bmi3d_hdf_table(data_dir, testfile, 'task')
        self.assertEqual(len(data), 534)
        self.assertEqual(len(metadata.keys()), 35)

        data, metadata = load_bmi3d_hdf_table(data_dir, testfile, 'sync_clock')
        self.assertEqual(len(data), 534)
        data, metadata = load_bmi3d_hdf_table(data_dir, testfile, 'sync_events')
        self.assertEqual(len(data), 6)

        self.assertRaises(ValueError, lambda: load_bmi3d_hdf_table(data_dir, testfile, 'nonexistent_table'))

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



if __name__ == "__main__":
    unittest.main()



