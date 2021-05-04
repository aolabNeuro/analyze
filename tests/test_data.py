# test_data.py 
# tests of aopy.data
from aopy.data import *
import unittest
import os

data_dir = 'tests/data/'
write_dir = 'tests/tmp'
if not os.path.exists(write_dir):
    os.mkdir(write_dir)
filename = 'Take 2021-03-10 17_56_55 (1039).csv'
test_filepath = "tests/data/short headstage test"

class LoadDataTests(unittest.TestCase):

    def test_get_filenames_in_dir(self):
        files = get_filenames_in_dir(data_dir, 1039)
        self.assertIn('optitrack', files)
        self.assertEqual(files['optitrack'], filename)

    def test_load_mocap(self):
        # Data directory and filepath

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
        data = process_channels(test_filepath, 'Headstages', [0], metadata['n_samples'], 'uint16')
        assert data.shape[0] == 1
        assert data.shape[1] == 214032

    def test_load_ecube_metadata(self):
        metadata = load_ecube_metadata(test_filepath, 'Headstages')
        assert metadata['samplerate'] == 25000.
        assert metadata['n_channels'] == 64

    def test_load_ecube_data(self):
        data = load_ecube_data(test_filepath, 'Headstages')
        assert data.shape[0] == 64
        assert data.shape[1] == 214032

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
        assert hdf['Headstages'].shape[0] == 64
        assert hdf['Headstages'].shape[1] == 214032
    
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

    def test_get_hdf_contents(self):
        testfile = 'load_hdf_contents_test.hdf'
        testpath = os.path.join(write_dir, testfile)
        if os.path.exists(testpath):
            os.remove(testpath)
        data_dict = {'test_data': np.arange(1000)}
        save_hdf(write_dir, testfile, data_dict=data_dict, data_group="/", append=False)
        group_data_dict = {'group_data': np.arange(1000)}
        save_hdf(write_dir, testfile, data_dict=group_data_dict, data_group="/group1", append=True)
        result = get_hdf_contents(write_dir, testfile, show_tree=True)
        self.assertIn('test_data', result)
        self.assertIn('group1', result)
        self.assertIn('group_data', result['group1'])

    def test_load_hdf_data(self):
        import os
        import h5py
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
    
if __name__ == "__main__":
    unittest.main()
