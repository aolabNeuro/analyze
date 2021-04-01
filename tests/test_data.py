# test_data.py 
# tests of aopy.data
from aopy.data import *
import unittest

data_dir = 'tests/data/'
filename = 'Take 2021-03-10 17_56_55 (1039).csv'
test_filepath = "tests/data/short headstage test"

class LoadDataTests(unittest.TestCase):
    def test_load_mocap(self):
        # Data directory and filepath

        # Load Meta data
        mocap_metadata = load_optitrack_metadata(data_dir, filename)
        print(mocap_metadata)
        assert mocap_metadata['samplerate'] == 240
        assert mocap_metadata['Rigid Body Name'] == 'Hand'
        assert mocap_metadata['Length Units'] == 'Meters'
        
        # Load Data
        mocap_data_pos, mocap_data_rot = load_optitrack_data(data_dir, filename)
        print(mocap_data_pos.shape)
        print(mocap_data_rot.shape)
        assert mocap_data_pos.shape[0] == 92383
        assert mocap_data_rot.shape[0] == 92383
        assert mocap_data_pos.shape[1] == 3
        assert mocap_data_rot.shape[1] == 4

    def test_get_sources(self):
        sources = get_eCube_data_sources(test_filepath)
        print(sources)
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
        hdf_filename = os.path.join(test_filepath, "preprocessed.hdf")
        if os.path.exists(hdf_filename):
            os.remove(hdf_filename)
        proc_ecube_data(test_filepath, 'Headstages', hdf_filename)
        assert os.path.exists(hdf_filename)
        hdf = h5py.File(hdf_filename, 'r')
        assert 'Headstages' in hdf
        assert hdf['Headstages'].attrs['samplerate'] == 25000.
        assert hdf['Headstages'].shape[0] == 64
        assert hdf['Headstages'].shape[1] == 214032
    
    
if __name__ == "__main__":
    unittest.main()
