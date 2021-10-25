from aopy.utils import *
from aopy.visualization import plot_timeseries, savefig
import os
import numpy as np
import unittest

test_dir = os.path.dirname(__file__)
write_dir = os.path.join(test_dir, 'tmp')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)


class FakeDataTests(unittest.TestCase):

    def test_gen_save_test_signal(self):

        # Generate a signal
        samplerate = 25000
        data = generate_multichannel_test_signal(1, samplerate, 8, 6, 1)

        self.assertEqual(data.shape, (25000, 8))

        # Pack it into bits
        voltsperbit = 1e-4
        base_dir = os.path.join(write_dir, 'test_ecube_data')
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        filename = save_test_signal_ecube(data, base_dir, voltsperbit)

        self.assertTrue('Headstages' in filename)

        plot_timeseries(data, samplerate)
        figname = 'gen_test_signal.png'
        savefig(write_dir, figname)

    def test_generate_test_signal(self):

        x, t = generate_test_signal(1, 1000, [1.2, 100, 250], [0.2, 1.5, 1])
        self.assertEqual(len(x), len(t))
        self.assertEqual(len(x), 1000)

class TestSymbols(unittest.TestCase):

    def test_count_unique_symbols(self):
        test_file = os.path.join(write_dir, 'test_symbols.txt')
        with open(test_file, 'w') as f:
            f.write("variable foo\nvariable bar\nvariable foo")
        symbols, counts = count_unique_symbols([test_file])
        self.assertCountEqual(symbols, ['foo', 'bar'])
        self.assertCountEqual(counts, [2, 1])

class TestDigitalCalc(unittest.TestCase):

    def test_get_edges_from_onsets(self):
        onsets = np.array([1, 1.5, 2])
        expected_timestamps = np.array([0, 1, 1.1, 1.5, 1.6, 2, 2.1])
        expected_values = np.array([0,1,0,1,0,1,0])

        timestamps, values = get_edges_from_onsets(onsets, 0.1)
        np.testing.assert_allclose(timestamps, expected_timestamps)
        np.testing.assert_allclose(values, expected_values)

    def test_convert_analog_to_digital(self):

        #data_length = 3
        #nchan = 2
        analog_data = [[2.9, 0.23], 
                        [1.9, 2.9], 
                        [1.74, 4.76]]
        thresh = 0.5
        expected_digital_array = [[1.0, 0.0], 
                                    [0.0, 1.0], 
                                    [0.0, 1.0]]
        digital_data = convert_analog_to_digital(analog_data, thresh)
        np.testing.assert_almost_equal(expected_digital_array, digital_data)

    def test_detect_edges(self):
        test_bool = [True, False, False, True, True, False, True, True, False, False, False, True]
        test_01 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        test_02 = [0b11, 0, 0, 0, 0, 0, 0, 0b01, 0b10, 0b01, 0b11, 0b01, 0b00, 0b10, 0b00, 0b01, 0, 0b01, 0, 0b01, 0, 0b01, 0]
        test_valid = [0, 0, 3, 0, 3, 2, 2, 0, 1, 7, 3, 2, 2, 0]

        ts, values = detect_edges(test_bool, 1)
        self.assertEqual(len(ts), 6)
        self.assertTrue(np.array_equal(ts, [1, 3, 5, 6, 8, 11]))
        self.assertTrue(np.array_equal(values, [0, 1, 0, 1, 0, 1]))

        # check rising edges only
        ts, values = detect_edges(test_bool, 1, falling=False)
        self.assertEqual(len(ts), 3)
        self.assertTrue(np.array_equal(ts, [3, 6, 11]))
        self.assertTrue(np.array_equal(values, [1, 1, 1]))
        
        # check falling edges only
        ts, values = detect_edges(test_bool, 1, rising=False)
        self.assertEqual(len(ts), 3)
        self.assertTrue(np.array_equal(ts, [1, 5, 8]))
        self.assertTrue(np.array_equal(values, [0, 0, 0]))

        # check numeric boolean data
        ts, values = detect_edges(test_01, 1, falling=False)
        self.assertEqual(len(ts), 6)
        self.assertTrue(np.array_equal(ts, [7, 13, 15, 17, 19, 21]))
        self.assertTrue(np.array_equal(values, [1, 1, 1, 1, 1, 1]))

        # check data values instead of a single bit
        ts, values = detect_edges(test_02, 1, falling=False, check_alternating=False)
        self.assertEqual(len(ts), 9)
        self.assertTrue(np.array_equal(ts, [7, 8, 9, 10, 13, 15, 17, 19, 21]))
        self.assertTrue(np.array_equal(values, [1, 2, 1, 3, 2, 1, 1, 1, 1]))

        # test that if there are multiple of the same edge only the last one counts
        ts, values = detect_edges(test_valid, 1)
        self.assertTrue(np.array_equal(ts, [2, 3, 4, 7, 9, 13]))
        self.assertTrue(np.array_equal(values, [3, 0, 3, 0, 7, 0]))

    def test_find_first_significant_bit(self):
        data = 0b0100
        ffs = find_first_significant_bit(data)
        self.assertEqual(ffs, 2)
        data = 0
        ffs = find_first_significant_bit(data)
        self.assertEqual(ffs, -1)

    def test_mask_and_shift(self):
        mask = 0x0000000000ff0000 # bits 17-24
        digital_data = [2, 4, 12*0x10000, 140*0x10000, 0xff0000, 0xff0000]
        masked = mask_and_shift(digital_data, mask)
        assert np.array_equal(masked, [0, 0, 12, 140, 255, 255])

    def test_convert_channels_to_mask(self):
        self.assertEqual(convert_channels_to_mask(0), 1)
        self.assertEqual(convert_channels_to_mask(5), 1 << 5)
        channels = [0,1]
        mask = convert_channels_to_mask(channels)
        self.assertEqual(mask, 0b11)
        channels = range(16,24)
        mask = convert_channels_to_mask(channels)
        self.assertEqual(mask, 0xff0000)

    def test_digital_to_channels(self):
        testdata = [0b0110100001011100101100000100011001010001101110101001000100111000, 0xff0ff00fffffff0f]
        unpacked = convert_digital_to_channels(testdata)
        packed = np.packbits(unpacked, bitorder='little').view(np.uint64)
        assert packed[0] == testdata[0]
        assert packed[1] == testdata[1]
        assert unpacked[0,0] == 0
        assert unpacked[1,0] == 1
        assert unpacked[0,1] == 0

if __name__ == "__main__":
    unittest.main()
