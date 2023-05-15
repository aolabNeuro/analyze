import time
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

        ts, values = detect_edges(test_bool, 1)
        self.assertEqual(len(ts), 6)
        np.testing.assert_allclose(ts, [1, 3, 5, 6, 8, 11])
        np.testing.assert_allclose(values, [0, 1, 0, 1, 0, 1])

        # check rising edges only
        ts, values = detect_edges(test_bool, 1, falling=False)
        self.assertEqual(len(ts), 3)
        np.testing.assert_allclose(ts, [3, 6, 11])
        np.testing.assert_allclose(values, [1, 1, 1])
        
        # check falling edges only
        ts, values = detect_edges(test_bool, 1, rising=False)
        self.assertEqual(len(ts), 3)
        np.testing.assert_allclose(ts, [1, 5, 8])
        np.testing.assert_allclose(values, [0, 0, 0])

        # check numeric boolean data
        ts, values = detect_edges(test_01, 1, falling=False)
        self.assertEqual(len(ts), 6)
        np.testing.assert_allclose(ts, [7, 13, 15, 17, 19, 21])
        np.testing.assert_allclose(values, [1, 1, 1, 1, 1, 1])

        # check data values instead of a single bit
        test_02 = [0b11, 0, 0, 0, 0, 0, 0, 0b01, 0b10, 0b01, 0b11, 0b01, 0b00, 0b10, 0b00, 0b01, 0, 0b01, 0, 0b01, 0, 0b01, 0]
        ts, values = detect_edges(test_02, 1, falling=False, check_alternating=False)
        self.assertEqual(len(ts), 9)
        np.testing.assert_allclose(ts, [7, 8, 9, 10, 13, 15, 17, 19, 21])
        np.testing.assert_allclose(values, [1, 2, 1, 3, 2, 1, 1, 1, 1])

        ts, values = detect_edges(test_02, 1, falling=False)
        self.assertEqual(len(ts), 8)
        np.testing.assert_allclose(ts, [7, 8, 10, 13, 15, 17, 19, 21])
        np.testing.assert_allclose(values, [1, 2, 3, 2, 1, 1, 1, 1])

        # test that if there are multiple of the same edge only the last one counts
        test_valid = [0, 0, 3, 0, 3, 2, 2, 0, 1, 7, 3, 2, 2, 0]
        ts, values = detect_edges(test_valid, 1)
        np.testing.assert_allclose(ts, [2, 3, 4, 7, 9, 13])
        np.testing.assert_allclose(values, [3, 0, 3, 0, 7, 0])

        # Test using min_pulse_width
        test_bool = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0] # Note the first 1 doesn't count as an edge!
        ts, values = detect_edges(test_bool, 1, min_pulse_width=4)
        np.testing.assert_allclose(ts,  [1, 2, 8])  # Can't get falling edge without rising edge in this case
        np.testing.assert_allclose(values, [0, 1, 0])
        
        test_01 = [0, 0, 4, 0, 3, 2, 0, 0, 0, 0, 0, 0]
        ts, values = detect_edges(test_01, 1, min_pulse_width=4)
        np.testing.assert_allclose(ts, [4, 8]) # the rising edge isn't finished until index 4 now
        np.testing.assert_allclose(values, [7, 0])

        test_02 = [0, 0, 4, 0, 3, 2, 0, 2, 0, 0, 0, 0]
        ts, values = detect_edges(test_02, 1, min_pulse_width=4)
        np.testing.assert_allclose(ts, [4, 11]) # falling edge extended to 11
        np.testing.assert_allclose(values, [7, 0])

        test_03 = [0, 0, 4, 0, 3, 2, 0, 0, 0, 2, 0, 0]
        ts, values = detect_edges(test_03, 1, min_pulse_width=4)
        np.testing.assert_allclose(ts, [4, 8, 9]) # new rising edge at the end
        np.testing.assert_allclose(values, [7, 0, 2])    

    def test_get_pulse_edge_times(self):
        # see test_data:E3vFrameTests
        pass

    def test_compute_pulse_duty_cycles(self):
        # see test_data:E3vFrameTests
        pass

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

    def test_channels_to_digital(self):
        testdata = [0b0110100001011100101100000100011001010001101110101001000100111000, 0xff0ff00fffffff0f]
        unpacked = convert_digital_to_channels(testdata)
        packed = convert_channels_to_digital(unpacked)
        np.testing.assert_allclose(testdata, packed)

    def test_copy_edges_forwards(self):
        print("")
        test_edges = np.array(
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        )
        expected_edges = np.array([
            [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        ])
        out = copy_edges_forwards(test_edges, 1, truncate_edges=True)
        np.testing.assert_allclose(out, expected_edges[0])

        out = copy_edges_forwards(test_edges, 2, truncate_edges=True)
        np.testing.assert_allclose(out, expected_edges[1])

        out = copy_edges_forwards(test_edges, 3, truncate_edges=True)
        np.testing.assert_allclose(out, expected_edges[2])

        # Test with 2D array
        test_edges = np.array([
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        ])
        expected_edges = np.array([
            [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        ])
        out = copy_edges_forwards(test_edges, 1, axis=1, truncate_edges=True)
        print(out.astype(int))
        np.testing.assert_allclose(out, expected_edges)

        out = copy_edges_forwards(test_edges.T, 1, axis=0, truncate_edges=True) # check axis=0 as well
        print(out.astype(int))
        np.testing.assert_allclose(out, expected_edges.T)

        # Test with no truncation
        test_edges = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0])
        expected_edges = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        out = copy_edges_forwards(test_edges, 1, truncate_edges=False)
        print(out.astype(int))
        np.testing.assert_allclose(out, expected_edges)

        test_edges = np.array([
            [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        ])
        expected_edges = np.array([
            [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        ])
        out = copy_edges_forwards(test_edges, 1, axis=1, truncate_edges=False)
        print(out.astype(int))
        np.testing.assert_allclose(out, expected_edges)

        out = copy_edges_forwards(test_edges.T, 1, axis=0, truncate_edges=False) # check axis=0 as well
        print(out.astype(int))
        np.testing.assert_allclose(out, expected_edges.T)

        # Test super long sequence with lots of edges
        test_edges = np.random.randint(10000, size=(25000*60,10)) < 1
        n_edges = np.count_nonzero(test_edges)
        n_steps = int(0.003*25000) # 3ms pulse at 25khz

        t0 = time.perf_counter()
        out = copy_edges_forwards(test_edges, n_steps, truncate_edges=False, copy_per_step=True)
        t1 = time.perf_counter()
        print(f"Copy-per-step method takes {t1-t0:0.2f} seconds on {n_edges} edges")

        t0 = time.perf_counter()
        out = copy_edges_forwards(test_edges, n_steps, truncate_edges=False, copy_per_step=False)
        t1 = time.perf_counter()
        print(f"Default method takes {t1-t0:0.2f} seconds on {n_edges} edges")

    def test_count_repetitions(self):
        arr1 = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5]
        arr2 = [1, 1, 1, 1]
        arr3 = [0]
        arr4 = []
        arr5 = [0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.5, 0.5]
        
        np.testing.assert_array_equal(count_repetitions(arr1), (np.array([1, 2, 3, 2, 4]), np.array([0, 1, 3, 6, 8])))
        np.testing.assert_array_equal(count_repetitions(arr2), (np.array([4]), np.array([0])))
        np.testing.assert_array_equal(count_repetitions(arr3), (np.array([1]), np.array([0])))
        np.testing.assert_array_equal(count_repetitions(arr4), (np.array([]), np.array([])))
        np.testing.assert_array_equal(count_repetitions(arr5, diff_thr=0.05), 
                                      (np.array([1, 2, 3, 1, 2]), np.array([0, 1, 3, 6, 7])))
        
    def test_segment_array(self):
        arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        categories = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5] 
        expected_segments = [[0], [1, 2], [3, 4, 5], [6, 7], [8, 9, 10, 11]]
        expected_categories = [1, 2, 3, 4, 5]
        
        segments, cats = segment_array(arr, categories)
        self.assertEqual(len(segments), len(expected_categories))
        for i in range(len(segments)):
            np.testing.assert_array_equal(segments[i], expected_segments[i])
        np.testing.assert_array_equal(cats, expected_categories)

        # With duplicate_endpoints = True
        arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        categories = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5] 
        expected_segments = [[0], [0, 1, 2], [2, 3, 4, 5], [5, 6, 7], [7, 8, 9, 10, 11]]
        expected_categories = [1, 2, 3, 4, 5]
        
        segments, cats = segment_array(arr, categories, duplicate_endpoints=True)
        print(segments)
        self.assertEqual(len(segments), len(expected_categories))
        for i in range(len(segments)):
            np.testing.assert_array_equal(segments[i], expected_segments[i])
        np.testing.assert_array_equal(cats, expected_categories)

        # Multidimensional
        arr = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [2, 2, 0],
            [3, 3, 0],
            [4, 2, 0]
        ])
        categories = [0, 0, 1, 1, 1] 
        expected_segments = [
            np.array([[0, 0, 0],
                      [1, 1, 0]]), 
            np.array([[1, 1, 0],
                      [2, 2, 0],
                      [3, 3, 0],
                      [4, 2, 0]])
        ] 
        expected_categories = [0, 1]
        segments, cats = segment_array(arr, categories, duplicate_endpoints=True)
        self.assertEqual(len(segments), len(expected_categories))
        for i in range(len(segments)):
            np.testing.assert_array_equal(segments[i], expected_segments[i])
            print('is the same?')
            print(segments[i], expected_segments[i])
        np.testing.assert_array_equal(cats, expected_categories)


class TestMath(unittest.TestCase):

    def test_derivative(self):
        # Test 1D case 
        x = np.linspace(0,10,1000)
        y = x**2
        dydx = derivative(x, y, norm=False)
        expected = x*2
        np.testing.assert_allclose(dydx, expected)

        # Test 2D case with component-wise derivative
        y = np.array([x**2, x**2]).T
        dydx = derivative(x, y, norm=False)
        expected = np.array([x*2, x*2]).T
        np.testing.assert_allclose(dydx, expected)

        # Test 2D case with norm derivative
        x = np.linspace(0, 10, 1000)
        y = np.array([x*2, x*2]).T
        dydx = derivative(x, y, norm=True)
        expected = np.ones(1000)*2*np.sqrt(2)
        np.testing.assert_allclose(dydx, expected)

    def test_calc_euclid_dist_mat(self):
        pos = np.array(
            [[1,1],
            [2,2],
            [-1,1]]
        )
        dist_mat = calc_euclid_dist_mat(pos)
        self.assertEqual(dist_mat.shape, (3, 3))
        self.assertEqual(np.min(dist_mat), 0)
        self.assertEqual(np.max(dist_mat), np.sqrt(3**2 + 1)) # between [2,2] and [-1,1]

    def test_calc_radial_dist(self):

        pos = np.array(
            [[1,1],
            [2,2],
            [-1,1]]
        )
        origin = np.array([1,1])
        dist = calc_radial_dist(pos, origin)
        self.assertEqual(dist.size, 3)
        np.testing.assert_allclose(dist, [0, np.sqrt(2), 2])

    def test_first_nonzero(self):
        p = np.array([0, 0, 1])
        q = first_nonzero(p, axis=0, all_zeros_val=-1)

        self.assertEqual(q, 2)

        p = np.array([1])
        q = first_nonzero(p, axis=0, all_zeros_val=-1)

        self.assertEqual(q, 0)

        p = np.array([0])
        q = first_nonzero(p, axis=0, all_zeros_val=-1)

        self.assertEqual(q, -1)

class MemoryTests(unittest.TestCase):

    def test_get_memory_available(self):

        avail = get_memory_available_gb()
        print(f"available memory: {avail} GB")

    def test_get_set_release_memory_limit(self):

        release_memory_limit()
        set_memory_limit_gb(1)
        
        if get_memory_limit_gb() == 1:
            self.assertRaises(MemoryError, lambda: np.ones((1000000,100))) # ~800 MB

        release_memory_limit()

        a = np.ones((1000000,100))
        print(f"allocated {a.nbytes/1e9} GB")

class TestSynchronization(unittest.TestCase):
    
    def test_get_first_last_times(self):
        barcode_ontimes = np.array([35.2, 66.3, 95.2, 125.1, 156.5, 186.4])
        barcode_ontimes_main = np.array([65.1, 96.3, 124.5, 155.2])
        barcode = [14256, 13556, 32364, 23425, 43525, 23534]
        barcode_main = [13556, 32364, 23425, 43525]

        first_last_times,first_last_times_main = get_first_last_times(barcode_ontimes, barcode_ontimes_main, barcode, barcode_main)
        self.assertEqual(first_last_times.shape[0], 2)
        self.assertEqual(first_last_times_main.shape[0], 2)
        
    def test_sync_timestamp_offline(self):
        timestamps = np.arange(100)
        on_times = [10,90] # the first and last times sync pulses come 
        on_times_main = [12,93] # the first and last times sync pulses come to the main stream
        sync_timestamp,scaling = sync_timestamp_offline(timestamps, on_times, on_times_main)
        
        # if synchronization works, on_times in the first stream are converted to on_times_main in the main stream
        self.assertEqual(sync_timestamp[10], 12)
        self.assertEqual(sync_timestamp[90], 93)
        self.assertEqual(scaling, (93-12)/(90-10))
        
if __name__ == "__main__":
    unittest.main()
