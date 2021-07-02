
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


if __name__ == "__main__":
    unittest.main()
