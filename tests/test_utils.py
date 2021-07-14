import unittest
from aopy.utils import *
import numpy as np


class utils_test(unittest.TestCase):

    def test_pkl_fn(self):
        test_dir = os.path.dirname(__file__)
        tmp_dir = os.path.join(test_dir, 'tmp')

         # Testing pkl_write
        val = np.random.rand(10,10)
        pkl_write('pickle_write_test.dat', val, tmp_dir)

        # Testing pkl_read
        dat_1 = pkl_read('pickle_write_test.dat', tmp_dir)

        self.assertEqual(np.shape(val), np.shape(dat_1))

if __name__ == "__main__":
    unittest.main()