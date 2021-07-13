import unittest
from aopy.utils import *
import numpy as np


class utils_test(unittest.TestCase):
    def pkl_test(self):
        test_dir = os.path.dirname(__file__)
        tmp_dir = os.path.join(test_dir, 'tmp')

         # Testing pkl_write with one variable
        val = np.random.rand(10,10)
        pkl_write('pickle_write_test1.dat', val, tmp_dir)

        # Testing pkl_write with multiple variables
        v1 = np.random.rand(10,10)
        v2 = np.random.randint(3,2)
        v3 = ['just', 'a', 'test']
        pkl_write('pickle_write_test2.dat', [v1,v2,v3], tmp_dir)

        # Testing pkl_read
        dat_1 = pkl_read('pickle_write_test1.dat', tmp_dir)
        dat_2 = pkl_read('pickle_write_test2.dat', tmp_dir)
        self.assertEqual(val, dat_1)
        self.assertEqual(dat_2,[v1,v2,v3])