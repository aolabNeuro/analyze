
from aopy.utils import *
import os
import numpy as np
import unittest

test_dir = os.path.dirname(__file__)
write_dir = os.path.join(test_dir, 'tmp')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

class TestUtils(unittest.TestCase):

    def test_count_unique_symbols(self):
        test_file = os.path.join(write_dir, 'test_symbols.txt')
        with open(test_file, 'w') as f:
            f.write("variable foo\nvariable bar\nvariable foo")
        symbols, counts = count_unique_symbols([test_file])
        self.assertCountEqual(symbols, ['foo', 'bar'])
        self.assertCountEqual(counts, [2, 1])


if __name__ == "__main__":
    unittest.main()
