import aopy
import unittest

class TutorialTests(unittest.TestCase):

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'You have 3 fish')

    def test_sijia_practice(self):
        output = aopy.tutorial_functions.practice_func_sijia('Cap')
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'hello, Cap')


if __name__ == "__main__":
    unittest.main()