import aopy
import unittest

class TutorialTests(unittest.TestCase):

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'You have 3 fish')

    def test_amy_practice(self):
        output = aopy.tutorial_functions.practice_func_amy(1)
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'You have 1 dog(s).')

if __name__ == "__main__":
    unittest.main()
