import aopy
import unittest

class TutorialTests(unittest.TestCase):

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'You have 3 fish')

    def test_gus_practice(self):
        output = aopy.tutorial_functions.practice_function_gus('chicken')
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'my favorite food is chicken')

if __name__ == "__main__":
    unittest.main()