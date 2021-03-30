import aopy
import unittest

class TutorialTests(unittest.TestCase):

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'You have 3 fish')


    def pamels_function(self):
        output = aopy.tutorial_functions.practice_func_leo(10)
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'Pamel has 10 cookies. She will eat all of them')

if __name__ == "__main__":
    unittest.main()