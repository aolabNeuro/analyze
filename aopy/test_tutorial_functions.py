import aopy
import unittest

class TutorialTests(unittest.TestCase):

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'You have 3 fish')


    def test_tomo_practice(self):
        output = aopy.tutorial_functions.practice_func_tomo('ryan')
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'Hello Tomo, from ryan')

    def test_pavi(self):
        output = aopy.tutorial_functions.practice_func_pavi('pavi')
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'hello! I am pavi')


if __name__ == "__main__":
    unittest.main()

