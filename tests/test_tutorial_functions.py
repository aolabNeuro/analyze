import aopy
import unittest

class TutorialTests(unittest.TestCase):

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'You have 3 fish')


    def test_display_cat(self):
        output = aopy.tutorial_functions.display_cat_leo()
        self.assertEqual(type(output), str)
        self.assertEqual(output, ' /\\_/\\\n( o.o )\n > ^ <')

    def test_miken_practice(self):
        best_num = 7
        not_best_num = 8
        best_result = aopy.tutorial_functions.practice_func_miken(best_num)
        not_best_result = aopy.tutorial_functions.practice_func_miken(not_best_num)
        self.assertTrue(best_result)
        self.assertTrue(~not_best_result)


if __name__ == "__main__":
    unittest.main()
