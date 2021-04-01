import aopy
import unittest

class TutorialTests(unittest.TestCase):

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'You have 3 fish')

    def test_pamels_function(self):
        output = aopy.tutorial_functions.pamels_function(10)
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'Pamel has 10 cookies. She will eat all of them')

    def test_gus_practice(self):
        output = aopy.tutorial_functions.practice_function_gus('chicken')
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'my favorite food is chicken')

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

    def test_sijia_practice(self):
        output = aopy.tutorial_functions.practice_func_sijia('Cap')
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'hello, Cap')


    def test_tomo_practice(self):
        output = aopy.tutorial_functions.practice_func_tomo('ryan')
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'Hello Tomo, from ryan')

    def test_pavi(self):
        output = aopy.tutorial_functions.practice_func_pavi('pavi')
        self.assertEqual(type(output), str)
        self.assertEqual(output, 'hello! I am pavi')

    def test_gus(self):
        ''' This tests tutorial_function.py '''
        favorite_food = 'chicken'
        aopy.tutorial_functions.practice_function_gus(favorite_food)

if __name__ == "__main__":
    unittest.main()
