Wriring tests
=============

To run the test modules individually, call
``> python tests/module_name.py``

When reviewing someone's code, you might want to run all the tests at once using
``python -m unittest discover -s tests``

Write tests as you go
---------------------

For example, let's say you are writing a function to format a string to
read "You have ## fish", where ## is an input to the function. You might
start by writing a function shell something like this, in
``aopy/tutorial_functions.py``:

::

    def practice_func_leo(number_of_fish):
        '''
        This function formats a string to say how many fish you have

        Inputs:
            number_of_fish (int): How many fish you have
            
        Outputs:
            str: a string describing how many fish you have
        '''
        pass

Next, you should write a test function in
``tests/test_tutorial_functions.py``. Name your test function ``test_``
followed by the function name:

::

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
            self.assertEqual(output, 'You have 3 fish')

Since I know what the output should be, without programming anything, I
can set up the test to check that the function works.

Now I can implement the function:

::

    def practice_func_leo(number_of_fish):
        '''
        '''
        return 'You have 3 fish'

... and it passes the tests! But wait! This function completely ignores
the ``number_of_fish`` input and always says you have 3 fish. We need to
add a slightly better test. For example:

::

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
        self.assertEqual(output, 'You have 3 fish')
        
        output = aopy.tutorial_functions.practice_func_leo(0)
        self.assertEqual(output, 'You have 0 fish')

And update the code to pass this test:

::

    def practice_func_leo(number_of_fish):
        '''
        '''
        return 'You have {} fish'.format(number_of_fish)

Of course, this is a contrived example and you wouldn't actually need to
write comprehensive tests on a one-line function.
