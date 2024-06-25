# tutorial_functions.py
# This is an example module to practice using git.

# Ryan's practice function
def practice_func_ryan(user_name):
    '''
    This function returns a print statement specific to Ryan from the input username

    Inputs:
        user_name [string]: Name of person running the function
        
    Outputs:
        string_out [string]: Hello Ryan from 
    '''
    string_out = 'Hello Ryan, from' + user_name
    return string_out

# Ryan's practice function 2
def practice_func_ryan2(favorite_food):
    '''
    This function returns a print statement specific to Ryan from the input username

    Inputs:
        favorite_food [string]: Your favorite food
        
    Outputs:
        None; print statement
    '''
    string_out = 'My favorite food is' + favorite_food
    return string_out

# Leo's practice function
def practice_func_leo(number_of_fish):
    '''
    This function prints a number of fish

    Inputs:
        number_of_fish [int]: How many fish you have
        
    Outputs:
        statement [str]: a statement of how many fish you have
    '''
    
    return 'You have {} fish'.format(number_of_fish)


#Pamel's practice function
def pamels_function(num_of_cookies):
    ''' This function prints the number of cookies Pamel has

    Inputs: num_of_cookies [int]: How many cookies Pamel has

    Outputs:
        statement[str]: a statement about my cookies
    '''
    return 'Pamel has {} cookies. She will eat all of them'.format(num_of_cookies)

def practice_function_gus(favorite_food):
    '''
    This function returns a print statement specific to Gus from the input username

    Inputs:
        favorite_food [string]: your favorite food

    Outputs:
        string_out [string]: returns a string saying what your favorite food is
    '''
    string_out = 'my favorite food is ' + favorite_food
    return string_out

  
def practice_func_tomo(user_name):
    '''
    This function returns a print statement specific to Tomo from the input username

    Inputs:
        user_name [string]: Name of person running the function
        
    Outputs:
        string_out [string]: Hello Tomo from 
    '''
    string_out = 'Hello Tomo, from ' + user_name
    return string_out


def practice_func_pavi(my_name):
    '''
    This is just a test function
    Inputs:
      param my_name: [str] enter your name
    
    Output:
      return: returns a statement introducing my_name
    '''

    return 'hello! I am '+ my_name


def practice_func_sijia(favourite_marvel_character):
    '''
    This function prints a number of fish

    Inputs:
        favourite_marvel_character [string ]: which marvel character we'd like to meet
        
    Outputs:
        statement [str]: a greeting statement to the favourite marvel character
    '''
    
    return f'hello, {favourite_marvel_character}'


def display_cat_leo():
    '''
    Cat

    Inputs:
        None

    Outputs:
        cat [str]: a cat
    '''
    cat = ' /\\_/\\\n( o.o )\n > ^ <'
    for t in range(3):
        print(cat)
    return cat

def practice_func_miken(your_number):
    '''
    This function tests if your number is the best number

    Inputs:
        your_number [num]: your number, any number.

    Outputs:
        result [bool]: True if your number is the best number, otherwise false.
    '''

    result = your_number == 7
    return result 


def practice_function_gus(favorite_food):
    '''
    This function returns a print statement specific to Gus from the input username

    Inputs:
        favorite_food [string]: your favorite food

    Outputs:
        string_out [string]: returns a string saying what your favorite food is
    '''
    string_out = 'my favorite food is ' + favorite_food
    return string_out

def practice_function_matt(favorite_number):
    '''
    This function judges your favorite number from the user input.

    Args:
        favorite_number (int): your favorite number

    Returns: 
        string: a judgmental string about your fav number
    '''

    if favorite_number != 7:
        result = 'Bad choice'
    else:
        result = 'Wow me too!'
    
    return result