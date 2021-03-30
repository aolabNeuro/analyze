'''
This is an example module to practice using git
'''

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
