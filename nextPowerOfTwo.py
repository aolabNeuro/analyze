import math

def nextPowerOfTwo(n):
    # next_power_of_2(n) = n
    # This function rounds value of n up to the nearest higher power of 2, and returns that power
    #
    # Inputs:
    # Any single number, no arrays, no negatives
    #
    # Outputs:
    # The next power of 2 from n (returns the power not the number
    #
    # See: https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/
    # Built to mirror MATLAB function nextPow2()
    #
    # Author: Seth Richards
    # Version Date 2019/11/11
    #
    if n < 0:
        raise Exception('This function does not accept negative numbers')
    elif n <= 1:
        return 1
    else:
        n = 2 ** (math.ceil(math.log(n,2)))
        n = math.log(n,2)
    return n

