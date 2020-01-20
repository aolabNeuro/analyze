import numpy as np

# extendArrayWithCurrentData(array, newColSize) = newArray
#
# This function is meant to mirror the matlab functionality by which any array
# when passed an array of ones to scale array size by (see example)
# , array will duplicate columns to meet intended length
# Ex.
# newArray = oldArray(:,ones(1,x),y)
# newArray will be extended in number of columns to x
# populated with duplicated column values
#
# Not sure if a python optimized equivalent exists, please correct if possible
#
# Inputs:
# array = array desired to be modified
# newColSize = the number of columns the modified array should have
#
# Outputs:
# newArray = the original Array with newColSize number of columns with duplicated values
#
# Author: Seth Richards
# Version Date 2019/12/03

def extendArrayWithCurrentData(array,newColSize):
    
    i = 0
    
    #initilizes array framework
    newArray = np.empty([array.shape[0], newColSize])

    #iterates and duplicates values through new array using values from original array
    for i in range(newColSize):
        newArray[:,newColSize - i - 1] = array[:]

    return newArray



