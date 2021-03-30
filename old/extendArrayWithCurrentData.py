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
# arrayIndex = the column (or row) index that will be duplicated
# newColSize = the number of columns the modified array should have
# duplicateColumns = defaults to true for duplicating columns, false for duplicating rows
#
# Outputs:
# newArray = the original Array with newColSize number of columns with duplicated values
#
# Author: Seth Richards
# Version Date 2020/02/02
import numpy as np

def extendArrayWithCurrentData(array, arrayIndex, newColSize,duplicateColumns = True):


    #initilizes array framework
    if duplicateColumns:
        newArray = np.empty([array.shape[0], newColSize])
    else:
        newArray = np.empty([newColSize, array.shape[0]])

    #iterates and duplicates values through new array using values from original array
    for i in range(newColSize):

        # duplicates columns
        if duplicateColumns:  
            if len(array.shape) == 1:
                newArray[:,i] = np.around(array[:],10)
            else:
                newArray[:, i] = np.around(array[:,arrayIndex], 10)

        # duplicates rows
        else:  
            if len(array.shape) == 1:
                newArray[i] = np.around(array[:],10)
            else:
                newArray[i] = np.around(array[arrayIndex, :], 10)

    return newArray


