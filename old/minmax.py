
import numpy as np


def minmax(array):

    # Minmax - finds the minimum and maximum of a 2d array
    #
    # Input is X-by-Y array, output is X-by-2 array
    #
    # Meant to mirror MATLAB's minmax function below:
    # https://www.mathworks.com/help/deeplearning/ref/minmax.html

    # Author: Seth Richards
    # Version Date: 2020/05/20

    if array.ndim != 2:
        raise Exception("Array must be of dimension 2")

    solutionArray = np.empty([array.shape[0], 2])

    solutionArray[:, 0] = array.min(axis=1)
    solutionArray[:, 1] = array.max(axis=1)

    return solutionArray
