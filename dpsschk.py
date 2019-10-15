import numpy as np
import math
import spectrum


def dpsschk(tapers):
   # DPSSCHK Helper function for multitaper routines.
   #
   # Inputs:
   #   E = DPSSCHK(TAPERS) returns the DPSS array based on the input
   #     TAPERS.  If TAPERS is a cell containing the DPSS array then E
   #     just returns the array.
   #     If TAPERS is in the form [N, NW, K] then E contains the corresponding
   #     sequences.
   #
   # Outputs:
   #   [E, V] = DPSSCHK(TAPERS) returns the eigenvalues as well as the
   #      sequences.
   #
   # Uses spectrum library function to find eigenvalues and sequences
   # See: http://thomas-cokelaer.info/software/spectrum/html/user/ref_mtm.html
   #
   # Origninal MATLAB Code .py adapted from:
   # Author: B. Pesaran, 03-12-98
   #
   # Author: Seth Richards
   # Version Date: 2019/10/14
   #
   # One translation issue: exception raised for matlab cell array; doesn't translate to np
   #

   #initilizes variables
   e = 0
   v = 0

   flag = 0

   sz = max(np.shape(tapers))
   if sz == 3:
       flag = True
   if flag:
       N = tapers[0]               # desired window length
       NW = tapers[1]              # time half bandwidth parameter
       K = tapers[2]               # slepian sequence (first)
       if np.remainder(N, 1):
           N = math.round(N)  # deal with rounding errors

       if K < 1:
           raise Exception('Error:  K must be greater than or equal to 1')

       if K < 3:
           print('Warning:  K is less than 3')

       if K > 2 * NW - 1:
           raise Exception('Error:  K must be less than 2*P-1')

       e, v = spectrum.dpss(N, NW, K)

   if not flag:
       print('Tapers already calculated')
       e = tapers
       v = 0


#Assumed that tapers isn't a cell array, didn't find example, so not quite sure what to filter
#
 #  if iscell(tapers):
  #   e = tapers[0][0]
   #  v = 0
    # if tapers.shape[1] == 2:
     #  v = tapers[0][1]


   return e,v


