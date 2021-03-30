import numpy as np

def gausswin(l, a = 2.5):
   # Gausswin(l, a) = L-point
   # This function is meant to mirror the matlab function gausswin, which
   # returns an L-point Gaussian window with width factor alpha.
   #
   # Inputs:
   # l = number of points in gaussian window
   # a = the relative width of the "bell curve" of the gaussian
   #
   # Outputs:
   # lp = the amplitudes of the l-points on the gaussian by each l value
   #
   # See: https://www.mathworks.com/help/signal/ref/gausswin.html
   #
   # Author: Seth Richards
   # Version Date 2019/05/28
  
   o = (l - 1) / (2 * a)
   p = (l - 1) / 2
   n = np.arange(-p, p + 1, 1)
   if o == 0:
	print('Gausswin of L = 1 is 1')
       return 1
   else:
       lp = np.exp((-n * n) / (2 * o * o))
       return lp





