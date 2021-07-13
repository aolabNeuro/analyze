import numpy as np
from numpy import linalg, matlib

 # run kalman filter forward

# assume that A, W, H, Q, Y and Xo are numpy arrays
# dimensions:
# A - state-transition matrix for the KF [#states x #states];
# W - state-transition covariance matrix [#states x #states];
# H - observation model matrix [#observations x #states];
# Q - observation model covariance matrix [#observations x #observations];
# Y - Observation data (e.g. spike firing rates) [#observations x time]
# Xo - initial state (i.e. state at t=0) [#states x 1];
def runKalmanForward(A, W, H, Q, Y, Xo):

  # get the size of the observations, states and time
  (n_obs, n_time) = Y.shape
  n_states = A.shape[0]

  # initialize matrices to store estimates
  X = np.zeros( (n_states, n_time) ) # predicted states
  X_hat = np.zeros( (n_states, n_time) ) # a-priori estimate of state
  P_AP = np.zeros( (n_states, n_states, n_time) ) #a-priori estimate of covariance
  P = np.zeros( (n_states, n_states, n_time) ) #a-posterior estimate of co-variances
  K = np.zeros( (n_states, n_obs, n_times) ) #kalman gain

  Y_tilda = np.zeros((n_obs, n_times))
  S = np.zeros((n_obs, n_obs, n_times))

  # initialize X to last observed state
  X[:, 0] = Xo

  # loop through time
  progCount = 0 #counter to display progress estimates
  for k in range(n_times):

    if k/n_times*100 <= progCount*10:
      print(str(progCnt))
      progCount = progCount + 1

    # (1) a priori estimate of x ( x_hat(k|k-1) = A*X(k-1) )
    X_hat[:, k] = A@X[:, k-1]

    # (2) a priori estimate of x covariance ( P(k) = A*P(k-1)*A' + W )
    P_AP[:,:, k] = A@(P[:,:,k-1]@A.T + W)

    # (3) compute the difference between expected and observed measurement
    # i.e. measurement residual
    Y_tilda[:, k] = Y[:, k] - H@X_hat[:, k]

    # (4) Compute the covariance of the measurement residual
    S[:, :, k] = H@(P_AP[:,:,k]@H.T) + Q

    # (5) a posteriori estimate
    X[:,k] = X_hat[:, k] + K@Y_tilda[:, k]

    # Update Kalman Gain
    K[:,:,k] = (P_AP[:,:,k]@H.T)@np.linalg.pinv(S[:, :, k])

    # Compute the covariance of the a posteriori estimate
    P[:,:,k] = ( np.identity(n_states) - K[:, :, k]@H)@P_AP[:,:,k]
  return X
