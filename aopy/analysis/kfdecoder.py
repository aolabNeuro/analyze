# kfdecoder.py
#
# Kalman Filter implementation

import numpy as np
from numpy.linalg import inv as inv # used in Kalman Filter

class KFDecoder(object):
    """
    Kalman Filter Decoder
    
    Args:
        C (float, optional): default 1
            This parameter scales the noise matrix associated with the transition in kinematic states.
            It effectively allows changing the weight of the new neural evidence in the current update.
            Our implementation of the Kalman filter for neural decoding is based on that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
            with the exception of the addition of the parameter C.
            The original implementation has previously been coded in Matlab by Dan Morris (http://dmorris.net/projects/neural_decoding.html#code)
    
    Attributes:
        model ([A, W, H, Q] list): list of matrices:
            | [State Transition model,
            | Covariance of State Transition model,
            | Observation model,
            | Covariance of Observation model]
    """

    def __init__(self, C=1):
        self.C = C

    def fit(self, X_kf_train, y_train):
        """
        Train Kalman Filter Decoder
        
        Args:
            X_kf_train (ntime , nfeatures): This is the neural data in Kalman filter format. See example file for an example of how to format the neural data correctly.
            y_train (ntime , noutputs): These are the outputs that are being predicted

        Calculations for :math:`A`, :math:`W`, :math:`H`, :math:`Q` are as follows:

        .. math:: 
            
            A = X2*X1' (X1*X1')^{-1}

        .. math:: 
            
            W = \\frac{(X_2 - A*X_1)(X_2 - A*X_1)'}{(timepoints - 1)}
        
        .. math:: 
        
            H = Y*X'(X*X')^{-1}
        
        .. math:: 
            
            Q = \\frac{(Y-HX)(Y-HX)' }{time points}
        """

        # Renaming and reformatting variables to be in a more standard kalman filter nomenclature (from Wu et al, 2003):
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_train.T)
        Z = np.matrix(X_kf_train.T)

        # number of time bins
        nt = X.shape[1]

        # Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        # In our case, this is the transition from one kinematic state to the next
        X2 = X[:, 1:]
        X1 = X[:, 0:nt - 1]

        A = X2 * X1.T * inv(X1 * X1.T)  # Transition matrix
        W = (X2 - A * X1) * (X2 - A * X1).T / (
                    nt - 1) / self.C  # Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        # Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        # In our case, this is the transformation from kinematics to spikes
        H = Z * X.T * (inv(X * X.T))  # Measurement matrix
        Q = ((Z - H * X) * ((Z - H * X).T)) / nt  # Covariance of measurement matrix

        params = [A, W, H, Q]
        print('Shape of State Transition model (A) :' + str(A.shape))
        print('Shape of Covariance of State Transition model :' + str(W.shape))
        print('Shape of Observation model (H) :' + str(H.shape))
        print('Shape of Covariance of Observation model :' + str(Q.shape))
        self.model = params

    def fit_awf(self, X_kf_train, y_train, A, W):
        """
        Train Kalman Filter Decoder with A and W fixed. A is the state transition model and W is the associated covariance

        
        Args:
            X_kf_train (ntime , nfeatures): This is the neural data in Kalman filter format. See example file for an example of how to format the neural data correctly
            y_train (ntime , noutputs): These are the outputs that are being predicted

        Calculations as follows:

        .. math::
            
            H = Y*X'(X*X')^{-1}

        .. math:: 
        
            Q = \\frac{(Y-HX)(Y-HX)' }{time points}
        """

        # Renaming and reformatting variables to be in a more standard kalman filter nomenclature (from Wu et al, 2003):
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_train.T)
        Z = np.matrix(X_kf_train.T)

        # number of time bins
        nt = X.shape[1]

        # Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        # In our case, this is the transition from one kinematic state to the next
        X2 = X[:, 1:]
        X1 = X[:, 0:nt - 1]

        # A=X2*X1.T*inv(X1*X1.T) #Transition matrix
        # W=(X2-A*X1)*(X2-A*X1).T/(nt-1)/self.C #Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        # Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        # In our case, this is the transformation from kinematics to spikes
        H = Z * X.T * (inv(X * X.T))  # Measurement matrix
        Q = ((Z - H * X) * ((Z - H * X).T)) / nt  # Covariance of measurement matrix

        print('Shape of State Transition model (A) :' + str(A.shape))
        print('Shape of Covariance of State Transition model :' + str(W.shape))
        print('Shape of Observation model (H) :' + str(H.shape))
        print('Shape of Covariance of Observation model :' + str(Q.shape))
        params = [A, W, H, Q]
        self.model = params

    def predict(self, X_kf_test, y_test):
        """
        Predict outcomes using trained Kalman Filter Decoder

        Args:
        X_kf_test (ntime, nfeatures): This is the neural data in Kalman filter format.
        y_test (ntime , noutputs): The actual outputs. This parameter is necesary for the Kalman filter (unlike other decoders) because the first value is nececessary for initialization
        
        Returns:
            y_test_predicted (ntime, noutputs): The predicted outputs
        """

        # Extract parameters
        A, W, H, Q = self.model

        # First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (I am following Wu et al):
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_test.T)
        Z = np.matrix(X_kf_test.T)

        # Initializations
        num_states = X.shape[0]  # Dimensionality of the state
        states = np.empty(
            X.shape)  # Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m = np.matrix(np.zeros([num_states, num_states]))  # This is a priori estimate of X covariance
        P = np.matrix(np.zeros([num_states, num_states]))  # This is a posteriori estimate of X covariance
        state = X[:, 0]  # Initial state
        states[:, 0] = np.copy(np.squeeze(state))

        # Get predicted state for every time bin
        for t in range(X.shape[1] - 1):
            # Do first part of state update - based on transition matrix
            P_m = A * P * A.T + W  # a priori estimate of x covariance ( P(k) = A*P(k-1)*A' + W )
            state_m = A * state  # a priori estimate of x ( X(k|k-1) = A*X(k-1) )

            # Do second part of state update - based on measurement matrix
            K = P_m * H.T * inv(H * P_m * H.T + Q)  # Calculate Kalman gain ( K = P_ap*H'* inv(H*P_ap*H' + Q) )
            P = (np.matrix(np.eye(num_states)) - K * H) * P_m  # (a posteriori estimate, P (I - K*H)*P_ap )
            state = state_m + K * (Z[:,t + 1] - H * state_m)  # compute a posteriori estimate of x (X(k) = X(k|k-1) + K*(Z - H*X(k|k-1))
            states[:, t + 1] = np.squeeze(state)  # Record state at the timestep
        y_test_predicted = states.T
        return y_test_predicted
