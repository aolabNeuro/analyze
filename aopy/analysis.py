# analysis.py
# code for basic neural data analysis

import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn import model_selection

def kfold_factor_analysis(data_in, ncomp_fa, nfold, maxiter = 1000, verbose = False):
    '''
    Apply cross validated factor analysis (FA) to input data. 
    
    Inputs:
        data_in [ntime, nchannel]: Time series data in
        ncomp_fa [n dimensions]: 1D Array of dimensions to compute FA for 
        nfold [int]: Number of cross validation folds to compute
        maxiter [int]: Maximum number of FA iterations to compute if there is no convergence. Defaults to 1000.
        verbose [bool]: Display % of dimensions completed. Defaults to False

    Outputs:
        log_likelihood_score [n dimensions, nfold]: Array of MLE FA score for each dimension for each fold
        iterations_required [n dimensions, nfold]: How many iterations of FA were required to converge for each fold
    '''

    # Initialize arrays
    log_likelihood_score = np.zeros((np.max(np.shape(ncomp_fa)), nfold))
    iterations_required = np.zeros((np.max(np.shape(ncomp_fa)), nfold))

    # Split data into training and testing sets and perform FA analysis for each dimension
    if verbose == True:
        print('Cross validating and fitting ...')
    
    for jj in range(len(ncomp_fa)):
        ii = 0
        for trainidx, testidx in model_selection.KFold(n_splits = nfold).split(data_in.T):
            fa = FactorAnalysis(n_components = ncomp_fa[jj], max_iter = maxiter)
            fafit = fa.fit(data_in[:,trainidx].T)
            log_likelihood_score[jj,ii] = fafit.score(data_in[:,testidx].T)
            iterations_required[jj,ii] = fafit.n_iter_
            ii += 1
        
        if verbose == True:
            print(str((100*(jj+1))//len(ncomp_fa)) + "% Complete")
            
    return log_likelihood_score, iterations_required