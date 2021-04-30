"""
Code for basic neural data analysis
"""

import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn import model_selection
import warnings

def factor_analysis_dimensionality_score(data_in, dimensions, nfold, maxiter=1000, verbose=False):
    '''
    Estimate the latent dimensionality of an input dataset by appling cross validated 
    factor analysis (FA) to input data and returning the maximum likelihood values. 
    
    Args:
        data_in (nt, nch): Time series data in
        dimensions (ndim): 1D Array of dimensions to compute FA for 
        nfold (int): Number of cross validation folds to compute. Must be >= 1
        maxiter (int): Maximum number of FA iterations to compute if there is no convergence. Defaults to 1000.
        verbose (bool): Display % of dimensions completed. Defaults to False

    Returns:
        tuple: tuple containing:

        (ndim, nfold): Array of MLE FA score for each dimension for each fold
        
        (ndim, nfold): How many iterations of FA were required to converge for each fold
    '''

    # Initialize arrays
    log_likelihood_score = np.zeros((np.max(np.shape(dimensions)), nfold))
    iterations_required = np.zeros((np.max(np.shape(dimensions)), nfold))

    if verbose == True:
        print('Cross validating and fitting ...')

    # Compute the maximum likelihood score for each dimension using factor analysis    
    for dim_idx in range(len(dimensions)):
        fold_idx = 0

        # Handle the case without cross validation.
        if nfold == 1:
            fa = FactorAnalysis(n_components=dimensions[dim_idx], max_iter = maxiter)
            fafit = fa.fit(data_in.T)
            log_likelihood_score[dim_idx,fold_idx] = fafit.score(data_in.T)
            iterations_required[dim_idx,fold_idx] = fafit.n_iter_
            warnings.warn("Without cross validation the highest dimensional model will always fit best.")
        
        # Every other case with cross validation
        else:
            for trainidx, testidx in model_selection.KFold(n_splits=nfold).split(data_in.T):
                fa = FactorAnalysis(n_components=dimensions[dim_idx], max_iter = maxiter)
                fafit = fa.fit(data_in[:,trainidx].T)
                log_likelihood_score[dim_idx,fold_idx] = fafit.score(data_in[:,testidx].T)
                iterations_required[dim_idx,fold_idx] = fafit.n_iter_
                fold_idx += 1
        
        if verbose == True:
            print(str((100*(dim_idx+1))//len(dimensions)) + "% Complete")
            
    return log_likelihood_score, iterations_required
