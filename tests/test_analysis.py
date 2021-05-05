import aopy
import numpy as np
import warnings
import unittest

class factor_analysis_tests(unittest.TestCase):

    def test_kfold_factor_analysis(self):
        n = 100
        factor1 = 5
        factor2 = np.arange(1,n+1, 1)/10

        test_data = np.zeros((n, 16))
        test_data[:,0] = 2*factor1 + 3*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,1] = 4*factor1 + factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,2] = factor1 - factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,3] = -2*factor1 + 4*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,4] = 3*(factor1+factor2)+ np.random.normal(0, 1, size = (n))
        test_data[:,5] = factor1 - 0.5*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,6] = 2*factor1 + 7*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,7] = 4*factor1 - 3*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,8] = factor1 - 0.5*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,9] = 2.5*factor1 + 9*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,10] = factor1 - .136*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,11] = 7+2*factor1 + 7*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,12] = 1.5*factor1 - 3*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,13] = 2.67*factor1 - 0.5*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,14] = 7*factor1 + 9*factor2 + np.random.normal(0, 1, size = (n))
        test_data[:,15] = 2+ factor1 - .136*factor2 + np.random.normal(0, 1, size = (n))


        dimensions = np.arange(0,5 ,1).astype(int)

        # Test for no cross validation (nfold = 1)
        log_likelihood_score_1, iterations_required_1 = aopy.analysis.factor_analysis_dimensionality_score(test_data, dimensions, nfold=1, maxiter=1000, verbose=True)
        data_dimensionality_1 = np.argmax(np.mean(log_likelihood_score_1, 1))
        self.assertEqual(data_dimensionality_1, 4) # With no cross validation, the model will always fit best to the highest dimension tested

        # Test for nfold > 1
        log_likelihood_score, iterations_required = aopy.analysis.factor_analysis_dimensionality_score(test_data, dimensions, nfold=3, maxiter=1000, verbose=True)
        data_dimensionality = np.argmax(np.mean(log_likelihood_score, 1))
        self.assertEqual(data_dimensionality, 2)

class find_extrema_tests(unittest.TestCase):
    def test_find_trough_peak_idx(self):
        #Test single waveform
        deg_step = np.pi/8
        theta = np.arange(-np.pi, (2*np.pi)+(deg_step), deg_step)
        y = -np.sin(theta)
        
        trough_idx, peak_idx = aopy.analysis.find_trough_peak_idx(y)

        self.assertEqual(trough_idx, 12) 
        self.assertEqual(peak_idx, 20)

        # Test multiple waveforms
        y_multiple = np.zeros((len(theta), 4))
        y_multiple[:,0] = -np.sin(theta)
        y_multiple[:,1] = -np.sin(theta-deg_step)
        y_multiple[:,2] = -np.sin(theta-2*deg_step)
        y_multiple[:,3] = -np.sin(theta-3*deg_step)

        trough_idx_multiple, peak_idx_multiple = aopy.analysis.find_trough_peak_idx(y_multiple)
        np.testing.assert_allclose(trough_idx_multiple, np.array([12, 13, 14, 15]))
        np.testing.assert_allclose(peak_idx_multiple, np.array([20, 21, 22, 23]))

    def test_interpolate_extremum_poly2(self):
        # Define waveform
        deg_step = np.pi/8
        theta = np.arange(-np.pi, (2*np.pi)+(deg_step), deg_step)
        y = -np.sin(theta-deg_step/2)
        
        # Test on value in the middle of the waveform
        extremum_time, extremum_value, f = aopy.analysis.interpolate_extremum_poly2(12, y)
        self.assertAlmostEqual(extremum_time, 12.5) 
        self.assertAlmostEqual(extremum_value, -1, places=2)

        # Test on values at the beginning and end of the waveform
        theta_edge = np.arange(-np.pi, (2*np.pi)+(2*deg_step), deg_step)
        y_edge = -np.sin(theta_edge)
        extremum_time_edge1, extremum_value_edge1, f = aopy.analysis.interpolate_extremum_poly2(0, y_edge)
        self.assertAlmostEqual(extremum_time_edge1, 0) 
        self.assertAlmostEqual(extremum_value_edge1, 0)

        extremum_time_edge2, extremum_value_edge2, f = aopy.analysis.interpolate_extremum_poly2(len(theta_edge)-1, y_edge, extrap_peaks=True)
        self.assertGreater(extremum_time_edge2, len(theta_edge)-1) 
        self.assertLess(extremum_value_edge2, 0)

        


if __name__ == "__main__":
    unittest.main()
