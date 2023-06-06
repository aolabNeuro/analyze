from aopy.visualization import savefig
from aopy.analysis import accllr
import aopy
import os
import numpy as np
import warnings
import unittest

import os
import matplotlib.pyplot as plt
from scipy import signal

test_dir = os.path.dirname(__file__)
data_dir = os.path.join(test_dir, 'data')
write_dir = os.path.join(test_dir, 'tmp')
docs_dir = os.path.join(test_dir, '../docs/source/_images')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

class FactorAnalysisTests(unittest.TestCase):

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

class classify_cells_tests(unittest.TestCase):
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

    def test_classify_cells_spike_width(self):
        # make artificial narrow and long spikes
        npts = 32
        nspikes = 100
        narrow_sp_width = 6 #pts
        long_sp_width = 18 #pts
        nunits = 10
        waveform_data=[]

        x = np.arange(npts)
        long_spikes = np.zeros(npts)
        narrow_spikes = np.zeros(npts)
        long_spikes[5:long_sp_width+5] = -np.sin(x[:long_sp_width]/3)
        narrow_spikes[5:narrow_sp_width+5] = -np.sin(x[:narrow_sp_width])

        for iunit in range(nunits):
            if iunit%2 == 0:
                randdata = np.random.rand(npts,nspikes)/10
                temp_wf_data = np.tile(narrow_spikes, (nspikes,1)).T + randdata
            else:
                randdata = np.random.rand(npts,nspikes)/10
                temp_wf_data = np.tile(long_spikes, (nspikes,1)).T + randdata
            waveform_data.append(temp_wf_data)
        
        TTP, unit_lbls, avg_wfs, _ = aopy.analysis.classify_cells_spike_width(waveform_data, 100)
        exp_unit_lbls = np.array([0,1,0,1,0,1,0,1,0,1])
        np.testing.assert_allclose(unit_lbls, exp_unit_lbls)

class FanoFactorTests(unittest.TestCase):
    def test_get_unit_spiking_mean_variance(self): 
        spiking_data = np.zeros((2,2,2)) #(ntime, nunits, ntr)
        spiking_data[0,:,:] = 1
        unit_mean, unit_var = aopy.analysis.get_unit_spiking_mean_variance(spiking_data)
        np.testing.assert_allclose(unit_mean, np.array([2, 0]))
        np.testing.assert_allclose(unit_var, np.array([0, 0]))

class PCATests(unittest.TestCase):
    # test variance accounted for
    def test_get_pca_dimensions(self):
        # test single dimension returns correctly
        single_dim_data = np.array([[2, 2, 2], [1, 1, 1], [1, 1, 1]])
        single_dim_VAF = [1.,0.,0.]
        single_num_dims = 1

        VAF, num_dims, proj_data = aopy.analysis.get_pca_dimensions(single_dim_data)

        np.testing.assert_allclose(VAF, single_dim_VAF, atol=1e-7)
        self.assertAlmostEqual(num_dims, single_num_dims)
        self.assertEqual(proj_data, None)

        # Test max_dims optional parameter
        np.random.seed(0)
        data = np.random.randn(3,3)
        _, num_dims, proj = aopy.analysis.get_pca_dimensions(data, max_dims=1, project_data=True)
        self.assertEqual(num_dims, 1)
        self.assertEqual(proj.shape[1], 1)

        # Test VAF parameter
        np.random.seed(0)
        data = np.random.randn(3,3)
        _, num_dims, proj = aopy.analysis.get_pca_dimensions(data, VAF=0.5, project_data=True)
        self.assertEqual(num_dims, 1)
        self.assertEqual(proj.shape[1], 1)

        # Test projection optional parameter
        single_dim_data = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        expected_single_dim_data = np.array([[0.75], [-0.25], [-0.25], [-0.25]])
        VAF, num_dims, proj_data = aopy.analysis.get_pca_dimensions(single_dim_data, project_data=True)
        np.testing.assert_allclose(expected_single_dim_data, proj_data)

class misc_tests(unittest.TestCase):
    def test_find_outliers(self):
        # Test correct identification of outliers
        data = np.array([[-0.5,-0.5],[0.01,0.01],[0.1,0.1],[-0.75,-0.75], [1,1], [10,10]])
        outliers_labels, _ = aopy.analysis.find_outliers(data, 2)
        expected_outliers_labels = np.array([True, True, True, True, True, False])
        np.testing.assert_allclose(outliers_labels, expected_outliers_labels)

        # Test correct distance calculation
        data = np.array([[1,0], [0,0], [0,0], [-1,0]])
        _, outliers_dist = aopy.analysis.find_outliers(data, 2)
        expected_outliers_dist = np.array([1, 0, 0, 1])
        np.testing.assert_allclose(outliers_dist, expected_outliers_dist)

    def test_calc_task_rel_dims(self):
        test_data = np.array([0,1,2,3,4])

        # Test 2D kinematics condition
        velocity2D = np.tile(test_data,(2,1)).T
        data2D = np.tile(test_data,(3,1)).T
        task_subspace, projected_data = aopy.analysis.calc_task_rel_dims(data2D, velocity2D)
        M = np.ones((5,3))
        M[:,1:] = velocity2D
        np.testing.assert_allclose(data2D-np.mean(data2D,axis=0), np.round(M @ task_subspace.T, 10))

        # Test list concatenation
        task_subspace, _ = aopy.analysis.calc_task_rel_dims([data2D, data2D], [velocity2D, velocity2D])

        np.testing.assert_allclose(np.vstack([data2D, data2D])-np.mean(np.vstack([data2D, data2D]),axis=0),
                                     np.round(np.vstack([M,M]) @ task_subspace.T, 10))

        # Test output concatenation
        task_subspace, projected_data = aopy.analysis.calc_task_rel_dims([data2D, data2D], [velocity2D, velocity2D], True)
        np.testing.assert_allclose(np.vstack([data2D, data2D])-np.mean(np.vstack([data2D, data2D]),axis=0),
                                     np.round(np.vstack([M,M]) @ task_subspace.T, 10))
        np.testing.assert_allclose(projected_data.shape, np.vstack([data2D, data2D]).shape)

class tuningcurve_fitting_tests(unittest.TestCase):
    def test_run_tuningcurve_fit(self):
        nunits = 7
        targets = np.arange(0, 360, 45)
        mds_true = np.linspace(1, 3, nunits)/2
        pds_offset = np.arange(-45,270,45)
        data = np.zeros((nunits,8))*np.nan
        for ii in range(nunits):
            data[ii,:] = mds_true[ii]*np.sin(np.deg2rad(targets)-np.deg2rad(pds_offset[ii])) + 2

        # If the mds and pds output are correct the fitting params are correct because they are required for the calculation.
        _, md, pd = aopy.analysis.run_tuningcurve_fit(data, targets)
        np.testing.assert_allclose(mds_true, md)
        np.testing.assert_allclose(pds_offset, np.rad2deg(pd)-90)

        # Check that forcing nans runs the function
        data[0,0] = np.nan
        _, md, pd = aopy.analysis.run_tuningcurve_fit(data, targets, fit_with_nans=True)
        np.testing.assert_allclose(mds_true, md)
        np.testing.assert_allclose(pds_offset, np.rad2deg(pd)-90)

        # Check that nans propogate correctly
        mds_true[0] = np.nan
        pds_true = pds_offset.astype(float)
        pds_true[0] = np.nan
        _, md, pd = aopy.analysis.run_tuningcurve_fit(data, targets)
        np.testing.assert_allclose(mds_true, md)
        np.testing.assert_allclose(pds_true, np.rad2deg(pd)-90)


        # Test that code runs with too many nans
        data[0,:] = np.nan
        _, md, pd = aopy.analysis.run_tuningcurve_fit(data, targets, fit_with_nans=True)
        np.testing.assert_allclose(mds_true, md)
        np.testing.assert_allclose(pds_true, np.rad2deg(pd)-90)

class CalcTests(unittest.TestCase):

    def test_calc_rms(self):
        # sanity check
        signal = np.array([1])
        rms = aopy.analysis.calc_rms(signal)
        self.assertEqual(rms, 0)

        # check dimensions are preserved
        signal = np.array([[0.5, -0.5], [1., -1.]]).T
        rms = aopy.analysis.calc_rms(signal)
        np.testing.assert_allclose(rms, np.array([0.5, 1.]))

        # check without remove offset
        signal = np.array([1])
        rms = aopy.analysis.calc_rms(signal, remove_offset=False)
        self.assertAlmostEqual(rms, 1.)

    def test_calc_freq_domain_amplitude(self):
        data = np.sin(np.pi*np.arange(1000)/10) + np.sin(2*np.pi*np.arange(1000)/10)
        samplerate = 1000
        freqs, ampls = aopy.analysis.calc_freq_domain_amplitude(data, samplerate)
        self.assertEqual(freqs.size, 500)
        self.assertEqual(ampls.size, 500)

        # Expect 100 and 50 Hz peaks at 1 V each
        self.assertAlmostEqual(ampls[freqs==100., 0][0], 1)
        self.assertAlmostEqual(ampls[freqs==50., 0][0], 1)
        self.assertAlmostEqual(ampls[freqs==25., 0][0], 0)

        # Expect 1/sqrt(2) V RMS
        freqs, ampls = aopy.analysis.calc_freq_domain_amplitude(data, samplerate, rms=True)
        self.assertAlmostEqual(ampls[freqs==100., 0][0], 1/np.sqrt(2))

        # Expect 2 channels with different signals
        data = np.vstack((np.sin(np.pi*np.arange(1000)/10), np.sin(2*np.pi*np.arange(1000)/10))).T
        freqs, ampls = aopy.analysis.calc_freq_domain_amplitude(data, samplerate)
        self.assertEqual(freqs.size, 500)
        self.assertEqual(ampls.shape[0], 500)
        self.assertEqual(ampls.shape[1], 2)
        self.assertAlmostEqual(ampls[freqs==50., 0][0], 1.)
        self.assertAlmostEqual(ampls[freqs==100., 1][0], 1.)

    def test_calc_ISI(self):
        data = np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]])
        data_T = data.T
        fs = 100
        bin_width = 0.01
        hist_width = 0.1
        ISI_hist, hist_bins = aopy.analysis.calc_ISI(data_T, fs, bin_width, hist_width)
        self.assertEqual(ISI_hist[1, 0], 2)
        self.assertEqual(ISI_hist[2, 0], 2)
        self.assertEqual(ISI_hist[1, 1], 3)

    def test_calc_sem(self):
        data = np.arange(10, dtype=float)
        SEM = aopy.analysis.calc_sem(data)
        self.assertAlmostEqual(np.std(data)/np.sqrt(10), SEM)

        # test with NaN
        data[5] = np.nan
        SEM = aopy.analysis.calc_sem(data)
        self.assertAlmostEqual(np.nanstd(data)/np.sqrt(9), SEM)

        # Test with multiple dimensions
        data = np.tile(data, (2,2, 1))
        SEM = aopy.analysis.calc_sem(data, axis=(0,2))
        np.testing.assert_allclose(SEM, np.nanstd(data, axis=(0,2))/np.sqrt(18) )

    def test_calc_rolling_average(self):
        data = np.array([1, 2, 3, 4, 5])
        kernel_size = 3
        data_convolved = aopy.analysis.calc_rolling_average(data, kernel_size)
        np.testing.assert_array_almost_equal(data_convolved, np.array([2., 2., 3., 4., 4.]))

    def test_calc_corr_over_elec_distance(self):
        acq_data = np.array([[1, 2, 3], [4, 5, 6]])
        acq_ch = np.array([1, 2])
        elec_pos = np.array(
            [[1, 1],
            [2,2],]
        )
        dist, corr = aopy.analysis.calc_corr_over_elec_distance(acq_data, acq_ch, elec_pos, method='pearson', bins=1, exclude_zero_dist=True)

        self.assertEqual(corr.size, 1)
        self.assertEqual(dist.size, 1)
        self.assertEqual(dist[0], np.sqrt(2))
        self.assertEqual(corr[0], 1.0)

    def test_calc_erp(self):
        nevents = 3
        event_times = 0.2 + np.arange(nevents)
        samplerate = 1000
        nch = 2
        data = np.zeros(((1+nevents)*samplerate, nch))

        # Make the data zero everywhere except for one sample after each event time
        print([int(t)+1 for t in event_times*samplerate])
        data[[int(t)+1 for t in event_times*samplerate],0] = 1
        data[[int(t)+1 for t in event_times*samplerate],1] = 2

        self.assertEqual(np.sum(data[:,0]), nevents)
        self.assertEqual(np.sum(data[:,1]), nevents*2)

        erp = aopy.analysis.calc_erp(data, event_times, 0.1, 0.1, samplerate, subtract_baseline=False)
        self.assertEqual(erp.shape[0], 3)

        mean_erp = np.mean(erp, axis=0)
        self.assertEqual(np.sum(mean_erp[:,0]), 1)
        self.assertEqual(np.sum(mean_erp[:,1]), 2)

        # Subtract baseline
        data += 1
        erp = aopy.analysis.calc_erp(data, event_times, 0.1, 0.1, samplerate)
        mean_erp = np.mean(erp, axis=0)
        self.assertEqual(np.sum(mean_erp[:,0]), 1)
        self.assertEqual(np.sum(mean_erp[:,1]), 2)

        # Specify baseline window
        data[0] = 100
        erp = aopy.analysis.calc_erp(data, event_times, 0.1, 0.1, samplerate, baseline_window=())
        mean_erp = np.mean(erp, axis=0)
        self.assertEqual(np.sum(mean_erp[:,0]), 1)
        self.assertEqual(np.sum(mean_erp[:,1]), 2)

    def test_calc_max_erp(self):
        nevents = 3
        event_times = 0.2 + np.arange(nevents)
        samplerate = 1000
        nch = 2
        data = np.zeros(((1+nevents)*samplerate, nch))

        # Make the data zero everywhere except for one sample after each event time
        data[[int(t)+1 for t in event_times*samplerate],0] = 1
        data[[int(t)+1 for t in event_times*samplerate],1] = 2

        max_erp = aopy.analysis.calc_max_erp(data, event_times, 0.1, 0.1, samplerate)
        self.assertEqual(max_erp[0], 1) 
        self.assertEqual(max_erp[1], 2)

        # Specify search window
        search_window = (0.05, 0.06)
        max_erp = aopy.analysis.calc_max_erp(data, event_times, 0.1, 0.1, samplerate, max_search_window=search_window)
        self.assertTrue(max_erp[0] == 0) 
        self.assertTrue(max_erp[1] == 0)

        # Test without trial averaging
        max_erp = aopy.analysis.calc_max_erp(data, event_times, 0.1, 0.1, samplerate, trial_average=False)
        self.assertEqual(max_erp.shape[0], len(event_times)) 
        self.assertEqual(max_erp.shape[1], nch) 

    def test_calc_corr2_map(self):
        # Test correlation map
        nrows = 11
        ncols = 11
        data1 = np.ones((nrows,ncols))
        data2 = np.ones(data1.shape)
        knlsz = 3
        nrows_changed = 5
        ncols_changed = 3
        for irow in range(nrows_changed):
            data2[irow,:ncols_changed] = np.arange(ncols_changed)+2

        NCC, _ = aopy.analysis.calc_corr2_map(data1, data2, knlsz, False)
        
        knl_offset = int((knlsz-1)/2)
        # # Check that everyone after the changed rows gives corr=1
        self.assertAlmostEqual(np.sum(NCC[int(nrows_changed+knl_offset):,0]), int((nrows-(nrows_changed+knl_offset))), 3)
        # Check that everyone after the changed col gives corr=1
        self.assertAlmostEqual(np.sum(NCC[0,int(ncols_changed+knl_offset):]), int((ncols-(ncols_changed+knl_offset))), 3)

        # Test shifts
        data2_shifted = np.roll(data2, 2, axis=0)

        NCC2, shifts = aopy.analysis.calc_corr2_map(data2,data2_shifted, 3, True)
        np.testing.assert_allclose(NCC2, np.ones((nrows, ncols)))
        self.assertEqual(shifts[0], -2)
        self.assertEqual(shifts[1], 0)

        # Plot for example figure 
        fig, [ax1, ax2, ax4] = plt.subplots(1,3, figsize=(12,3))
        im1 = ax1.pcolor(data1)
        ax1.set(title='Data 1')
        plt.colorbar(im1, ax=ax1)
        im2 = ax2.pcolor(data2)
        ax2.set(title='Data 2')
        plt.colorbar(im2, ax=ax2)

        im4 = ax4.pcolor(NCC)
        ax4.set(title='NCC')
        plt.colorbar(im4, ax=ax4)
        fig.tight_layout()
    
        filename = 'calc_corr2_map.png'
        aopy.visualization.savefig(docs_dir, filename)


class CurveFittingTests(unittest.TestCase):

    def test_fit_linear_regression(self):
        """
        Creates same columns of elements 1 through 9, and linearly transform with known slope and intercept
        """
    
        NUM_ROWS, NUM_COLUMNS = 10,3
        
        X = np.arange(NUM_ROWS).reshape(-1,1)
        X = np.tile(X, (1,NUM_COLUMNS) )
        
        #create dependant vars.
        slope = 2.0
        intercept = 3.0
        r = 1.0 
        Y = slope * X + intercept    
        
        result_slope, result_intercept, result_coeff = aopy.analysis.fit_linear_regression(X, Y)
        np.testing.assert_allclose(np.tile([slope], (NUM_COLUMNS,) ), result_slope)
        np.testing.assert_allclose(np.tile([intercept], (NUM_COLUMNS,) ), result_intercept)
        np.testing.assert_allclose(np.tile([r], (NUM_COLUMNS,) ), result_coeff)
        
class ModelFitTests(unittest.TestCase):

    def test_linear_fit_analysis(self):
        test_slope = 2
        test_intercept = 1
        xdata = np.arange(5)
        ydata = test_slope*xdata + test_intercept
        linear_fit, _, pcc, _, _ = aopy.analysis.linear_fit_analysis2D(xdata, ydata)
        np.testing.assert_allclose(linear_fit, ydata)
        self.assertAlmostEqual(pcc, 1.)

        # Test fit intercept
        _, _, _, _, fit = aopy.analysis.linear_fit_analysis2D(xdata, ydata, fit_intercept=False)
        self.assertAlmostEqual(fit.intercept_, 0)

        # Test Weights
        ydata[0] = 3
        weights = np.array([0, 1, 1, 1, 1])
        linear_fit, _, pcc, _, _ = aopy.analysis.linear_fit_analysis2D(xdata, ydata, weights=weights)
        np.testing.assert_allclose(linear_fit[1:], ydata[1:])
        
    def test_lda(self):
        # idea is to have 10 points in the lower left quadrant
        # and another points in the upper right quadrant.
        # lda should be able to classify this distribution into 100 percent
        X_train_lda = np.array([[-1, -1], [-2, -1], [-3, -2], [-5, -7], [-3, -5], [-6,  -4],
                                [1, 1],   [2, 1],   [3, 2],   [3,1], [3,9], [2,3]])
        y_class_train = np.array([1,1,1,1,1,1, -1, -1, -1, -1,-1,-1])
        
        accuracy, std = aopy.analysis.classify_by_lda(X_train_lda, y_class_train, 
                                 n_splits=5,
                                 n_repeats=3, 
                                 random_state=1)
        
        self.assertAlmostEqual(accuracy, 1.0)
        self.assertAlmostEqual(std, 0.0 )

class AccLLRTests(unittest.TestCase):

    def test_detect_accLLR(self, upper=10, lower=-10):
      
        accllr_data = np.array(
            [
                [0, 0, 1, 2, 3, 4, 4, 5, 6, 10, 10, 11],            # even trials
                [0, -1, -1, -2, -3, -4, -5, -5, -6, -10, -11, -11]  # odd trials
            ])
        accllr_data = np.tile(accllr_data.T, (1, 20)) # simulate 40 trials
        self.assertEqual(accllr_data.shape, (12, 40))

        p, st = accllr.detect_accllr(accllr_data, upper, lower)

        p_fast, st_fast = accllr.detect_accllr_fast(accllr_data, upper, lower)
        
        np.testing.assert_allclose(p, p_fast)
        np.testing.assert_allclose(st, st_fast)

        np.testing.assert_allclose(p, [.5, .5, 0])
        self.assertEqual(st.shape, (40,))

        import time
        t0 = time.perf_counter()
        for i in range(1000):
            accllr.detect_accllr(accllr_data, upper, lower)
        t1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        for i in range(1000):
            accllr.detect_accllr_fast(accllr_data, upper, lower)
        t2 = time.perf_counter() - t0

        print(f"detect_accllr() took {t1:.2f} s")
        print(f"detect_accllr_fast() took {t2:.2f} s")
        
    def test_delong_roc_variance(self):
        alpha = .95
        y_pred = np.array([0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
        y_true = np.array([0,    1,    0,    0,    1,    1,    0,    1,    0   ])

        auc, auc_cov = accllr.calc_delong_roc_variance(
            y_true,
            y_pred)

        # Compare to values in the original paper
        self.assertAlmostEqual(auc, 0.8)
        self.assertAlmostEqual(auc_cov, 0.02875)
    
    def test_calc_accllr_roc(self):
        test_data1 = [0, 1, 2, 4, 6, 7, 3, 4, 5, 16, 7]
        test_data2 = [0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0]
        auc, se = accllr.calc_accllr_roc(test_data1, test_data2)

        matlab_auc = 0.921487603
        matlab_se = 0.06396
        
        self.assertAlmostEqual(auc, matlab_auc)
        # self.assertAlmostEqual(se, matlab_se) # known difference between matlab and python version

    def test_llr(self):
        test_lfp = np.array([0, 0.8, 1, 5, 4, 1, 2.5, 3, 4, 4, 4])
        test_data1 = np.array([0, 1, 2, 4, 6, 2, 3, 4, 5, 16, 7])
        test_data2 = np.array([0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0])

        llr = accllr.calc_llr_gauss(test_lfp, test_data1, test_data2, np.std(test_data1), np.std(test_data2))

        matlab_llr = np.array(
            [-1.863, -1.09,  -0.682, 17.468, 17.38,  -1.892,  5.692,  8.998,  2.948, 13.3, 17.235]
        )
        np.testing.assert_allclose(llr.round(3), matlab_llr)

    def test_accllr(self):
        
        lfp_altcond = np.array([
            [0, 1.1, 2, 4, 6, 2, 3, 4, 5, 4, 7],
            [0.5, 0.8, 2, 5, 4, 3, 2.5, 3, 4, 4, 4],
            [0.2, 0.3, 2., 4, 8, 3, 2.1, 7, 4, 4, 4]

        ]).T
        lfp_nullcond = np.array([
            [0, 0.2, 0, 1, 0, 1, 0, 0, 0.3, 0, 0],
            [0, 1, 0, 0.8, 0, 0, 0, 0.7, 0.8, 0, 0],
            [0, 0, 0, 0., 0, 2, 0, 0, 2, 0, 0],

        ]).T

        accllr_altcond, accllr_nullcond = accllr.calc_accllr_lfp(lfp_altcond, lfp_nullcond,
                                                        lfp_altcond, lfp_nullcond, 
                                                        common_variance=True)
        '''
        MATLAB accllr_altcond (rounded to 2 decimals): 
        [[ 8.0000e-02  2.9000e-01 -4.5000e-01]
        [ 4.9100e+00  5.1200e+00  4.3700e+00]
        [ 1.8550e+01  2.7270e+01  1.8020e+01]
        [ 6.2000e+01  3.5720e+01  8.4390e+01]
        [ 6.2000e+01  4.0250e+01  8.8920e+01]
        [ 7.2270e+01  4.7780e+01  9.3730e+01]
        [ 8.8180e+01  4.9480e+01  1.3420e+02]
        [ 1.0596e+02  5.9800e+01  1.4452e+02]
        [ 1.2527e+02  7.9110e+01  1.6383e+02]
        [ 1.7354e+02  9.5700e+01  1.8042e+02]]
        MATLAB accllr_nullcond: 
        [[  -0.3     0.83   -0.28]
        [  -5.13   -4.     -5.11]
        [ -18.1   -18.96  -26.79]
        [ -61.54  -62.4   -70.23]
        [ -64.9   -68.27  -68.05]
        [ -72.64  -76.01  -75.8 ]
        [ -98.77  -94.41 -101.93]
        [-116.95 -109.33 -105.96]
        [-136.25 -128.63 -125.27]
        [-166.42 -158.8  -155.44]]
        '''

        # Known difference between matlab and python version:
        #   python accumulates LLR from t=0, matlab from t=1
        self.assertAlmostEqual(accllr_altcond[1,0], 8.0e-2, places=2)
        self.assertAlmostEqual(accllr_altcond[-1,0], 1.7354e2, places=2)
        self.assertAlmostEqual(accllr_nullcond[1,0], -0.3, places=2)
        self.assertAlmostEqual(accllr_nullcond[-1,0], -166.42, places=2)

    def test_choose_accllr_level(self):

        lfp_altcond = np.array([
            [0, 1.1, 2, 4, 6, 2, 3, 4, 5, 4, 7],
            [0.5, 0.8, 2, 5, 4, 3, 2.5, 3, 4, 4, 4],
            [0.2, 0.3, 2., 4, 8, 3, 2.1, 7, 4, 4, 4]

        ]).T
        lfp_nullcond = np.array([
            [0, 0.2, 0, 1, 0, 1, 0, 0, 0.3, 0, 0],
            [0, 1, 0, 0.8, 0, 0, 0, 0.7, 0.8, 0, 0],
            [0, 0, 0, 0., 0, 2, 0, 0, 2, 0, 0],

        ]).T
        accllr_altcond, accllr_nullcond = accllr.calc_accllr_lfp(lfp_altcond, lfp_nullcond,
                                                        lfp_altcond, lfp_nullcond, 
                                                        common_variance=True)

        nlevels = 200
        p_altcond, p_nullcond, _, levels = accllr.calc_accllr_performance(accllr_altcond, accllr_nullcond, nlevels)
        level = accllr.choose_best_level(p_altcond, p_nullcond, levels)
        print(level)
        
        # From running the matlab version we expect
        matlab_level = 0.902120
        self.assertAlmostEqual(level, matlab_level, places=5)


    def test_accllr_single_ch(self):
        '''
        Check that the selection time wrapper computes accllr correctly compared to the accllr_functions
        '''
        samplerate = 1000
        lfp_altcond = np.array([
            [0, 1.1, 2, 4, 6, 2, 3, 4, 5, 4, 7],
            [0.5, 0.8, 2, 5, 4, 3, 2.5, 3, 4, 4, 4],
            [0.2, 0.3, 2., 4, 8, 3, 2.1, 7, 4, 4, 4]

        ]).T
        lfp_nullcond = np.array([
            [0, 0.2, 0, 1, 0, 1, 0, 0, 0.3, 0, 0],
            [0, 1, 0, 0.8, 0, 0, 0, 0.7, 0.8, 0, 0],
            [0, 0, 0, 0., 0, 2, 0, 0, 2, 0, 0],

        ]).T
        data_altcond = np.expand_dims(lfp_altcond,1)
        data_nullcond = np.expand_dims(lfp_nullcond,1)
        lowpass_altcond = data_nullcond
        lowpass_nullcond = data_altcond
        
        ch = 0
        wrapper_accllr_altcond, wrapper_accllr_nullcond, p_altcond, p_nullcond, selection_t, roc_auc, roc_se = \
            accllr.calc_accllr_st_single_ch(data_altcond[:,ch,:], data_nullcond[:,ch,:], lowpass_altcond[:,ch,:],
                                 lowpass_nullcond[:,ch,:], 'lfp', 1./samplerate, nlevels=200, verbose_out=True) # try parallel True and False
        
        single_accllr_altcond, single_accllr_nullcond = accllr.calc_accllr_lfp(data_altcond[:,ch,:], data_nullcond[:,ch,:],
                                                        lowpass_altcond[:,ch,:], lowpass_nullcond[:,ch,:], 
                                                        common_variance=True)

        np.testing.assert_allclose(single_accllr_altcond, wrapper_accllr_altcond)
        np.testing.assert_allclose(single_accllr_nullcond, wrapper_accllr_nullcond)
        

    def test_calc_accLLR_threshold(self):
        # LFP data 
        npts = 50
        ntrials = 10
        onset_idx = 20
        noise_sd = npts/2
        altcond = np.zeros(npts)
        altcond[onset_idx:] = np.arange(npts-onset_idx)
        altcond = np.repeat(altcond[:,None], ntrials, axis=1)
        np.random.seed(0)
        nullcond = np.random.normal(0,noise_sd,size=altcond.shape)
        altcond += np.random.normal(0,noise_sd,size=altcond.shape)
        
        accllr_altcond, accllr_nullcond = accllr.calc_accllr_lfp(altcond, nullcond,
                                                        altcond, nullcond)

        p_altcond, p_nullcond, _, _ = accllr.calc_accllr_performance(accllr_altcond, accllr_nullcond, 200)
        p_correct_detect = (p_altcond[:,0]+p_nullcond[:,1])/2
        p_incorrect_detect = (p_nullcond[:,0]+p_altcond[:,1])/2

        plt.figure()
        plt.plot(p_correct_detect, p_incorrect_detect, 'o')
        plt.xlabel('Hit prob')
        plt.ylabel('False Alarm prob')
        filename = 'accllr_roc.png'
        aopy.visualization.savefig(write_dir, filename)

    def test_match_selectivity_accLLR(self):
        # LFP data 
        npts = 50
        nch = 2
        ntrials = 10
        onset_idx = 20
        noise_sd = npts/2
        altcond = np.zeros(npts)
        altcond[onset_idx:] = np.arange(npts-onset_idx)
        altcond = np.repeat(np.tile(altcond, (nch,1)).T[:,:,None], ntrials, axis=2)
        np.random.seed(0)
        nullcond = np.random.normal(0,noise_sd,size=altcond.shape)
        altcond += np.random.normal(0,noise_sd,size=altcond.shape)

        # Make ch2 have higher selectivity
        altcond[onset_idx:,1,:] *= 2

        test_data_altcond = altcond.copy()
        test_data_nullcond = nullcond.copy()

        # First test without matching selectivity
        sTime_alt, roc_auc, roc_se, roc_p_fdrc = accllr.calc_accllr_st(altcond, nullcond, altcond, nullcond, 'lfp', 1)
        print("Matching selectivities:")
        print(roc_auc)
        self.assertTrue(roc_auc[1] > roc_auc[0])

        # Test wrapper with LFP data and no selectivity matching and trial_average=True
        sTime_alt, roc_auc, roc_se, roc_p_fdrc = accllr.calc_accllr_st(altcond, nullcond, altcond, nullcond, 'lfp', 1, 
                                                                       match_selectivity=True, noise_sd_step=noise_sd)
        print("Matched selectivities:")
        print(roc_auc)
        self.assertTrue(roc_auc[1] <= roc_auc[0]) # selectivity should flip

        # One channel should remain the same pre- and post- selectivity matching,
        # while the other channel should have added noise.
        before = np.mean(test_data_altcond, axis=2) # mean across trials
        after = np.mean(altcond, axis=2) # mean across trials
        plt.figure()
        plt.plot(before[:,0], 'r', label='before matching ch 1')
        plt.plot(before[:,1], 'b', label='before matching ch 2')
        plt.plot(after[:,0], 'g--', label='after matching ch 1')
        plt.plot(after[:,1], 'c--', label='after matching ch 2')
        plt.legend()
        filename = 'match_selectivity_accllr.png'
        aopy.visualization.savefig(docs_dir, filename)
        plt.close()

    def test_calc_accLLR_wrapper(self):
        npts = 100
        nch = 50
        ntrials = 30
        onset_idx = 50
        altcond = np.zeros(npts)
        altcond[onset_idx:] = np.arange(npts-onset_idx)
        altcond = np.repeat(np.tile(altcond, (nch,1)).T[:,:,None], ntrials, axis=2)
        np.random.seed(0)
        nullcond = np.random.normal(0,5,size=altcond.shape)

        # Test wrapper with LFP data and no selectivity matching and trial_average=True
        st, roc_auc, roc_se, roc_p_fdrc = accllr.calc_accllr_st(altcond, nullcond, altcond, nullcond, 'lfp', 1)
    
        self.assertEqual(st.shape, (nch, ntrials))
        mask = np.logical_and(st > 50, st < 70)
        mask = np.all(mask, axis=1) # all trials should have selectivity within 50-70 s
        self.assertEqual(np.sum(mask), nch) # all channels should have selectivity within 50-70 s

        # Test on some real data
        test_data = aopy.data.load_hdf_group(data_dir, 'accllr_test_data.hdf')
        altcond = test_data['data_altcond']
        nullcond = test_data['data_nullcond']
        altcond_lp = altcond.copy()
        nullcond_lp = nullcond.copy()
        samplerate = 1000
        time_before = 0.05

        st, roc_auc, roc_se, roc_p_fdrc = accllr.calc_accllr_st(altcond, nullcond, altcond_lp, nullcond_lp, 'lfp', 1./samplerate)
        accllr_mean = np.nanmean(st, axis=1)

        plt.figure()
        ch_data = np.concatenate((np.mean(nullcond, axis=2), np.mean(altcond, axis=2)), axis=0)
        t = np.arange(ch_data.shape[0])/samplerate - time_before
        plt.plot(t, ch_data)
        x = np.tile(accllr_mean, (2,1))
        min_max = np.array([np.min(ch_data, axis=0), np.max(ch_data, axis=0)])
        plt.gca().set_prop_cycle(None) # reset the color cycler
        plt.plot(x, min_max, '--')
        filename = 'accllr_test_data.png'
        aopy.visualization.savefig(docs_dir, filename)
        plt.close()

        # Test again with selectivity matching
        np.random.seed(0)
        st, roc_auc, roc_se, roc_p_fdrc = accllr.calc_accllr_st(altcond, nullcond, altcond_lp, nullcond_lp, 'lfp', 1./samplerate, 
                                                                match_selectivity=True, noise_sd_step=8)
        accllr_mean = np.nanmean(st, axis=1)

        plt.figure()
        ch_data = np.concatenate((np.mean(nullcond, axis=2), np.mean(altcond, axis=2)), axis=0)
        t = np.arange(ch_data.shape[0])/samplerate - time_before
        plt.plot(t, ch_data)
        x = np.tile(accllr_mean, (2,1))
        min_max = np.array([np.min(ch_data, axis=0), np.max(ch_data, axis=0)])
        plt.gca().set_prop_cycle(None) # reset the color cycler
        plt.plot(x, min_max, '--')
        filename = 'accllr_test_data_match_selectivity.png'
        aopy.visualization.savefig(docs_dir, filename)
        plt.close()


class SpectrumTests(unittest.TestCase):

    def setUp(self):
        self.T = 0.05
        self.fs = 25000
        self.freq = [600, 312, 2000]
        self.a = 0.02
        # testing generate test_signal
        self.f0 = self.freq[0]
        self.x, self.t = aopy.utils.generate_test_signal(self.T, self.fs, self.freq, [self.a * 2, self.a*0.5, self.a*1.5, self.a*20 ])

        self.n_ch = 8
        self.x2 = aopy.utils.generate_multichannel_test_signal(self.T, self.fs, self.n_ch, self.freq[0], self.a*0.5) + \
            aopy.utils.generate_multichannel_test_signal(self.T, self.fs, self.n_ch, self.freq[1], self.a*1.5)
        self.t2 = np.arange(self.T*self.fs)/self.fs

        self.win_t = 0.01
        self.step_t = 0.005
        self.bw = 4.
    
    def test_multitaper(self):
        f, psd_filter, mu = aopy.analysis.get_psd_multitaper(self.x, self.fs)
        psd = aopy.analysis.get_psd_welch(self.x, self.fs, np.shape(f)[0])[1]

        fname = 'multitaper_powerspectrum.png'
        plt.figure()
        plt.plot(f, psd, label='Welch')
        plt.plot(f, psd_filter, label='Multitaper')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.legend()
        savefig(write_dir, fname) # both figures should have peaks at [600, 312, 2000] Hz

        bands = [(0, 10), (250, 350), (560, 660), (2000, 2010), (4000, 4100)]
        lfp = aopy.analysis.multitaper_lfp_bandpower(f, psd_filter, bands, False)
        plt.figure()
        plt.plot(np.arange(len(bands)), np.squeeze(lfp), '-bo')
        plt.xticks(np.arange(len(bands)), bands)
        plt.xlabel('Frequency band (Hz)')
        plt.ylabel('Band Power')
        fname = 'lfp_bandpower.png'
        savefig(write_dir, fname) # Should have power in [600, 312, 2000] Hz but not 10 or 4000

        f, psd_filter, mu = aopy.analysis.get_psd_multitaper(self.x2, self.fs)
        self.assertEqual(psd_filter.shape[1], self.n_ch)
        print(mu.shape)
        lfp = aopy.analysis.multitaper_lfp_bandpower(f, psd_filter, bands, False)
        self.assertEqual(lfp.shape[1], self.x2.shape[1])
        self.assertEqual(lfp.shape[0], len(bands))

        #TODO: complete sgram test
        f_sg, t_sg, sgram = aopy.analysis.get_sgram_multitaper(
            self.x2, self.fs, self.win_t, self.step_t, self.bw
        )
        self.assertEqual(sgram.shape[0], self.win_t*self.fs // 2 + 1) # correct freq. bin count
        self.assertEqual(sgram.shape[-1], self.x2.shape[-1]) # correct channel output count

    def test_tfspec(self):
        num_points = 5000
        T = 5
        t = np.linspace(0,T,num_points)
        test_data = np.zeros((t.shape[0],2))
        test_data[:,0] = 1.5*np.sin(2*np.pi*10*t)
        test_data[:,1] = 2*np.cos(2*np.pi*30*t)
        test_data[t>=T/2,0] = 2*np.sin(2*np.pi*5*t[t<=T/2])
        test_data[t>=T/2,1] = 1*np.cos(2*np.pi*15*t[t<=T/2])
        
        NW = 2
        BW = 1
        fs = num_points/T
        dn = 0.1
        fk = 50
        n, p, k = aopy.precondition.convert_taper_parameters(NW, BW)
        f_spec,t_spec,spec = aopy.analysis.calc_mt_tfr(test_data, n, p, k, fs, dn, fk, pad=2, ref=False)
        
        fig,ax=plt.subplots(1,4,figsize=(8,2),tight_layout=True)
        ax[0].plot(t,test_data[:,0],linewidth=0.2)
        ax[0].plot(t,test_data[:,1],linewidth=0.2)
        ax[0].set(xlabel='Time (s)',ylabel='Amplitude',title='Signals')
        ax[1].imshow((spec[:,:,0]),aspect='auto',origin='lower',extent=[0,T,0,f_spec[-1]])
        ax[1].set(ylabel='Frequency (Hz)',xlabel='Time [s]',title='Spectrogram (ch1)')
        ax[2].imshow((spec[:,:,1]),aspect='auto',origin='lower',extent=[0,T,0,f_spec[-1]])
        ax[2].set(ylabel='Frequency (Hz)',xlabel='Time [s]',title='Spectrogram (ch2)')
        ax[3].plot(f_spec,spec[:,10,0],'-',label='ch 1')
        ax[3].plot(f_spec,spec[:,10,1],'-',label='ch 2')
        ax[3].set(ylabel='Power',xlabel='Frequency (Hz)',xlim=(0,50),title='Power spectral')
        ax[3].legend(title=f't = {t_spec[10]}s',frameon=False, fontsize=7)
        filename = 'tfspec.png'
        savefig(docs_dir,filename)
        
        NW = 0.075
        BW = 20
        n, p, k = aopy.precondition.convert_taper_parameters(NW, BW)
        step = None
        fk = 1000
        samplerate = 1000
        
        t = np.arange(2*samplerate)/samplerate
        f0 = 1
        t1 = 2
        f1 = 1000
        data = 1e-6*np.expand_dims(signal.chirp(t, f0, t1, f1, method='quadratic', phi=0),1)

        fig, ax = plt.subplots(3,1,figsize=(4,6),tight_layout=True)
        aopy.visualization.plot_timeseries(data, samplerate, ax=ax[0])
        aopy.visualization.plot_freq_domain_amplitude(data, samplerate, ax=ax[1])
        f_spec,t_spec,spec = aopy.analysis.calc_mt_tfr(data, n, p, k, samplerate, step=step, fk=fk, pad=2, ref=False)
        pcm = aopy.visualization.plot_tfr(spec[:,:,0], t_spec, f_spec, 'plasma', ax=ax[2])
        fig.colorbar(pcm, label='Power', orientation = 'horizontal', ax=ax[2])
        filename = 'tfr_mt_chirp.png'
        savefig(docs_dir,filename)
        
    def test_tfr_wavelets(self):
        fig, ax = plt.subplots(3,1,figsize=(4,6))

        samplerate = 1000
        data_200_hz = aopy.utils.generate_multichannel_test_signal(2, samplerate, 8, 200, 2)
        nt = data_200_hz.shape[0]
        data_200_hz[:int(nt/3),:] /= 3
        data_200_hz[int(2*nt/3):,:] *= 2

        data_50_hz = aopy.utils.generate_multichannel_test_signal(2, samplerate, 8, 50, 2)
        data_50_hz[:int(nt/2),:] /= 2

        data = data_50_hz + data_200_hz
        print(data.shape)
        aopy.visualization.plot_timeseries(data, samplerate, ax=ax[0])
        aopy.visualization.plot_freq_domain_amplitude(data, samplerate, ax=ax[1])

        freqs = np.linspace(1,250,100)
        coef = aopy.analysis.calc_cwt_tfr(data, freqs, samplerate, fb=10, f0_norm=1, verbose=True)
        t = np.arange(nt)/samplerate
        
        print(data.shape)
        print(coef.shape)
        print(t.shape)
        print(freqs.shape)
        pcm = aopy.visualization.plot_tfr(abs(coef[:,:,0]), t, freqs, 'plasma', ax=ax[2])

        fig.colorbar(pcm, label='Power', orientation = 'horizontal', ax=ax[2])
        filename = 'tfr_cwt_50_200.png'
        plt.tight_layout()
        savefig(docs_dir,filename)


class BehaviorMetricsTests(unittest.TestCase):

    def test_calc_success_percent(self):
        # Test integer events
        events = [0, 2, 4, 6, 0, 2, 3, 6]
        start_evt = 0
        end_events = [3, 6]
        reward_evt = 3
        success_perc = aopy.analysis.calc_success_percent(events, start_evt, end_events, reward_evt)
        self.assertEqual(success_perc, 0.5)
        # Test string events
        events = [b"TARGET_ON", b"TARGET_OFF", b"TRIAL_END", b"TARGET_ON", b"TARGET_ON", b"TARGET_OFF", b"REWARD"]
        start_events = [b"TARGET_ON"]
        end_events = [b"REWARD", b"TRIAL_END"]
        success_events = [b"REWARD"]
        success_perc = aopy.analysis.calc_success_percent(events, start_events, end_events, success_events)
        self.assertEqual(success_perc, 0.5)

        # Test rolling success percent calculation
        events = [0,2,6, 0,3, 0,2,6, 0,2,6, 0,3, 0,2,6, 0,2,6, 0,3, 0,2,6, 0,2,6, 0,3, 0,2,6]
        ntrials = 12
        window_size = 3
        start_evt = 0
        end_events = [3, 6]
        reward_evt = 3
        expected_success_perc = np.ones(ntrials)*(1/3)
        success_perc = aopy.analysis.calc_success_percent(events, start_evt, end_events, reward_evt, window_size=window_size)
        np.testing.assert_allclose(success_perc, expected_success_perc)

        # Test calling the trial helper function directly with trial separated data
        trial_success = [0, 1, 1, 0]
        success_perc = aopy.analysis.calc_success_percent_trials(trial_success)
        self.assertEqual(success_perc, 0.5)

    def test_calc_success_rate(self):
        # Test integer events
        events = [0, 2, 4, 6, 0, 2, 3, 6]
        event_times = np.arange(len(events))
        start_evt = 0
        end_events = [3, 6]
        reward_evt = 3
        success_rate = aopy.analysis.calc_success_rate(events, event_times, start_evt, end_events, reward_evt)
        self.assertEqual(success_rate, 1/5)
        # Test string events
        events = [b"TARGET_ON", b"TARGET_OFF", b"TRIAL_END", b"TARGET_ON", b"TARGET_ON", b"TARGET_OFF", b"REWARD"]
        start_events = [b"TARGET_ON"]
        end_events = [b"REWARD", b"TRIAL_END"]
        success_events = [b"REWARD"]
        success_rate = aopy.analysis.calc_success_rate(events,event_times, start_events, end_events, success_events)
        self.assertEqual(success_rate, 1/4)

        # Test rolling success rate calculation
        events = [0,2,6, 0,3, 0,2,6, 0,2,6, 0,3, 0,2,6, 0,2,6, 0,3, 0,2,6, 0,2,6, 0,3, 0,2,6]
        event_times = np.arange(len(events))
        ntrials = 12
        window_size = 3
        start_evt = 0
        end_events = [3, 6]
        reward_evt = 3
        expected_success_rate = np.ones(ntrials)*(1/5)
        success_perc = aopy.analysis.calc_success_rate(events,event_times, start_evt, end_events, reward_evt, window_size=window_size)
        print(success_perc)
        np.testing.assert_allclose(success_perc, expected_success_rate)

        # Test calling the trial helper function directly with trial separated data
        trial_success = [0, 1, 1, 0]
        trial_time = [0.5, 0.3, 0.1, 0.2]
        success_perc = aopy.analysis.calc_success_rate_trials(trial_success, trial_time)
        self.assertEqual(success_perc, 2/1.1)

    def test_compute_movement_error(self):
        traj = np.array([
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 0.8, 1.2, 2.0, 2.5]
        ]).T

        target_position = np.array([1, 0])
        rotation_axis = np.array([1, 0])
        
        dist = aopy.analysis.compute_movement_error(traj, target_position, rotation_axis)
        np.testing.assert_array_almost_equal(dist, traj[:,1])

        # target_position with all zero values
        target_position = np.array([0, 0])
        self.assertRaises(AssertionError, lambda: aopy.analysis.compute_movement_stats(traj, target_position, rotation_axis))

        # target position flipping 180 degrees
        target_position = np.array([-1, 0])
        traj = np.array([
            [-0.5, -1.0, -1.5, -2.0, -2.5],
            [-0.5, -0.8, -1.2, 0.0, 2.5]
        ]).T
                    
        dist = aopy.analysis.compute_movement_error(traj, target_position, rotation_axis)
        np.testing.assert_array_almost_equal(dist, -traj[:,1])

    def test_compute_movement_stats(self):
        traj = np.array([
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 0.8, 1.2, 2.0, 2.5]
        ]).T

        target_position = np.array([1, 0])
        rotation_axis = np.array([1, 0])
        
        mean, std, auc = aopy.analysis.compute_movement_stats(traj, target_position, rotation_axis)
        
        np.testing.assert_array_almost_equal(mean, np.mean(traj[:,1]))
        np.testing.assert_array_almost_equal(std, np.std(traj[:,1]))
        np.testing.assert_array_almost_equal(auc, np.sum(traj[:,1]))

        (mean, std, auc, abs_mean, abs_min, abs_max, abs_auc, sign, signed_min, signed_max, 
         signed_abs_mean) = aopy.analysis.compute_movement_stats(traj, target_position, rotation_axis,
                                                                 return_all_stats=True)

        np.testing.assert_array_almost_equal(abs_mean, np.mean(abs(traj[:,1])))
        np.testing.assert_array_almost_equal(abs_min, np.min(abs(traj[:,1])))
        np.testing.assert_array_almost_equal(abs_max, np.max(abs(traj[:,1])))
        np.testing.assert_array_almost_equal(abs_auc, np.sum(abs(traj[:,1])))

        np.testing.assert_equal(sign, 1)
        np.testing.assert_array_almost_equal(signed_min, np.min(traj[:,1]))
        np.testing.assert_array_almost_equal(signed_max, np.max(traj[:,1]))
        np.testing.assert_array_almost_equal(signed_abs_mean, 1*np.mean(abs(traj[:,1])))

        # target_position with all zero values
        target_position = np.array([0, 0])
        self.assertRaises(AssertionError, lambda: aopy.analysis.compute_movement_stats(traj, target_position, rotation_axis))

        # target position flipping 180 degrees
        target_position = np.array([-1, 0])
        traj = np.array([
            [-0.5, -1.0, -1.5, -2.0, -2.5],
            [-0.5, -0.8, -1.2, 0.0, 2.5]
        ]).T
                    
        mean, std, auc = aopy.analysis.compute_movement_stats(traj, target_position, rotation_axis)
        
        np.testing.assert_array_almost_equal(mean, np.mean(-traj[:,1]))
        np.testing.assert_array_almost_equal(std, np.std(-traj[:,1]))
        np.testing.assert_array_almost_equal(auc, np.sum(-traj[:,1]))

        (mean, std, auc, abs_mean, abs_min, abs_max, abs_auc, sign, signed_min, signed_max, 
         signed_abs_mean) = aopy.analysis.compute_movement_stats(traj, target_position, rotation_axis,
                                                                 return_all_stats=True)

        np.testing.assert_array_almost_equal(abs_mean, np.mean(abs(-traj[:,1])))
        np.testing.assert_array_almost_equal(abs_min, np.min(abs(-traj[:,1])))
        np.testing.assert_array_almost_equal(abs_max, np.max(abs(-traj[:,1])))
        np.testing.assert_array_almost_equal(abs_auc, np.sum(abs(-traj[:,1])))

        np.testing.assert_equal(sign, -1)
        np.testing.assert_array_almost_equal(signed_min, np.max(-traj[:,1]))
        np.testing.assert_array_almost_equal(signed_max, np.min(-traj[:,1]))
        np.testing.assert_array_almost_equal(signed_abs_mean, -1*np.mean(abs(-traj[:,1])))


    def test_compute_path_length_per_trajectory(self):
        pts = [(0,0), (0,1), (3,1), (3,0)]
        path_length = aopy.analysis.compute_path_length_per_trajectory(pts)
        self.assertEqual(path_length, 5.0)

    def test_time_to_target(self):
        events =  [[80, 17, 32, 81, 48],
                   [80, 23, 32, 87, 48] ,
                   [80, 19, 32, 83, 48] ,
                   [80, 18, 32, 82, 48],
                   [80, 22, 32, 86, 128], # unrewarded trial should be filtered out
                   [80, 17, 32, 81, 48],]
        times = [[ 74., 74., 74., 75., 75.],
                 [ 97., 97., 97., 99., 99.],
                 [115., 115., 115., 118., 118.],
                 [135., 135., 135., 139., 139.],
                 [144., 144., 144., 149., 149.],
                 [154., 154., 154., 160., 160.]]
        rt, target_dir = aopy.analysis.time_to_target(events, times)
        np.testing.assert_allclose(rt, [1., 2., 3., 4., 6.]) # difference from go cue to entering peripheral target, skipping unrewarded trial
        np.testing.assert_allclose(target_dir, [81, 87, 83, 82, 81]) # there are two appearances of target 1


    def test_calc_segment_duration(self):
        events =  [80, 17, 32, 81, 48,
                   80, 23, 32, 87, 48,
                   80, 19, 32, 83, 48,
                   80, 18, 32, 82, 48,
                   80, 22, 32, 86, 128, # unrewarded trial should be filtered out
                   80, 17, 32, 81, 48]
        times = [74., 74., 74., 75., 75.,
                 97., 97., 97., 99., 99.,
                 115., 115., 115., 118., 118.,
                 135., 135., 135., 139., 139.,
                 144., 144., 144., 149., 149.,
                 154., 154., 154., 160., 160.]
        rt, target_idx = aopy.analysis.calc_segment_duration(events, times, [32], [48, 128])
        np.testing.assert_allclose(rt, [1., 2., 3., 4., 5., 6.]) # difference from go cue to entering peripheral target
        np.testing.assert_allclose(target_idx, [0, 6, 2, 1, 5, 0])

        # With filtering out unsuccessful trials
        rt, target_idx = aopy.analysis.calc_segment_duration(events, times, [32], [48, 128], trial_filter=lambda x: 48 in x)
        np.testing.assert_allclose(rt, [1., 2., 3., 4., 6.]) # difference from go cue to entering peripheral target
        np.testing.assert_allclose(target_idx, [0, 6, 2, 1, 0])


if __name__ == "__main__":
    unittest.main()


