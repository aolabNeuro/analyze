from aopy.analysis import calc_success_rate
import aopy
import numpy as np
import warnings
import unittest

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
        expected_success_perc = np.ones(ntrials-window_size+1)*(1/3)
        success_perc = aopy.analysis.calc_success_percent(events, start_evt, end_events, reward_evt, window_size=window_size)
        np.testing.assert_allclose(success_perc, expected_success_perc)

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
        expected_success_rate = np.ones(ntrials-window_size+1)*(1/5)
        success_perc = aopy.analysis.calc_success_rate(events,event_times, start_evt, end_events, reward_evt, window_size=window_size)
        print(success_perc)
        np.testing.assert_allclose(success_perc, expected_success_rate)

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


if __name__ == "__main__":
    unittest.main()
