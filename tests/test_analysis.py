import time
from aopy.visualization import savefig
from aopy.analysis import controllers, latency
import aopy
import os
import numpy as np
import warnings
import unittest
import scipy

import os
import matplotlib.pyplot as plt
from scipy import signal
from unittest.mock import MagicMock

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

class TestTuning(unittest.TestCase):

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


    def test_get_mean_fr_per_condition(self):
        ntime = 4
        nch = 3
        ntrials = 10
        ncond = 2
        data = np.random.normal(0,0.0001,size=(ntime, nch, ntrials))
        cond_labels = np.zeros(ntrials)
        cond_labels[6:] = 1
        data[:,1,cond_labels==1] = 1
        means, std, pvalue = aopy.analysis.get_mean_fr_per_condition(data, cond_labels, return_significance=True)
        expected_means = np.zeros((nch, ncond))
        expected_std = np.zeros((nch, ncond))
        expected_means[1,1] = 1
        np.testing.assert_allclose(means, expected_means, atol=0.001)
        np.testing.assert_allclose(std, expected_std, atol=0.001)
        np.testing.assert_equal(len(pvalue), nch)
        np.testing.assert_equal(pvalue[0]>0.05, True)
        np.testing.assert_equal(pvalue[1]<0.05, True)
        np.testing.assert_equal(pvalue[2]>0.05, True)

    def test_calc_dprime(self):

        # Single channel test
        np.random.seed(0)
        noise_dist = np.random.normal(0, 1, 1000)
        signal_dist = np.random.normal(1, 1, 1000)
        
        dprime = aopy.analysis.calc_dprime(noise_dist, signal_dist)

        # Recalculate by hand
        m1 = np.mean(noise_dist, axis=0)
        m2 = np.mean(signal_dist, axis=0)
        s1 = np.std(noise_dist, axis=0)
        s2 = np.std(signal_dist, axis=0)
        mean_diff = m2 - m1
        sd_sum = (len(noise_dist) * s1 + (len(signal_dist) * s2))
        sd_pooled = sd_sum/(len(noise_dist) + len(signal_dist))
        dprime_hand = mean_diff / sd_pooled

        np.testing.assert_allclose(dprime, dprime_hand)

        # Simple multi-channel test
        noise_dist = np.array([[0, 1], [0, 1], [0, 1]]).T
        signal_dist = np.array([[0.5, 1.5], [1, 2], [1.5, 2.5]]).T
        dprime = aopy.analysis.calc_dprime(noise_dist, signal_dist)

        np.testing.assert_allclose(dprime, np.array([1, 2, 3]))

        # Multi-class test
        dist_1 = np.array([0, 1])
        dist_2 = np.array([1, 2])
        dist_3 = np.array([2, 3])
        dprime = aopy.analysis.calc_dprime(dist_1, dist_2, dist_3)

        np.testing.assert_allclose(dprime, 4)

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

    def test_calc_freq_domain_values(self):
        samplerate = 100
        t = np.arange(samplerate) # 1sec signal
        fig, ax = plt.subplots(4,1, figsize=(10,10))

        # signal 1
        A1 = 1 # amplitude
        f1 = 2 # Hz
        p1 = 0 # phase
        y1 = A1 * np.sin((2*np.pi)*(f1/samplerate)*t + p1)
        ax[0].plot(t, y1)
        ax[0].set_ylabel('magnitude'); ax[0].set_xlabel('sample')

        freqs, freqvalues = aopy.analysis.calc_freq_domain_values(y1, samplerate)
        self.assertAlmostEqual(abs(freqvalues[freqs==f1,0])[0], A1)
        self.assertAlmostEqual(np.angle(freqvalues[freqs==f1,0], deg=True)[0], -90)
        ax[1].plot(freqs, abs(freqvalues), '-o')
        ax[1].set_ylabel('magnitude')

        # signal 2
        A2 = 2
        f2 = 5
        p2 = np.pi/2
        y2 = A2 * np.sin((2*np.pi)*(f2/samplerate)*t + p2)
        ax[0].plot(t, y2)
        
        freqs, freqvalues = aopy.analysis.calc_freq_domain_values(y2, samplerate)
        self.assertAlmostEqual(abs(freqvalues[freqs==f2,0])[0], A2)
        self.assertAlmostEqual(np.angle(freqvalues[freqs==f2,0], deg=True)[0], 0)
        ax[2].plot(freqs, abs(freqvalues), '-o', color='tab:orange')
        ax[2].set_ylabel('magnitude')

        # signal 1 + signal 2
        ax[0].plot(t, y1+y2)

        freqs, freqvalues = aopy.analysis.calc_freq_domain_values(y1+y2, samplerate)
        self.assertAlmostEqual(abs(freqvalues[freqs==f1,0])[0], A1)
        self.assertAlmostEqual(np.angle(freqvalues[freqs==f1,0], deg=True)[0], -90)
        self.assertAlmostEqual(abs(freqvalues[freqs==f2,0])[0], A2)
        self.assertAlmostEqual(np.angle(freqvalues[freqs==f2,0], deg=True)[0], 0)
        ax[3].plot(freqs, abs(freqvalues), '-o', color='tab:green')
        ax[3].set_ylabel('magnitude'); ax[3].set_xlabel('frequency')

        filename = 'freq_domain_decomposition.png'
        aopy.visualization.savefig(docs_dir, filename)

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

        # Check mode 'nan'
        data_convolved = aopy.analysis.calc_rolling_average(data, kernel_size, mode='nan')
        np.testing.assert_array_almost_equal(data_convolved, np.array([np.nan, 2., 3., 4., np.nan]))

        # Check if data is smaller than window
        kernel_size = 7
        data_convolved = aopy.analysis.calc_rolling_average(data, kernel_size)
        np.testing.assert_array_almost_equal(data_convolved, np.mean(data)*np.ones(len(data)))
        data_convolved = aopy.analysis.calc_rolling_average(data, kernel_size, mode='nan')
        np.testing.assert_array_almost_equal(data_convolved, np.nan*np.ones(len(data)))

        # Test with 2D data
        data = np.array([[1, 2, 3, 4, 5], 
                         [2, 3, 4, 5, 6]]).T
        kernel_size = 3
        expected = np.array([[2, 2, 3, 4, 4],
                             [3, 3, 4, 5, 5]]).T
        data_convolved = aopy.analysis.calc_rolling_average(data, kernel_size)
        np.testing.assert_array_almost_equal(data_convolved, expected)


    def test_calc_corr_over_elec_distance(self):
        elec_data = np.array([[1, 2, 3], [4, 5, 6]]).T
        elec_pos = np.array(
            [[1, 1],
            [2, 2]]
        )
        dist, corr = aopy.analysis.calc_corr_over_elec_distance(elec_data, elec_pos, method='pearson', bins=1, exclude_zero_dist=True)

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
        self.assertEqual(erp.shape[2], 3)

        mean_erp = np.mean(erp, axis=2)
        self.assertEqual(np.sum(mean_erp[:,0]), 1)
        self.assertEqual(np.sum(mean_erp[:,1]), 2)

        # Subtract baseline
        data += 1
        erp = aopy.analysis.calc_erp(data, event_times, 0.1, 0.1, samplerate)
        mean_erp = np.mean(erp, axis=2)
        self.assertEqual(np.sum(mean_erp[:,0]), 1)
        self.assertEqual(np.sum(mean_erp[:,1]), 2)

        # Specify baseline window
        data[0] = 100
        erp = aopy.analysis.calc_erp(data, event_times, 0.1, 0.1, samplerate, baseline_window=())
        mean_erp = np.mean(erp, axis=2)
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
        self.assertEqual(max_erp.shape[1], len(event_times)) 
        self.assertEqual(max_erp.shape[0], nch) 

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

    def test_get_random_timestamps(self):
        nshuffled_points = 5
        
        # Test random timestamps without time axis
        random_timestamps = aopy.analysis.base.get_random_timestamps(nshuffled_points, max_time=3, min_time=1)
        self.assertTrue(np.min(random_timestamps) >= .999999) # Weird numbers to handle float to int comparison
        self.assertTrue(np.max(random_timestamps) <= 3.00001)
        self.assertTrue(len(random_timestamps)==5)
        self.assertTrue(np.diff(random_timestamps).all()>0)

        # Test random timestamps with time axis
        random_timestamps = aopy.analysis.base.get_random_timestamps(nshuffled_points, max_time=3, min_time=1.1, time_samplerate=10)
        self.assertAlmostEqual(np.sum(random_timestamps%0.1), 0)
        self.assertTrue(np.min(random_timestamps) >= 1.099999)
        self.assertTrue(np.max(random_timestamps) <= 3.00001)
        self.assertTrue(len(random_timestamps)==5)
        self.assertTrue(np.diff(random_timestamps).all()>0)

        # Test that warning message appears
        random_timestamps = aopy.analysis.base.get_random_timestamps(nshuffled_points, max_time=3, min_time=1, time_samplerate=0.1)
        self.assertIsNone(random_timestamps)

    def test_get_empirical_pvalue(self):
        data_distribution = np.random.randn(1000000)

        # Test distribution calculated from data
        # Test two-sided test 
        data_sample = 1
        pvalue = aopy.analysis.base.get_empirical_pvalue(data_distribution, data_sample)
        self.assertAlmostEqual(1, 0.6827+pvalue, 2) # Data should be one standard dev away
        pvalue = aopy.analysis.base.get_empirical_pvalue(data_distribution, np.array((-1,1)))
        self.assertAlmostEqual(1, 0.6827+pvalue[0], 2) # Data should be one standard dev away
        self.assertAlmostEqual(1, 0.6827+pvalue[1], 2) # Data should be one standard dev away

        # Test lower test
        data_sample = -1
        pvalue = aopy.analysis.base.get_empirical_pvalue(data_distribution, data_sample, 'lower')
        self.assertAlmostEqual(1, 0.6827+(2*pvalue), 2) # Data should be one standard dev away but pvalue ismultiplied by 2 b/c single bound test
        pvalue = aopy.analysis.base.get_empirical_pvalue(data_distribution, np.array((-1,1)), 'lower')
        self.assertAlmostEqual(len(pvalue), 2) 

        # Test upper test
        data_sample = 1
        pvalue = aopy.analysis.base.get_empirical_pvalue(data_distribution, data_sample, 'upper')
        self.assertAlmostEqual(1, 0.6827+(2*pvalue), 2) # Data should be one standard dev away but pvalue ismultiplied by 2 b/c single bound test
        pvalue = aopy.analysis.base.get_empirical_pvalue(data_distribution, np.array((-1,1)), 'upper')
        self.assertAlmostEqual(len(pvalue), 2) 

        # Test Gaussian assumption
        # Test two-sided test 
        data_sample = 1
        pvalue = aopy.analysis.base.get_empirical_pvalue(data_distribution, data_sample, assume_gaussian=True)
        self.assertAlmostEqual(1, 0.6827+pvalue, 2) # Data should be one standard dev away
        pvalue = aopy.analysis.base.get_empirical_pvalue(data_distribution, np.array((-1,1)), assume_gaussian=True)
        self.assertAlmostEqual(1, 0.6827+pvalue[0], 2) # Data should be one standard dev away
        self.assertAlmostEqual(1, 0.6827+pvalue[1], 2) # Data should be one standard dev away


        # Test lower test
        data_sample = -1
        pvalue = aopy.analysis.base.get_empirical_pvalue(data_distribution, data_sample, 'lower', assume_gaussian=True)
        self.assertAlmostEqual(1, 0.6827+(2*pvalue), 2) # Data should be one standard dev away but pvalue ismultiplied by 2 b/c single bound test

        # Test upper test
        data_sample = 1
        pvalue = aopy.analysis.base.get_empirical_pvalue(data_distribution, data_sample, 'upper', assume_gaussian=True)
        self.assertAlmostEqual(1, 0.6827+(2*pvalue), 2) # Data should be one standard dev away but pvalue ismultiplied by 2 b/c single bound test

class LatencyTests(unittest.TestCase):

    def test_detect_erp_response(self):
        np.random.seed(0)

        # Make a null baseline and an alternate condition response
        fs = 100
        nt = fs * 2
        nch = 2
        ntr = 10
        null_data = np.random.uniform(-1, 1, (nt, nch, ntr))
        alt_data = np.random.uniform(-1, 1, (nt, nch, ntr)) 
        alt_data[:,1,:] += np.tile(np.expand_dims(np.arange(nt)/10, 1), (1,ntr))

        latency = aopy.analysis.latency.detect_erp_response(null_data, alt_data, fs, 3, debug=True)
        self.assertEqual(latency.shape, (nch, ntr))

        filename = 'detect_erp_response.png'
        aopy.visualization.savefig(docs_dir, filename, transparent=False)

    def test_detect_itpc_response(self):
        np.random.seed(0)

        # Make a null baseline and an alternate condition response
        fs = 100
        nt = fs * 2
        nch = 2
        ntr = 10
        null_data = np.random.uniform(-1, 1, (nt, nch, ntr))
        alt_data = np.random.uniform(-1, 1, (nt, nch, ntr))
        alt_data[:,0,:] += np.tile(np.expand_dims(np.sin(np.arange(nt)*np.pi/10), 1), (1,ntr))
        alt_data[:,1,:] += np.tile(np.expand_dims(10*np.sin(np.arange(nt)*np.pi/10), 1), (1,ntr))

        im_nullcond = signal.hilbert(null_data, axis=0)
        im_altcond = signal.hilbert(alt_data, axis=0)

        latency = aopy.analysis.latency.detect_itpc_response(im_nullcond, im_altcond, fs, 5, debug=True)
        self.assertEqual(latency.shape, (nch,))

        filename = 'detect_itpc_response.png'
        aopy.visualization.savefig(docs_dir, filename, transparent=False)

    def test_detect_accLLR(self, upper=10, lower=-10):
      
        accllr_data = np.array(
            [
                [0, 0, 1, 2, 3, 4, 4, 5, 6, 10, 10, 11],            # even trials
                [0, -1, -1, -2, -3, -4, -5, -5, -6, -10, -11, -11]  # odd trials
            ])
        accllr_data = np.tile(accllr_data.T, (1, 20)) # simulate 40 trials
        self.assertEqual(accllr_data.shape, (12, 40))

        p, st = latency.detect_accllr(accllr_data, upper, lower)

        p_fast, st_fast = latency.detect_accllr_fast(accllr_data, upper, lower)
        
        np.testing.assert_allclose(p, p_fast)
        np.testing.assert_allclose(st, st_fast)

        np.testing.assert_allclose(p, [.5, .5, 0])
        self.assertEqual(st.shape, (40,))

        import time
        t0 = time.perf_counter()
        for i in range(1000):
            latency.detect_accllr(accllr_data, upper, lower)
        t1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        for i in range(1000):
            latency.detect_accllr_fast(accllr_data, upper, lower)
        t2 = time.perf_counter() - t0

        print(f"detect_accllr() took {t1:.2f} s")
        print(f"detect_accllr_fast() took {t2:.2f} s")
        
    def test_delong_roc_variance(self):
        alpha = .95
        y_pred = np.array([0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
        y_true = np.array([0,    1,    0,    0,    1,    1,    0,    1,    0   ])

        auc, auc_cov = latency.calc_delong_roc_variance(
            y_true,
            y_pred)

        # Compare to values in the original paper
        self.assertAlmostEqual(auc, 0.8)
        self.assertAlmostEqual(auc_cov, 0.02875)
    
    def test_calc_accllr_roc(self):
        test_data1 = [0, 1, 2, 4, 6, 7, 3, 4, 5, 16, 7]
        test_data2 = [0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0]
        auc, se = latency.calc_accllr_roc(test_data1, test_data2)

        matlab_auc = 0.921487603
        matlab_se = 0.06396
        
        self.assertAlmostEqual(auc, matlab_auc)
        # self.assertAlmostEqual(se, matlab_se) # known difference between matlab and python version

    def test_llr(self):
        test_lfp = np.array([0, 0.8, 1, 5, 4, 1, 2.5, 3, 4, 4, 4])
        test_data1 = np.array([0, 1, 2, 4, 6, 2, 3, 4, 5, 16, 7])
        test_data2 = np.array([0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0])

        llr = latency.calc_llr_gauss(test_lfp, test_data1, test_data2, np.std(test_data1), np.std(test_data2))

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

        accllr_altcond, accllr_nullcond = latency.calc_accllr_lfp(lfp_altcond, lfp_nullcond,
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
        accllr_altcond, accllr_nullcond = latency.calc_accllr_lfp(lfp_altcond, lfp_nullcond,
                                                        lfp_altcond, lfp_nullcond, 
                                                        common_variance=True)

        nlevels = 200
        p_altcond, p_nullcond, _, levels = latency.calc_accllr_performance(accllr_altcond, accllr_nullcond, nlevels)
        level = latency.choose_best_level(p_altcond, p_nullcond, levels)
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
            latency.calc_accllr_st_single_ch(data_altcond[:,ch,:], data_nullcond[:,ch,:], lowpass_altcond[:,ch,:],
                                 lowpass_nullcond[:,ch,:], 'lfp', 1./samplerate, nlevels=200, verbose_out=True) # try parallel True and False
        
        single_accllr_altcond, single_accllr_nullcond = latency.calc_accllr_lfp(data_altcond[:,ch,:], data_nullcond[:,ch,:],
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
        
        accllr_altcond, accllr_nullcond = latency.calc_accllr_lfp(altcond, nullcond,
                                                        altcond, nullcond)

        p_altcond, p_nullcond, _, _ = latency.calc_accllr_performance(accllr_altcond, accllr_nullcond, 200)
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
        sTime_alt, roc_auc, roc_se, roc_p_fdrc = latency.calc_accllr_st(altcond, nullcond, altcond, nullcond, 'lfp', 1)
        print("Matching selectivities:")
        print(roc_auc)
        self.assertTrue(roc_auc[1] > roc_auc[0])

        # Test wrapper with LFP data and no selectivity matching and trial_average=True
        sTime_alt, roc_auc, roc_se, roc_p_fdrc = latency.calc_accllr_st(altcond, nullcond, altcond, nullcond, 'lfp', 1, 
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
        st, roc_auc, roc_se, roc_p_fdrc = latency.calc_accllr_st(altcond, nullcond, altcond, nullcond, 'lfp', 1)
    
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

        st, roc_auc, roc_se, roc_p_fdrc = latency.calc_accllr_st(altcond, nullcond, altcond_lp, nullcond_lp, 'lfp', 1./samplerate)
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
        st, roc_auc, roc_se, roc_p_fdrc = latency.calc_accllr_st(altcond, nullcond, altcond_lp, nullcond_lp, 'lfp', 1./samplerate, 
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

    def test_prepare_erp(self):
        npts = 100
        nch = 50
        ntrials = 30
        align_idx = 50
        onset_idx = 40
        samplerate = 100
        time_before = align_idx/samplerate
        time_after = time_before
        window_nullcond = (-0.4, -0.1)
        window_altcond = (-0.1, 0.3)
        data = np.ones(npts)*10
        data[onset_idx:] = 10 + np.arange(npts-onset_idx)
        data = np.repeat(np.tile(data, (nch,1)).T[:,:,None], ntrials, axis=2)

        data_altcond, data_nullcond, lowpass_altcond, lowpass_nullcond = latency.prepare_erp(
            data, data, samplerate, time_before, time_after, window_nullcond, window_altcond,
        )

        altcond_dur = window_altcond[1] - window_altcond[0]
        nullcond_dur = window_nullcond[1] - window_nullcond[0]

        self.assertEqual(data_altcond.shape, (int(altcond_dur*samplerate), nch, ntrials))
        self.assertEqual(data_nullcond.shape, (int(nullcond_dur*samplerate), nch, ntrials))
        self.assertEqual(lowpass_altcond.shape, (int(altcond_dur*samplerate), nch, ntrials))
        self.assertEqual(lowpass_nullcond.shape, (int(nullcond_dur*samplerate), nch, ntrials))

        plt.figure()
        plt.subplot(2,1,1)
        time = np.arange(len(data))/samplerate - time_before
        plt.plot(time, data[:,0,0])
        plt.xlabel('time (s)')
        plt.axvspan(*window_nullcond, color='g', alpha=0.25, label='null condition')
        plt.axvspan(*window_altcond, color='m', alpha=0.25, label='alt condition')
        plt.title('full erp')
        plt.legend()

        plt.subplot(2,2,3)
        aopy.visualization.plot_timeseries(data_nullcond[:,0,0], samplerate)
        plt.ylabel('')
        plt.title('null condition')

        plt.subplot(2,2,4)
        aopy.visualization.plot_timeseries(data_altcond[:,0,0], samplerate)
        plt.title('alternate condition')
        plt.ylabel('')
        plt.tight_layout()

        filename = 'prepare_erp_for_accllr.png'
        aopy.visualization.savefig(docs_dir, filename)
        
class HelperFunctions:

    @staticmethod
    def test_tfr_sines(tfr_fun):
        '''
        Tests TFR functions. Input a TFR function with the following signature:
        tfr_fun(ts_data, samplerate) -> (frequency, time, spectrogram)
        '''
        T = 5
        fs = 1000
        num_points = int(T*fs)
        t = np.linspace(0,T,num_points)
        test_data = np.zeros((t.shape[0],2))
        test_data[:,0] = 1.5*np.sin(2*np.pi*10*t) # 10 Hz sine
        test_data[:,1] = 2*np.cos(2*np.pi*30*t) # 30 Hz cosine
        test_data[t>=T/2,0] = 2*np.sin(2*np.pi*5*t[t<=T/2]) # 5 Hz sine
        test_data[t>=T/2,1] = 1*np.cos(2*np.pi*15*t[t<=T/2]) # 15 Hz cosine

        f_spec,t_spec,spec = tfr_fun(test_data, fs)
        test_idx = np.where(t_spec >= 2)[0][0]
        
        fig,ax=plt.subplots(1,4,figsize=(8,2),layout='compressed')
        ax[0].plot(t,test_data[:,0],linewidth=0.2)
        ax[0].plot(t,test_data[:,1],linewidth=0.2)
        ax[0].set(xlabel='Time (s)',ylabel='Amplitude',title='Signals')
        im = ax[1].imshow((spec[:,:,0]),aspect='auto',origin='lower',extent=[0,T,0,f_spec[-1]])
        plt.colorbar(im, ax=ax[1])
        ax[1].set(ylabel='Frequency (Hz)',xlabel='Time [s]',title='Spectrogram (ch1)')
        im = ax[2].imshow((spec[:,:,1]),aspect='auto',origin='lower',extent=[0,T,0,f_spec[-1]])
        plt.colorbar(im, ax=ax[2])
        ax[2].set(ylabel='Frequency (Hz)',xlabel='Time [s]',title='Spectrogram (ch2)')
        ax[3].plot(f_spec,spec[:,test_idx,0],'-',label='ch 1')
        ax[3].plot(f_spec,spec[:,test_idx,1],'-',label='ch 2')
        ax[3].set(ylabel='Power',xlabel='Frequency (Hz)',xlim=(0,50),title='Power spectral')
        ax[3].legend(title=f't = {t_spec[test_idx]}s',frameon=False, fontsize=7)
        
    @staticmethod
    def test_tfr_chirp(tfr_fun):
        '''
        Tests TFR functions. Input a TFR function with the following signature:
        tfr_fun(ts_data, samplerate) -> (frequency, time, spectrogram)
        '''
        samplerate = 1000
        t = np.arange(2*samplerate)/samplerate
        f0 = 1
        t1 = 2
        f1 = 500
        data = np.expand_dims(signal.chirp(t, f0, t1, f1, method='quadratic', phi=0),1)

        f_spec,t_spec,spec = tfr_fun(data, samplerate)

        fig, ax = plt.subplots(2,1, layout='compressed')
        aopy.visualization.plot_timeseries(data, samplerate, ax=ax[0])
        pcm = aopy.visualization.plot_tfr(spec[:,:,0], t_spec, f_spec, 'plasma', ax=ax[1])
        fig.colorbar(pcm, label='power (V)', orientation = 'horizontal', ax=ax[1])

    def test_tfr_lfp(tfr_fun):
        '''
        Tests TFR functions. Input a TFR function with the following signature:
        tfr_fun(ts_data, samplerate) -> (frequency, time, spectrogram)
        '''
        fig, ax = plt.subplots(4,1,figsize=(5,8), layout='compressed')

        # Collect trials of data starting at go-cue
        exp_data, exp_metadata = aopy.data.load_preproc_exp_data(data_dir, 'beignet', 5974, '2022-07-01')
        lfp_data, lfp_metadata = aopy.data.load_preproc_lfp_data(data_dir, 'beignet', 5974, '2022-07-01')
        samplerate = lfp_metadata['lfp_samplerate']
        go_cue = 32
        trial_end = 239
        reward = 48
        trial_times, trial_events = aopy.preproc.get_trial_segments_and_times(exp_data['events']['code'], exp_data['events']['code'], [go_cue], [trial_end])
        go_cues = [t[0] for t, e in zip(trial_times, trial_events) if reward in e]
        time_before = 1.0
        time_after = 2.0
        erp = aopy.analysis.calc_erp(lfp_data[:,0], go_cues, time_before, time_after, samplerate)

        # Plot time domain
        aopy.visualization.plot_timeseries(np.mean(erp, axis=2), samplerate, ax=ax[0])
        ax[0].set_ylabel('voltage (a.u.)')

        # Plot frequency domain
        f, amp = aopy.analysis.calc_freq_domain_amplitude(erp[:,0,:], samplerate)
        amp = np.mean(amp, axis=1)
        ax[1].plot(f, amp)
        ax[1].set_xlabel('frequency (Hz)')
        ax[1].set_ylabel('voltage (a.u.)')

        # Compute and time spectrogram
        freqs = np.linspace(1,200,100)
        t0 = time.perf_counter()
        freqs, times, coef = tfr_fun(erp[:,0,:], samplerate)
        dur = time.perf_counter() - t0

        print(f"{repr(tfr_fun)} took {dur:.3f} seconds")

        # Plot spectrogram
        avg_coef = np.mean(abs(coef), axis=2)
        pcm = aopy.visualization.plot_tfr(avg_coef, times - time_before, freqs, 'plasma', logscale=True, ax=ax[2])
        fig.colorbar(pcm, label='power (log V)', orientation='horizontal', ax=ax[2])

        # Plot beta-band power
        band_power = aopy.analysis.get_tfr_feats(freqs, abs(coef), [(12.5,30)])
        mean_band_power = np.mean(band_power, axis=1)
        ax[3].plot(times - time_before, mean_band_power)
        ax[3].set_xlabel('time (s)')
        ax[3].set_ylabel('beta power (V)')


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
        f, psd_filter, mu = aopy.analysis.calc_mt_psd(self.x, self.fs)
        psd = aopy.analysis.calc_welch_psd(self.x, self.fs, np.shape(f)[0])[1]

        fname = 'multitaper_powerspectrum.png'
        plt.figure()
        plt.plot(f, psd, label='Welch')
        plt.plot(f, psd_filter, label='Multitaper')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.legend()
        savefig(write_dir, fname) # both figures should have peaks at [600, 312, 2000] Hz

        bands = [(0, 10), (250, 350), (560, 660), (2000, 2010), (4000, 4100)]
        lfp = aopy.analysis.get_tfr_feats(f, psd_filter, bands, False)
        plt.figure()
        plt.plot(np.arange(len(bands)), np.squeeze(lfp), '-bo')
        plt.xticks(np.arange(len(bands)), bands)
        plt.xlabel('Frequency band (Hz)')
        plt.ylabel('Band Power')
        fname = 'lfp_bandpower.png'
        savefig(write_dir, fname) # Should have power in [600, 312, 2000] Hz but not 10 or 4000

        f, psd_filter, mu = aopy.analysis.calc_mt_psd(self.x2, self.fs)
        self.assertEqual(psd_filter.shape[1], self.n_ch)
        print(mu.shape)
        lfp = aopy.analysis.get_tfr_feats(f, psd_filter, bands, False)
        self.assertEqual(lfp.shape[1], self.x2.shape[1])
        self.assertEqual(lfp.shape[0], len(bands))

    def test_tfr_ft(self):
        win_t = 0.5
        step = 0.01
        f_max = 50
        tfr_fun = lambda data, fs: aopy.analysis.calc_ft_tfr(data, fs, win_t, step, f_max, pad=3, window=('tukey', 0.5))
        filename = 'tfr_ft_sines.png'
        HelperFunctions.test_tfr_sines(tfr_fun)
        savefig(docs_dir,filename)
        
        f_max = 500
        filename = 'tfr_ft_chirp.png'
        tfr_fun = lambda data, fs: aopy.analysis.calc_ft_tfr(data, fs, win_t, step, f_max, pad=3, window=('tukey', 0.5))
        HelperFunctions.test_tfr_chirp(tfr_fun)
        savefig(docs_dir,filename)
        
        f_max = 200
        tfr_fun = lambda data, fs: aopy.analysis.calc_ft_tfr(data, fs, win_t, step, f_max, pad=3, window=('tukey', 0.5))
        HelperFunctions.test_tfr_lfp(tfr_fun)
        filename = 'tfr_ft_lfp.png'
        savefig(docs_dir,filename)
        
    def test_tfr_mt(self):
        NW = 0.3
        BW = 10
        step = 0.01
        fk = 50
        n, p, k = aopy.precondition.convert_taper_parameters(NW, BW)
        print(f"using {k} tapers length {n} half-bandwidth {p}")
        tfr_fun = lambda data, fs: aopy.analysis.calc_mt_tfr(data, n, p, k, fs, step=step, fk=fk, pad=2, ref=False)
        filename = 'tfspec.png'
        HelperFunctions.test_tfr_sines(tfr_fun)
        savefig(docs_dir,filename)
        
        fk = 500
        filename = 'tfr_mt_chirp.png'
        tfr_fun = lambda data, fs: aopy.analysis.calc_mt_tfr(data, n, p, k, fs, step=step, fk=fk, pad=2, ref=False)
        HelperFunctions.test_tfr_chirp(tfr_fun)
        savefig(docs_dir,filename)
        
        fk = 200
        tfr_fun = lambda data, fs: aopy.analysis.calc_mt_tfr(data, n, p, k, fs, step=step, fk=fk, pad=2, ref=False, dtype='int16')
        HelperFunctions.test_tfr_lfp(tfr_fun)
        filename = 'tfr_mt_lfp.png'
        savefig(docs_dir,filename)

        # Test two alignments
        fk = 100
        fig, ax = plt.subplots(3,1,figsize=(5,8), layout='compressed')

        # Make some test data with change at t=0
        T = 2
        fs = 1000
        num_points = int(T*fs)
        t = np.linspace(0,T,num_points)
        erp = np.zeros((t.shape[0],))
        erp = np.sin(2*np.pi*20*t) # 20 Hz sine
        time_before = T/2
        erp[t>=time_before] = np.sin(2*np.pi*80*t[t>=time_before]) # 80 Hz sine

        # Plot time domain
        ax[0].plot(t - time_before, erp)
        ax[0].set_ylabel('Voltage (a.u.)')
        ax[0].set_xlabel('Time (s)')

        # Compute and time spectrogram
        t0 = time.perf_counter()
        freqs, times, coef = aopy.analysis.calc_mt_tfr(erp, n, p, k, fs, step=step, fk=fk, pad=2, ref=False)
        dur = time.perf_counter() - t0

        print(f"{repr(tfr_fun)} took {dur:.3f} seconds")

        # Plot spectrogram
        pcm = aopy.visualization.plot_tfr(abs(coef[:,:,0]), times - time_before, freqs, 'plasma', ax=ax[1])
        ax[1].set_title('Align center (default)')

        pcm = aopy.visualization.plot_tfr(abs(coef[:,:,0]), times - time_before + n/2, freqs, 'plasma', ax=ax[2])
        fig.colorbar(pcm, label='power (a.u.)', orientation='horizontal', ax=ax[2])
        ax[2].set_title('Align right (time += n/2)')
        filename = 'tfr_mt_alignment.png'
        savefig(docs_dir,filename)

    def test_tfr_mt_tsa(self):
        win_t = 0.3
        step_t = 0.01
        bw = 20
        fk = 50
        tfr_fun = lambda data, fs: aopy.analysis.calc_tsa_mt_tfr(data, fs, win_t, step_t, bw=bw, f_max=fk)
        filename = 'tfr_mt_tsa_sines.png'
        HelperFunctions.test_tfr_sines(tfr_fun)
        savefig(docs_dir,filename)
        
        fk = 500
        filename = 'tfr_mt_tsa_chirp.png'
        tfr_fun = lambda data, fs: aopy.analysis.calc_tsa_mt_tfr(data, fs, win_t, step_t, bw=bw, f_max=fk)
        HelperFunctions.test_tfr_chirp(tfr_fun)
        savefig(docs_dir,filename)
        
        fk = 200
        tfr_fun = lambda data, fs: aopy.analysis.calc_tsa_mt_tfr(data, fs, win_t, step_t, bw=bw, f_max=fk)
        HelperFunctions.test_tfr_lfp(tfr_fun)
        filename = 'tfr_mt_tsa_lfp.png'
        savefig(docs_dir,filename)

    def test_tfr_wavelets(self):
        fb = 10.
        f0_norm = 2.
        freqs = np.linspace(1,50,50)
        tfr_fun = lambda data, fs: aopy.analysis.calc_cwt_tfr(data, freqs, fs, fb=fb, f0_norm=f0_norm, verbose=True)
        filename = 'tfr_cwt_sines.png'
        HelperFunctions.test_tfr_sines(tfr_fun)
        savefig(docs_dir,filename)
        
        freqs = np.linspace(1,500,500)
        tfr_fun = lambda data, fs: aopy.analysis.calc_cwt_tfr(data, freqs, fs, fb=fb, f0_norm=f0_norm, verbose=True)
        filename = 'tfr_cwt_chirp.png'
        HelperFunctions.test_tfr_chirp(tfr_fun)
        savefig(docs_dir,filename)
        
        freqs = np.linspace(1,200,200)
        tfr_fun = lambda data, fs: aopy.analysis.calc_cwt_tfr(data, freqs, fs, fb=fb, f0_norm=f0_norm, verbose=True)
        filename = 'tfr_cwt_lfp.png'
        HelperFunctions.test_tfr_lfp(tfr_fun)
        savefig(docs_dir,filename)
        
    def test_coherency(self):
        fs = 1000
        N = 1e5
        T = N/fs
        amp = 1
        noise_power = 0.001 * fs / 2
        time = np.arange(N) / fs

        rng = np.random.default_rng(seed=0)
        signal1 = rng.normal(scale=np.sqrt(noise_power), size=time.shape)

        b, a = scipy.signal.butter(2, 0.25, 'low')
        signal2 = scipy.signal.lfilter(b, a, signal1)
        signal2 += rng.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)

        # Add a 100 hz sine wave only to signal 1
        freq = 100.0
        signal1[time > T/2] += amp*np.sin(2*np.pi*freq*time[time > T/2])

        # Add a 400 hz sine wave to both signals
        freq = 400.0
        signal1[time < T/2] += amp*np.sin(2*np.pi*freq*time[time < T/2])
        signal2[time < T/2] += amp*np.sin(2*np.pi*freq*time[time < T/2])

        # Add a 200 hz sine wave to both signals but with phase modulated by a 0.05 hz sine wave
        freq = 200.0
        freq2 = 0.05
        signal1 += amp*np.sin(2*np.pi*freq*time)
        signal2 += amp*np.sin(2*np.pi*freq*time + np.pi*np.sin(2*np.pi*freq2*time))

        # Calculate mt coh
        n = 1
        w = 2
        n, p, k = aopy.precondition.convert_taper_parameters(n, w)
        print(k)
        # k = k-1
        fk = fs / 2  # Maximum frequency of interest
        step = n # no overlap
        signal_combined = np.stack((signal1, signal2), axis=1)

        # Calculate spectrograms for each signal
        f, t, spec1 = aopy.analysis.calc_mt_tfr(signal1, n, p, k, fs, step, fk=fk,
                                                              ref=False)
        f, t, spec2 = aopy.analysis.calc_mt_tfr(signal2, n, p, k, fs, step, fk=fk,
                                                              ref=False)

        # Calculate coherence
        f, t, coh = aopy.analysis.calc_mt_tfcoh(signal_combined, [0,1], n, p, k, fs, step, fk=fk,
                                                              ref=False)
        f, t, coh_im, angle = aopy.analysis.calc_mt_tfcoh(signal_combined, [0,1], n, p, k, fs, step, fk=fk,
                                                              ref=False, imaginary=True, return_angle=True)
        
        # Calculate coherency from scipy
        f_scipy, coh_scipy = scipy.signal.coherence(signal1, signal2, fs=fs, nperseg=fs*n, noverlap=0, axis=0)

        # Plot the coherence over time
        plt.figure(figsize=(10, 15))
        plt.subplot(5, 1, 1)
        im = aopy.visualization.plot_tfr(spec1[:,:,0], t, f)
        plt.colorbar(im, orientation='horizontal', location='top', label='Signal 1')
        im.set_clim(0,3)

        plt.subplot(5, 1, 2)
        im = aopy.visualization.plot_tfr(spec2[:,:,0], t, f)
        plt.colorbar(im, orientation='horizontal', location='top', label='Signal 2')
        im.set_clim(0,3)

        plt.subplot(5, 1, 3)
        im = aopy.visualization.plot_tfr(coh, t, f)
        plt.colorbar(im, orientation='horizontal', location='top', label='Coherence')
        im.set_clim(0,1)

        # Plot the average coherence across windows
        plt.subplot(5, 1, 4)
        plt.plot(f, np.mean(coh, axis=1))
        plt.plot(f, np.mean(coh_im, axis=1))
        plt.plot(f_scipy, coh_scipy)
        plt.title('Average coherence across time')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Coherency')
        plt.legend(['coh', 'imag coh', 'scipy'])

        # Also plot the phase difference
        plt.subplot(5, 1, 5)
        im = aopy.visualization.plot_tfr(angle, t, f, cmap='bwr')
        plt.colorbar(im, orientation='horizontal', location='top', label='Phase difference (rad)')
        im.set_clim(-np.pi,np.pi)

        plt.tight_layout()
        figname = 'coherency.png'
        aopy.visualization.savefig(docs_dir, figname)

        # Also test some other features
        f, t, coh = aopy.analysis.calc_mt_tfcoh(signal_combined, [0,1], n, p, k, fs, step, fk=fk,
                                                        ref=True, workers=2, dtype='int16')

    def test_calc_itpc(self):
        np.random.seed(0)

        # Generate some test data with varying phase consistency
        fs = 1000
        nt = fs * 2
        ntr = 100
        t = np.arange(nt)/fs
        data = np.zeros((t.shape[0],2,ntr)) # 2 channels

        # 10 Hz sine with gaussian phase distribution across trials
        for tr in range(ntr):
            data[:,0,tr] = np.sin(2*np.pi*10*t + np.random.normal(np.pi/4, np.pi/8)) 

        # 10 Hz sine with uniform random phase distribution across trials
        for tr in range(ntr):
            data[:,1,tr] = np.sin(2*np.pi*10*t + np.random.uniform(-np.pi, np.pi)) 

        # Calculate an analytical signal using hilbert transform
        im_data = signal.hilbert(data, axis=0)
        itpc = aopy.analysis.calc_itpc(im_data)

        plt.figure()

        # Plot the data
        plt.subplot(3,1,1)
        aopy.visualization.plot_timeseries(np.mean(data, axis=2), fs)
        plt.legend(['Channel 1', 'Channel 2'])
        plt.ylabel('amplitude (a.u.)')
        plt.title('Trial averaged data')

        # Plot the angles at the first timepoint
        angles = np.angle(im_data[0])
        plt.subplot(3,2,3, projection= 'polar')
        aopy.visualization.plot_angles(angles[0,:], color='tab:blue', alpha=0.5, linewidth=0.75)
        plt.subplot(3,2,4, projection= 'polar')
        aopy.visualization.plot_angles(angles[1,:], color='tab:orange', alpha=0.5, linewidth=0.75)

        # Plot ITPC
        plt.subplot(3,1,3)
        aopy.visualization.plot_timeseries(itpc, fs)
        plt.ylabel('ITPC')
        plt.title('ITPC')

        plt.tight_layout()
        filename = 'itpc.png'
        aopy.visualization.savefig(docs_dir, filename, transparent=False)


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

    def test_movement_onset_and_cursor_leave_time(self):
        fs = 1000
        duration = 10
        t = np.arange(duration*fs)/fs
        y1 = 10*np.sin(2*np.pi*30*t) + 0.1*np.sin(2*np.pi*5*t)
        y2 = np.zeros(t.shape[0])
        y2[t>5] = 5*np.sin(2*np.pi*1*t[t>5])
        yz = np.zeros(t.shape[0])
        cursor_test = np.array([np.stack([y1+y2,yz]).T])
        
        trial_start = np.array([0])
        target_onset= np.array([2])
        gocue = np.array([4])
        movement_onset = aopy.analysis.get_movement_onset(cursor_test, fs, trial_start, target_onset, gocue, numsd=20.0, butter_order=4, low_cut=20, thr=None)
        self.assertTrue((movement_onset > 5)*(movement_onset < 5.1))
        
        fs = 1
        cursor_test = np.array([np.array([[0,0,0,0,0,1,1,1,1,1],[0,0.5,0.5,0,0,1,1,1,1,1,]]).T,\
            np.array([[0.5,0,0,0,0,0,0,-1,-1,-1],[0,0.5,0,0,0,0,0,1,1,1,]]).T])
        cursor_leave_time = aopy.analysis.get_cursor_leave_time(cursor_test, fs, 0.8)
        self.assertTrue(np.all(cursor_leave_time == np.array([5,7])))

    def test_calc_tracking_error(self):
        samplerate = 100
        t = np.arange(samplerate*20) # 20sec signal
        exp_freqs = [.2, .5] # [f1, f2] Hz
        
        A1 = 4
        A2 = 3

        offset = 0
        target_traj = A1 * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t) + A2 * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t)
        cursor_traj = (A1+offset) * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t) + (A2+offset) * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t)
        self.assertAlmostEqual(offset**2, aopy.analysis.calc_tracking_error(cursor_traj, target_traj))

        offset = 1
        target_traj = A1 * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t) + A2 * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t)
        cursor_traj = (A1+offset) * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t) + (A2+offset) * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t)
        self.assertAlmostEqual(offset**2, aopy.analysis.calc_tracking_error(cursor_traj, target_traj))

        fig, ax = plt.subplots(4,1, figsize=(10,10))
        ax[0].set_title(f'MSE = {offset**2}cm $^2$')
        ax[0].plot(t, target_traj, 'tab:orange', label='target')
        ax[0].plot(t, cursor_traj, 'darkviolet', label='cursor')
        ax[0].set_ylabel('position (cm)'); ax[0].set_ylim([-10,10])
        ax[0].set_xticklabels([])
        ax[0].legend()

        offset = 3
        target_traj = A1 * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t) + A2 * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t)
        cursor_traj = (A1+offset) * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t) + (A2+offset) * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t)
        self.assertAlmostEqual(offset**2, aopy.analysis.calc_tracking_error(cursor_traj, target_traj))

        ax[1].set_title(f'MSE = {offset**2}cm $^2$')
        ax[1].plot(t, target_traj, 'tab:orange', label='target')
        ax[1].plot(t, cursor_traj, 'darkviolet', label='cursor')
        ax[1].set_ylabel('position (cm)'); ax[1].set_ylim([-10,10])
        ax[1].set_xticklabels([])

        offset = -2
        target_traj = A1 * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t) + A2 * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t)
        cursor_traj = (A1+offset) * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t) + (A2+offset) * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t)
        self.assertAlmostEqual(offset**2, aopy.analysis.calc_tracking_error(cursor_traj, target_traj))

        ax[2].set_title(f'MSE = {offset**2}cm $^2$')
        ax[2].plot(t, target_traj, 'tab:orange', label='target')
        ax[2].plot(t, cursor_traj, 'darkviolet', label='cursor')
        ax[2].set_ylabel('position (cm)'); ax[2].set_ylim([-10,10])
        ax[2].set_xticklabels([])

        offset = -2
        target_traj = A1 * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t) + A2 * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t)
        cursor_traj = target_traj + offset
        self.assertAlmostEqual(offset**2, aopy.analysis.calc_tracking_error(cursor_traj, target_traj))

        ax[3].set_title(f'MSE = {offset**2}cm $^2$')
        ax[3].plot(t, target_traj, 'tab:orange', label='target')
        ax[3].plot(t, cursor_traj, 'darkviolet', label='cursor')
        ax[3].set_ylabel('position (cm)'); ax[3].set_ylim([-10,10])
        ax[3].set_xlabel('samples')
        filename = 'tracking_error.png'
        savefig(docs_dir,filename)

    def test_calc_tracking_in_time(self):
        inter_event_int = 1
        event_codes = [16, 2, 80, 96, 
                       80, 96, 
                       80, 96, 
                       80, 96, 
                       80, 96, 48, 239] # 5 "tracking in" segments
        event_times = np.arange(0, len(event_codes), step=inter_event_int) # events are all 1 sec apart
        self.assertEqual(5*inter_event_int, aopy.analysis.calc_tracking_in_time(event_codes, event_times))
        self.assertEqual(5*inter_event_int/event_times[-1], aopy.analysis.calc_tracking_in_time(event_codes, event_times, proportion=True))

        inter_event_int = 1
        event_codes = [16, 2, 80, 96, 
                       80, 96, 
                       80, 96, 
                       80, 96, 
                       80, 96, 
                       80, 48, 239] # 6 "tracking in" segments
        event_times = np.arange(0, len(event_codes), step=inter_event_int) # events are all 1 sec apart
        self.assertEqual(6*inter_event_int, aopy.analysis.calc_tracking_in_time(event_codes, event_times))
        self.assertEqual(6*inter_event_int/event_times[-1], aopy.analysis.calc_tracking_in_time(event_codes, event_times, proportion=True))

        inter_event_int = 1
        event_codes = [16, 2, 80, 96, 
                       80, 96, 
                       80, 96, 
                       80, 96, 
                       80, 96, 79, 239] # 5 "tracking in" segments
        event_times = np.arange(0, len(event_codes), step=inter_event_int) # events are all 1 sec apart
        self.assertEqual(5*inter_event_int, aopy.analysis.calc_tracking_in_time(event_codes, event_times))
        self.assertEqual(5*inter_event_int/event_times[-1], aopy.analysis.calc_tracking_in_time(event_codes, event_times, proportion=True))

    def test_vector_angle(self):
        # test with vectors in Q1, Q2, Q3, Q4
        for i in range(9):
            theta = i/4*np.pi
            v = [np.cos(theta), np.sin(theta)]
            self.assertAlmostEqual(aopy.analysis.vector_angle(v), theta)
            self.assertAlmostEqual(aopy.analysis.vector_angle(v, in_degrees=True), theta*180/np.pi)

class ControlTheoreticAnalysisTests(unittest.TestCase):
    def test_get_machine_dynamics(self):
        freqs = np.linspace(0,1,20)
        exp_freqs = freqs[::2] # "even" freqs
        print(freqs)
        print(exp_freqs)
        
        M = controllers.get_machine_dynamics(freqs, 0)
        np.testing.assert_equal(len(M), 20) # check length of M matches length of freqs 
        np.testing.assert_array_equal(M, np.ones(20,)) # check M is all 1s (0th order system)

        M = controllers.get_machine_dynamics(freqs, 1)
        np.testing.assert_equal(len(M), 20) # check length of M matches length of freqs

        M = controllers.get_machine_dynamics(freqs, 2)
        np.testing.assert_equal(len(M), 20) # check length of M matches length of freqs

        M_exp_freqs = controllers.get_machine_dynamics(freqs, 2, exp_freqs) # same as above, but only return M at exp_freqs
        np.testing.assert_equal(len(M_exp_freqs), 10) # check length of M_exp_freqs matches length of exp_freqs
        np.testing.assert_array_equal(M[::2], M_exp_freqs) # check M_exp_freqs matches M indexed at even freqs

        # only need to run once, because it fails (as expected - function doesn't recognize 3rd order system)
        # M = controllers.get_machine_dynamics(freqs, 3)

    def test_calc_transfer_function(self):
        samplerate = 100
        t = np.arange(samplerate) # 1sec signal
        exp_freqs = [2, 5] # [f1, f2] Hz

        # input signal
        A1_in = 4 # amplitude
        p1_in = 0 # phase
        A2_in = 2
        p2_in = 0
        input_signal = A1_in * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t + p1_in) + A2_in * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t + p2_in)
        plt.plot(t, input_signal)

        # output signal
        A1_out = 3
        p1_out = np.pi/6
        A2_out = 3
        p2_out = -np.pi/4
        output_signal = A1_out * np.sin((2*np.pi)*(exp_freqs[0]/samplerate)*t + p1_out) + A2_out * np.sin((2*np.pi)*(exp_freqs[1]/samplerate)*t + p2_out)
        plt.plot(t, output_signal)

        # input--> input
        freqs, transfer_func = controllers.calc_transfer_function(input_signal, input_signal, samplerate)
        self.assertEqual(len(transfer_func), len(t)/2)
        np.testing.assert_array_almost_equal(np.squeeze(abs(transfer_func)), np.ones(len(freqs))) # magnitude transformation is 1 at all freqs  
        np.testing.assert_array_almost_equal(np.squeeze(np.angle(transfer_func)), np.zeros(len(freqs))) # phase shift is 0 at all freqs

        # input--> output
        freqs, transfer_func = controllers.calc_transfer_function(input_signal, output_signal, samplerate)
        self.assertEqual(len(transfer_func), len(t)/2)
        np.testing.assert_array_equal(np.squeeze(abs(transfer_func))[np.isin(freqs, exp_freqs)], [A1_out/A1_in, A2_out/A2_in])
        np.testing.assert_array_almost_equal(np.squeeze(np.angle(transfer_func))[np.isin(freqs, exp_freqs)], [p1_out, p2_out])

        # input--> output, only at experimental freqs
        freqs, transfer_func = controllers.calc_transfer_function(input_signal, output_signal, samplerate, exp_freqs)
        self.assertEqual(len(transfer_func), len(exp_freqs))
        np.testing.assert_array_equal(np.squeeze(abs(transfer_func)), [A1_out/A1_in, A2_out/A2_in])
        np.testing.assert_array_almost_equal(np.squeeze(np.angle(transfer_func)), [p1_out, p2_out])

    def test_pair_trials_by_frequency(self):
        # trials with perfectly alternating frequency content
        e = [0.1, 0.2, 0.3, 0.4]
        o = [0.05, 0.15, 0.25, 0.35]

        ref_freqs = [e, o]*5
        dis_freqs = [e, o]*5

        trial_pairs = controllers.pair_trials_by_frequency(ref_freqs, dis_freqs, max_trial_distance=1, limit_pairs_per_trial=True, max_pairs_per_trial=2)
        np.testing.assert_array_equal(trial_pairs, np.array([[i,i+1] for i in range(9)]))

        # trials with some repeating frequency content
        e = [0.1, 0.2, 0.3, 0.4]
        o = [0.05, 0.15, 0.25, 0.35]

        ref_freqs = [e, o, o, o, e, o, e]
        dis_freqs = [o, e, e, e, o, e, o]
        expected_pairs = [[0,1],
                          [3,4],
                          [4,5],
                          [5,6]
                          ]

        trial_pairs = controllers.pair_trials_by_frequency(ref_freqs, dis_freqs, max_trial_distance=1, limit_pairs_per_trial=True, max_pairs_per_trial=2)
        np.testing.assert_array_equal(trial_pairs, expected_pairs)

        # trials with some repeating frequency content & non-default function parameters
        e = [0.1, 0.2, 0.3, 0.4]
        o = [0.05, 0.15, 0.25, 0.35]

        ref_freqs = [e, o, o, o, e, o, e]
        dis_freqs = [o, e, e, e, o, e, o]
        expected_pairs = [[0,1],
                          [0,2],
                          [2,4],
                          [3,4],
                        #   [4,5], would not get this pair because trial 4 already used in 2 pairs 
                          [5,6]
                          ]

        trial_pairs = controllers.pair_trials_by_frequency(ref_freqs, dis_freqs, max_trial_distance=2, limit_pairs_per_trial=True, max_pairs_per_trial=2)
        np.testing.assert_array_equal(trial_pairs, expected_pairs)

        # trials with some repeating frequency content & non-default function parameters
        e = [0.1, 0.2, 0.3, 0.4]
        o = [0.05, 0.15, 0.25, 0.35]

        ref_freqs = [e, o, o, o, e, o, e]
        dis_freqs = [o, e, e, e, o, e, o]
        expected_pairs = [[0,1],
                          [0,2],
                          [2,4],
                          [3,4],
                          [4,5],
                          [5,6]
                          ]

        trial_pairs = controllers.pair_trials_by_frequency(ref_freqs, dis_freqs, max_trial_distance=2, limit_pairs_per_trial=False)
        np.testing.assert_array_equal(trial_pairs, expected_pairs)

class ConfidenceIntervalTests(unittest.TestCase):
    def test_get_confidence_interval(self):
        np.random.seed(1)
        uniform_random = np.random.uniform(size=10000)
        hist_bins = np.linspace(0,1,10000)
        interval = aopy.analysis.get_confidence_interval(uniform_random, hist_bins)
        self.assertEqual(round(interval[0],3), 0.026)
        self.assertEqual(round(interval[1],3), 0.975)
    
    def test_calc_confidence_interval_overlap(self):
        CI1 = [10,20]
        CI2 = [17,30]
        overlap = aopy.analysis.calc_confidence_interval_overlap(CI1,CI2)
        self.assertEqual(overlap,(20-17)/(20-10))
        

class TestCalcMaxCohAndLags(unittest.TestCase):
    def setUp(self):
        # Define the parameters for the test
        self.lags = [-0.1, 0, 0.1]  # Lags in seconds
        self.frequency_range = [0, 4]  # Frequency range (0 to 4 Hz)
        
        # Simulate some LFP data with random values
        self.num_channels = 3
        self.num_timepoints = 5000
        self.lfp_array1 = np.random.randn(self.num_channels, self.num_timepoints)
        self.lfp_array2 = np.random.randn(self.num_channels, self.num_timepoints)

        # Mock taper parameters and the coherence calculation function
        global precondition
        precondition = MagicMock()
        precondition.convert_taper_parameters = MagicMock(return_value=(256, 2, 3))  # Mock return values for tapers

        global calc_mt_tfcoh
        calc_mt_tfcoh = MagicMock(return_value=(
            np.linspace(0, 200, 100),  # Mock frequency array (0-200 Hz)
            np.linspace(-0.5, 0.5, 50),  # Mock time array (centered around 0)
            np.random.rand(100, 50)  # Mock coherence array (freq x time)
        ))

    def test_calc_max_coh_and_lags(self):
        # Run the function
        max_coherence_matrix, lag_of_max_coherence_matrix = aopy.analysis.calc_max_coh_and_lags(
            self.lags, self.lfp_array1, self.lfp_array2, self.frequency_range
        )

        # Check the shapes of the output matrices
        self.assertEqual(max_coherence_matrix.shape, (self.num_channels, self.num_channels))
        self.assertEqual(lag_of_max_coherence_matrix.shape, (self.num_channels, self.num_channels))

        # Verify the values fall within expected ranges
        self.assertTrue(np.all(max_coherence_matrix >= 0) and np.all(max_coherence_matrix <= 1), 
                        "Coherence values should be between 0 and 1")

        # Check if lags are converted to milliseconds in lag_of_max_coherence_matrix
        self.assertTrue(np.all(np.isin(lag_of_max_coherence_matrix, np.array(self.lags) * 1000)),
                        "Lags should be in milliseconds and match the provided lags")
        
if __name__ == "__main__":

    unittest.main()


