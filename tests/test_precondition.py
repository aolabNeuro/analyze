# we are generating noisy test data using sine and cosine functions with multiple frequencies
import unittest
from aopy.data.base import load_preproc_eye_data
from aopy.precondition.eye import *
from aopy import visualization
from aopy.visualization import *
import matplotlib.pyplot as plt
from aopy import precondition
from aopy import utils
import time
from scipy.signal import freqz


test_dir = os.path.dirname(__file__)
write_dir = os.path.join(test_dir, 'tmp')
data_dir = os.path.join(test_dir, 'data')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)
docs_dir = os.path.join(os.path.dirname(test_dir),'docs', 'source', '_images')

'''
Plots to test filter performance
'''

class HelperFunctions:

    @staticmethod
    def test_filter(filt_fun, fs=25000, T=0.05, freq=[600, 312, 2000], a=[5, 2, 0.5], noise=0.2):
        '''
        Helper function to test filters

        Args:
            filt_fun (function): function which inputs a signal and outputs a filtered signal (no other arguments allowed)
            fs (int): sampling rate to use. default 25000
            T (float): period
            freq (list): list of frequencies to generate
            a (list): list of amplitudes
            noise (float): noise amplitude
        '''
        # Generate test_signal
        np.random.seed(0)
        x_single, t = utils.generate_test_signal(T, fs, [freq[0]], [a[0]])
        x_noise, t = utils.generate_test_signal(T, fs, freq, a, noise) # with noise

        # Filter and plot
        x_filt = filt_fun(x_noise)
        fig, ax = plt.subplot_mosaic([['A', 'B'],
                                    ['C', 'C']])
        
        ax['A'].plot(t, x_noise, label='Noisy signal')
        ax['A'].plot(t, x_filt, label='Filtered signal')
        ax['A'].set_xlabel('time (seconds)')
        
        x_filt_simple = filt_fun(x_single)
        ax['B'].plot(t, x_single, label=f'{freq[0]} Hz signal')
        ax['B'].plot(t, x_filt_simple, label='Filtered signal')
        ax['B'].set_xlabel('time (seconds)')

        f_noise, psd_noise = analysis.get_psd_welch(x_noise, fs)
        f_filt, psd_filt = analysis.get_psd_welch(x_filt, fs)
        ax['C'].semilogy(f_noise, psd_noise, label='Noisy signal')
        ax['C'].semilogy(f_filt, psd_filt, label='Filtered signal')
        ax['C'].set_xlabel('frequency (Hz)')
        ax['C'].set_ylabel('PSD')

        for ax in ax.values():
            ax.grid(True)
            ax.axis('tight')
            ax.legend(loc='best')

    @staticmethod
    def plot_freq_response_vs_filter_order(lowcut, highcut, fs):
        # Plot the frequency response for a few different orders
        for order in [2, 3, 4, 5, 6]:  # trying  different order of butterworth to see the roll off around cut-off frequencies
            b, a = precondition.butterworth_params(lowcut, highcut, fs, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

        plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.title('Comparison of Frequency response for Diff. Orders of Butterworth Filter')

class FilterTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.T = 0.05
        self.fs = 25000
        self.freq = [600, 312, 2000]
        self.a = 0.02
        # testing generate test_signal
        self.f0 = self.freq[0]
        self.x, self.t = utils.generate_test_signal(self.T, self.fs, self.freq, [self.a * 2, self.a*0.5, self.a*1.5, self.a*20 ])

        self.n_ch = 8
        self.x2 = utils.generate_multichannel_test_signal(self.T, self.fs, self.n_ch, self.freq[0], self.a*0.5) + \
            utils.generate_multichannel_test_signal(self.T, self.fs, self.n_ch, self.freq[1], self.a*1.5)
        self.t2 = np.arange(self.T*self.fs)/self.fs

    def test_butterworth(self):
        
        # Band pass 1500-2500 Hz
        lowcut = 1500.0
        highcut = 2500.0
        fs = 25000
        fn = lambda x: precondition.butterworth_filter_data(x, fs, bands=[(lowcut, highcut)])[0][0]
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_butterworth_bp_2000.png'
        savefig(write_dir, fname)
        plt.figure()
        HelperFunctions.plot_freq_response_vs_filter_order(lowcut, highcut, self.fs)
        fname = 'test_butterworth_order.png'
        savefig(write_dir, fname)

        # Band pass 600 hz
        lowcut = 575.0
        highcut = 625.0
        fn = lambda x: precondition.butterworth_filter_data(x, fs, filter_type='lowpass', cutoff_freqs=[500], order=4)[0][0]
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_butterworth_bp_600.png'
        savefig(write_dir, fname)

        # Low pass 500 Hz
        fn = lambda x: precondition.butterworth_filter_data(x, fs, filter_type='lowpass', cutoff_freqs=[500], order=4)[0][0]
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_butterworth_lp_500.png'
        savefig(write_dir, fname)

        # High pass 500 Hz
        fn = lambda x: precondition.butterworth_filter_data(x, fs=fs, cutoff_freqs=[500], filter_type='highpass', order=4)[0][0]
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_butterworth_hp_500.png'
        savefig(write_dir, fname)

        # Test that multichannel filtering works
        t = 20 # seconds
        nch = 64
        x = utils.generate_multichannel_test_signal(t, fs, nch, 300, 2) # 8 channels data
        tic = time.perf_counter()
        x_filter, _ = precondition.butterworth_filter_data(x, fs=fs, bands=[(lowcut, highcut)])
        toc = time.perf_counter()
        print(f"Butterworth filtering {nch} channels of {t} seconds data took {toc-tic:0.2f} seconds")
        self.assertEqual(x_filter[0].shape, (t*fs, nch))

    def test_mtfilter(self):

        # Low pass 500 Hz
        band = [-500, 500]
        N = 0.1 # N*sampling_rate is time window you analyze
        NW = (band[1]-band[0])/2
        f0 = np.mean(band)
        n, p, k = precondition.convert_taper_parameters(N, NW)
        fs = 25000
        fn = lambda x: precondition.mtfilter(x, n, p, k, fs=fs, f0=f0)
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_mtfilt_lp_500.png'
        savefig(write_dir, fname)

         # Low pass 500 Hz with complex output
        band = [-500, 500]
        N = 0.1
        NW = (band[1]-band[0])/2
        f0 = np.mean(band)
        n, p, k = precondition.convert_taper_parameters(N, NW)
        fn = lambda x: precondition.mtfilter(x, n, p, k, fs=fs, f0=f0, complex_output=True)
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_mtfilt_lp_500_complex.png'
        savefig(write_dir, fname)

        # Low pass 500 Hz without centering the output
        band = [-500, 500]
        N = 0.1
        NW = (band[1]-band[0])/2
        f0 = np.mean(band)
        n, p, k = precondition.convert_taper_parameters(N, NW)
        fn = lambda x: precondition.mtfilter(x, n, p, k, fs=fs, f0=f0, center_output=False)
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_mtfilt_lp_500_noncentered.png'
        savefig(write_dir, fname)

        # Low pass 500 Hz but using convolve instead of fftconvolve
        band = [-500, 500]
        N = 0.1
        NW = (band[1]-band[0])/2
        f0 = np.mean(band)
        n, p, k = precondition.convert_taper_parameters(N, NW)
        fn = lambda x: precondition.mtfilter(x, n, p, k, fs=fs, f0=f0, use_fft=False)
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_mtfilt_lp_500_nofft.png'
        savefig(write_dir, fname)

        # Low pass 500 Hz with smaller time window, using wrapper
        lowcut = 500
        taper_len = 0.01
        fn = lambda x: precondition.mt_lowpass_filter(x, lowcut, taper_len, fs)
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_mtfilt_lp_500_small_time.png'
        savefig(write_dir, fname)

        # Narrow band-pass
        band = [575, 625] # signals within band can pass
        taper_len = 0.1
        fn = lambda x: precondition.mt_bandpass_filter(x, band, taper_len, fs)
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_mtfilt_bp_600_100ms.png'
        savefig(write_dir, fname)
        fname = 'mtfilter.png' # this is the figure that goes in the documentation
        savefig(docs_dir, fname)

        # Narrow band-pass
        band = [575, 625] # signals within band can pass
        taper_len = 0.05
        fn = lambda x: precondition.mt_bandpass_filter(x, band, taper_len, fs)
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_mtfilt_bp_600_50ms.png'
        savefig(write_dir, fname)

        # Narrow band-pass without fft
        band = [575, 625] # signals within band can pass
        taper_len = 0.05
        fn = lambda x: precondition.mt_bandpass_filter(x, band, taper_len, fs, use_fft=False)
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_mtfilt_bp_600_50ms_nofft.png'
        savefig(write_dir, fname)

        # High freq band-pass
        band = [1500, 2500] # signals within band can pass
        taper_len = 0.01
        fn = lambda x: precondition.mt_bandpass_filter(x, band, taper_len, fs)
        HelperFunctions.test_filter(fn, fs=fs)
        fname = 'test_mtfilt_bp_2000.png'
        savefig(write_dir, fname)

        # Test that multichannel filtering works
        t = 20 # seconds
        nch = 64
        x = utils.generate_multichannel_test_signal(t, fs, nch, 300, 2) # 8 channels data
        tic = time.perf_counter()
        x_filter = precondition.mtfilter(x, n, p, k, fs=fs, f0=f0)
        toc = time.perf_counter()
        print(f"Multitaper filtering {nch} channels of {t} seconds data took {toc-tic:0.2f} seconds")
        self.assertEqual(x_filter.shape, (t*fs, nch))


    def test_downsample(self):
        data = np.arange(100)
        data_ds = precondition.downsample(data, 100, 10)
        self.assertEqual(data_ds.shape, (10,))
        self.assertTrue(abs(np.mean(data) - np.mean(data_ds)) < 1)

        data = np.vstack((data, np.arange(100))).T
        data_ds = precondition.downsample(data, 100, 10)
        self.assertEqual(data_ds.shape, (10, 2))
        self.assertTrue(abs(np.mean(data) - np.mean(data_ds)) < 1)

        fs_ds = self.fs/5
        x_ds = precondition.downsample(self.x, self.fs, fs_ds)
        fig, ax = plt.subplots(2,2)
        ax[0,0].plot(self.t, self.x)
        ax[0,0].set_ylabel(f"{self.fs} hz")
        t_ds = np.arange(len(x_ds))/fs_ds
        ax[1,0].plot(t_ds, x_ds)
        ax[1,0].set_ylabel(f"{fs_ds} hz")
        ax[1,0].set_xlabel("time (s)")
        visualization.plot_freq_domain_amplitude(1e-6*self.x, self.fs, ax=ax[0,1])
        ax[0,1].set_xlim(0,5000)
        ax[0,1].set_xlabel('')
        visualization.plot_freq_domain_amplitude(1e-6*x_ds, fs_ds, ax=ax[1,1])
        ax[1,1].set_xlim(0,5000)
        plt.tight_layout()
        filename = 'downsample.png'
        savefig(docs_dir, filename)

        # Test non-integer downsample factor
        data_ds = precondition.downsample(data, 100, 13)
        self.assertEqual(data_ds.shape, (13, 2))
        self.assertTrue(abs(np.mean(data) - np.mean(data_ds)) < 1)

        # Test large data with large upsampling rate
        data = np.arange(1e7)
        samplerate = 25000
        ds_samplerate = 100
        tic = time.perf_counter()
        data_ds = precondition.downsample(data, samplerate, ds_samplerate)
        toc = time.perf_counter()
        print(f"Downsampling {len(data)/samplerate:0.2f} seconds of data from {samplerate} to {ds_samplerate} hz took {toc-tic:0.2f} seconds")

        ds_samplerate = 120
        tic = time.perf_counter()
        data_ds = precondition.downsample(data, samplerate, ds_samplerate)
        toc = time.perf_counter()
        print(f"Downsampling {len(data)/samplerate:0.2f} seconds of data from {samplerate} to {ds_samplerate} hz took {toc-tic:0.2f} seconds")


    def test_filter_lfp(self):
        
        test_data = np.random.uniform(size=(100000,2))
        filt = precondition.filter_lfp(test_data, 25000)
        self.assertEqual(filt.shape, (100000/25, 2))
        self.assertAlmostEqual(np.mean(test_data), np.mean(filt), places=3)

    def test_filter_spikes(self):

        test_data = np.random.uniform(size=(100000,2))
        filt = precondition.filter_spikes(test_data, 25000)
        self.assertEqual(filt.shape, test_data.shape)
        self.assertNotAlmostEqual(np.mean(test_data), np.mean(filt), places=3) # After filtering these should be different

    def test_filter_kinematics(self):
        fs = 100
        fn = lambda x: precondition.filter_kinematics(x, fs, low_cut=15, buttord=4)
        HelperFunctions.test_filter(fn, fs=fs, T=5, freq=[1,3,30], a=[5, 2, 0.5], noise=0.2)
        fname = 'filter_kinematics.png'
        savefig(docs_dir, fname)


class SpikeDetectionTests(unittest.TestCase):
        
    def test_calc_spike_threshold(self):
        data = np.array(((0,0,1),(4,0,-1),(0,9,-1), (4,9,1)))
        threshold_values = precondition.calc_spike_threshold(data, high_threshold=True, rms_multiplier=3)
        expected_thresh_values = np.array((8,18,3))
        np.testing.assert_allclose(threshold_values, expected_thresh_values)

        # Test low_threhshold
        threshold_values = precondition.calc_spike_threshold(data, high_threshold=False, rms_multiplier=3)
        expected_thresh_values = np.array((-4,-9,-3))
        np.testing.assert_allclose(threshold_values, expected_thresh_values)

    def test_detect_spikes(self):
        # Test spike time detection
        data = np.array(((0,0,1),(4,0,-1),(0,9,-1), (4,9,1)))
        threshold_values = np.array((0.5, 0.5, 0.5))
        spike_times, wfs = precondition.detect_spikes(data, 10, threshold=threshold_values, above_thresh=True, wf_length=None)
        np.testing.assert_allclose(spike_times[0], np.array((0.1, 0.3)))
        np.testing.assert_allclose(spike_times[1], np.array((0.2)))
        np.testing.assert_allclose(spike_times[2], np.array((0.3)))
        self.assertEqual(len(wfs), 0)

        # Test negative threshold detection
        data = np.array(((0,0,1),(4,0,-1),(0,9,-1), (4,9,1)))
        threshold_values = np.array((0.5, 0.5, 0.5))
        spike_times, wfs = precondition.detect_spikes(-data, 10, threshold=-threshold_values, above_thresh=False, wf_length=None)
        np.testing.assert_allclose(spike_times[0], np.array((0.1, 0.3)))
        np.testing.assert_allclose(spike_times[1], np.array((0.2)))
        np.testing.assert_allclose(spike_times[2], np.array((0.3)))
        self.assertEqual(len(wfs), 0)

        # Test uneven threshold detection.
        data = np.array(((0,0,1),(4,0,-1),(0,9,-1), (4,9,1)))
        threshold_values = np.array((2, 5, 0.5))
        spike_times, wfs = precondition.detect_spikes(-data, 10, threshold=-threshold_values, above_thresh=False, wf_length=None)
        np.testing.assert_allclose(spike_times[0], np.array((0.1, 0.3)))
        np.testing.assert_allclose(spike_times[1], np.array((0.2)))
        np.testing.assert_allclose(spike_times[2], np.array((0.3)))
        self.assertEqual(len(wfs), 0)


        # Test waveforms
        large_data = np.zeros((20,4))
        threshold = np.array([2.5,7.5,12.5,17.5])
        for ii in range(large_data.shape[1]):
            large_data[:,ii] = np.arange(large_data.shape[0])
            
        large_data[10:,0] = np.arange(0,large_data.shape[0]-10) 

        _, wfs = precondition.detect_spikes(large_data, 300, threshold=threshold, tbefore_wf=3333, wf_length=10000)
        np.testing.assert_allclose(wfs[0], np.array(((2,3,4),(2,3,4))))
        np.testing.assert_allclose(wfs[1], np.array((7,8,9)).reshape(1,-1))
        np.testing.assert_allclose(wfs[2], np.array((12,13,14)).reshape(1,-1))
        np.testing.assert_allclose(wfs[3], np.array((np.nan,np.nan,np.nan)).reshape(1,-1))

        # Test speed
        test_speed_data = np.random.normal(size=(250000, 256))
        start = time.time()
        threshold = precondition.calc_spike_threshold(test_speed_data)
        spike_times, wfs = precondition.detect_spikes(test_speed_data, 25000, threshold=threshold, wf_length=10000)
        stop = time.time()

        print('Spike detection on 250,000 samples by 256ch takes ' + str(round(stop-start, 3)) + ' sec')

    def test_spike_times_chunk(self):
        data = np.array(((0,0,1),(4,3,1),(0,3,-1),(4,0,1),(0,1,0),(4,1,4),(0,2,1),(3,5,1)))
        threshold_values = np.array((0.5, 0.5, 0.5))
        spike_times, wfs1 = precondition.detect_spikes(data, 10, threshold_values, above_thresh=True, wf_length=None)
        spike_times_chunk, wfs2 = precondition.detect_spikes_chunk(data, 10, threshold_values, chunksize=2, above_thresh=True, wf_length=None)
        for ich in range(3):
            np.testing.assert_allclose(spike_times[ich],spike_times_chunk[ich])
        
        # different chunk size
        spike_times, wfs1 = precondition.detect_spikes(data, 10, threshold_values, above_thresh=True, wf_length=None)
        spike_times_chunk, wfs2 = precondition.detect_spikes_chunk(data, 10, threshold_values, chunksize=3, above_thresh=True, wf_length=None)
        for ich in range(3):
            np.testing.assert_allclose(spike_times[ich],spike_times_chunk[ich])

        # above_threshold = false
        spike_times, wfs1 = precondition.detect_spikes(data, 10, threshold_values, above_thresh=False, wf_length=None)
        spike_times_chunk, wfs2 = precondition.detect_spikes_chunk(data, 10, threshold_values, chunksize=2, above_thresh=False, wf_length=None)
        for ich in range(3):
            np.testing.assert_allclose(spike_times[ich],spike_times_chunk[ich]) 
            
    def test_filter_spike_times_fast(self):
        data = np.array(((0,0),(1,1),(0,0),(1,0),(0,0),(1,1)))
        threshold = np.array((0.5,0.5))
        spike_times, _ = precondition.detect_spikes(data,1,threshold=threshold,tbefore_wf=1e6,wf_length=2e6)
        filtered_spike_times1, _ = precondition.filter_spike_times_fast(spike_times[0], refractory_period=2.5e6)
        filtered_spike_times2, _ = precondition.filter_spike_times_fast(spike_times[1], refractory_period=2.5e6)
        np.testing.assert_allclose(filtered_spike_times1, np.array((1)))
        np.testing.assert_allclose(filtered_spike_times2, np.array((1,5)))

    def test_filter_spike_times(self):
        data = np.array(((0,0),(1,1),(0,0),(1,0),(0,0),(1,1)))
        threshold = np.array((0.5,0.5))
        spike_times, _ = precondition.detect_spikes(data,1,threshold=threshold,tbefore_wf=1e6,wf_length=2e6)
        filtered_spike_times1, _ = precondition.filter_spike_times(spike_times[0], refractory_period=2.5e6)
        filtered_spike_times2, _ = precondition.filter_spike_times(spike_times[1], refractory_period=2.5e6)
        np.testing.assert_allclose(filtered_spike_times1, np.array((1,5)))
        np.testing.assert_allclose(filtered_spike_times2, np.array((1,5)))

    def test_filter_spike_times_speed(self):
        test_speed_data = np.random.normal(size=(250000, 256))
        threshold = precondition.calc_spike_threshold(test_speed_data)
        spike_times, wfs = precondition.detect_spikes(test_speed_data, 25000, threshold=threshold, wf_length=10000)
        
        start = time.time()
        for ich in range(len(spike_times)):
            filtered_spike_times1, _ = precondition.filter_spike_times_fast(spike_times[ich], refractory_period=100)
        stop = time.time()
        print('Fast spike filtering on 250,000 samples by 256ch takes ' + str(round(stop-start, 3)) + ' sec')

        start = time.time()
        for ich in range(len(spike_times)):
            filtered_spike_times1, _ = precondition.filter_spike_times(spike_times[ich], refractory_period=100)
        stop = time.time()
        print('Regular spike filtering on 250,000 samples by 256ch takes ' + str(round(stop-start, 3)) + ' sec')

    def test_binspikes(self):
        data = np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]])
        data_T = data.T
        fs = 10
        binned_spikes = precondition.bin_spikes(data_T, fs, 0.5)

        self.assertEqual(binned_spikes.shape[0], 4)
        self.assertEqual(binned_spikes.shape[1], 2)
        self.assertEqual(binned_spikes[0,0], 4) # Sum first 5 points and * 2

    def test_bin_spike_times(self):
        spike_times = np.array([0.0208, 0.0341, 0.0347, 0.0391, 0.0407])
        spike_times = spike_times.T
        time_before = 0
        time_after = 0.05
        bin_width = 0.01
        binned_unit_spikes, time_bins = precondition.bin_spike_times(spike_times, time_before, time_after, bin_width)
        
        self.assertEqual(binned_unit_spikes[2], 100)
        self.assertEqual(binned_unit_spikes[3], 300)
        self.assertEqual(time_bins[2], 0.025)
        self.assertEqual(time_bins[3], 0.035)

class EyeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        preproc_dir = data_dir
        subject = 'beignet'
        te_id = 5974
        date = '2022-07-01'
        
        eye_data, eye_metadata = load_preproc_eye_data(preproc_dir, subject, te_id, date)
        cls.calibrated_eye_data = eye_data['calibrated_data']
        cls.samplerate = eye_metadata['samplerate']

    def test_filter_eye(self):
        orig = self.calibrated_eye_data
        t_orig = np.arange(len(orig))/self.samplerate
        data_filt = filter_eye(orig, self.samplerate, downsamplerate=100)
        samplerate = 100

        le_orig = orig[:,:2]
        le_filt = data_filt[:,:2]
        t = np.arange(len(le_filt))/self.samplerate
        self.assertNotEqual(self.samplerate, samplerate)
        fig, ax = plt.subplots(4,1)
        ax[0].plot(t_orig, le_orig)
        plot_freq_domain_amplitude(1e-6*le_orig, self.samplerate, ax=ax[1])
        ax[1].set_ylim(0,1)
        ax[1].set_xlim(1,1000)
        ax[2].plot(t, le_filt)
        plot_freq_domain_amplitude(1e-6*le_filt, samplerate, ax=ax[3])
        ax[3].set_ylim(0,1)
        ax[3].set_xlim(1,1000)
        plt.tight_layout()
        savefig(docs_dir, 'filter_eye.png')

        self.assertEqual(data_filt.shape[1], orig.shape[1])
        self.assertEqual(le_filt.shape[1], le_orig.shape[1])


    def test_convert_pos_to_accel(self):
        pos = self.calibrated_eye_data[:,:2]
        accel = convert_pos_to_accel(pos, self.samplerate)
        fig, ax = plt.subplots(4,1)
        plot_timeseries(1e-6*pos, self.samplerate, ax=ax[0])        
        plot_freq_domain_amplitude(1e-6*pos, self.samplerate, ax=ax[1])
        ax[1].set_ylabel('pos')

        plot_timeseries(1e-6*accel, self.samplerate, ax=ax[2])        
        plot_freq_domain_amplitude(1e-6*accel, self.samplerate, ax=ax[3])
        ax[1].set_ylabel('accel')
        plt.tight_layout()
        savefig(docs_dir, 'convert_pos_to_accel_nofilter.png')

        # Important to low-pass filter before computing acceleration
        pos_filt = filter_eye(pos, self.samplerate, downsamplerate=100)
        samplerate = 100
        accel = convert_pos_to_accel(pos, samplerate)
        fig, ax = plt.subplots(4,1)
        plot_timeseries(1e-6*pos, self.samplerate, ax=ax[0])        
        ax[0].set_ylabel('pos')
        plot_freq_domain_amplitude(1e-6*pos, self.samplerate, ax=ax[1])
        ax[1].set_ylabel('pos')
        ax[1].set_ylim(0,1)
        ax[1].set_xlim(1,1000)

        plot_timeseries(1e-6*accel, samplerate, ax=ax[2])        
        ax[2].set_ylabel('pos')
        plot_freq_domain_amplitude(1e-6*accel, samplerate, ax=ax[3])
        ax[3].set_ylabel('accel')
        ax[3].set_xlim(1,1000)
        plt.tight_layout()
        savefig(docs_dir, 'convert_pos_to_accel_filter.png')

    def test_detect_saccades(self):
        le_data_filt = filter_eye(self.calibrated_eye_data[:,:2], self.samplerate, downsamplerate=100)
        samplerate = 100

        onset, duration, distance = detect_saccades(le_data_filt, samplerate, debug=True, debug_window=(0, 5))
        savefig(docs_dir, 'detect_saccades.png')
        plt.close()

        self.assertTrue(np.all(duration < 0.16)) # max saccade duration
        self.assertTrue(np.all(duration > 0.015)) # min saccade duration
        offset = np.array([o + d for o, d in zip(onset, duration)])
        self.assertTrue(np.any(onset[1:]-offset[:-1] <= 0.02)) # intersaccade min by default is not set

        plt.figure()
        plt.hist(1000*duration)
        plt.xlabel('Duration (ms)')
        plt.figure()
        plt.hist(distance)
        plt.xlabel('Distance (cm)')
        savefig(docs_dir, 'detect_saccades_hist.png')
        plt.close()

        plt.figure()
        plt.scatter(1000*duration, distance)
        plt.xlabel('Duration (ms)')
        plt.ylabel('Distance (cm)')
        savefig(docs_dir, 'detect_saccades_scatter.png')
        plt.close()

        onset, duration, distance = detect_saccades(le_data_filt, samplerate, intersaccade_min=0.02)
        offset = np.array([o + d for o, d in zip(onset, duration)])
        self.assertTrue(np.all(onset[1:]-offset[:-1] > 0.02)) # intersaccade min



if __name__ == "__main__":
    unittest.main()

