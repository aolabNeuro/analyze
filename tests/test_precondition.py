# we are generating noisy test data using sine and cosine functions with multiple frequencies
from platform import python_branch
import unittest
from aopy.visualization import *
import matplotlib.pyplot as plt
from aopy import precondition
from aopy import utils
import time
from scipy.signal import freqz


test_dir = os.path.dirname(__file__)
write_dir = os.path.join(test_dir, 'tmp')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

'''
Plots to test filter performance
'''

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
        
        # Band pass 500-1200 Hz
        lowcut = 500.0
        highcut = 1200.0
        fs = 25000
        fn = lambda x: precondition.butterworth_filter_data(x, fs, bands=[(lowcut, highcut)])[0][0]
        test_filter(fn, fs=fs)
        fname = 'test_butterworth_bp_500_1200.png'
        savefig(write_dir, fname)
        plt.figure()
        plot_freq_response_vs_filter_order(lowcut, highcut, self.fs)
        fname = 'test_butterworth_order.png'
        savefig(write_dir, fname)

        # Low pass 500 Hz
        fn = lambda x: precondition.butterworth_filter_data(x, fs, filter_type='lowpass', cutoff_freqs=[500], order=4)[0][0]
        test_filter(fn, fs=fs)
        fname = 'test_butterworth_lp_500.png'
        savefig(write_dir, fname)

        # High pass 500 Hz
        fn = lambda x: precondition.butterworth_filter_data(x, fs=fs, cutoff_freqs=[500], filter_type='highpass', order=4)[0][0]
        test_filter(fn, fs=fs)
        fname = 'test_butterworth_hp_500.png'
        savefig(write_dir, fname)

        # Test that multichannel filtering works
        t = 2 # seconds
        nch = 8
        x = utils.generate_multichannel_test_signal(t, fs, nch, 300, 2) # 8 channels data
        x_filter, _ = precondition.butterworth_filter_data(x, fs=fs, bands=[(lowcut, highcut)])
        self.assertEqual(x_filter[0].shape, (t*fs, nch))

    def test_mtfilter(self):

        # Low pass 500 Hz
        band = [-500, 500]
        N = 0.1 # N*sampling_rate is time window you analyze
        NW = (band[1]-band[0])/2
        f0 = np.mean(band)
        tapers = [N, NW]
        fs = 25000
        fn = lambda x: precondition.mtfilter(x, tapers, fs=fs, f0=f0)
        test_filter(fn, fs=fs)
        fname = 'test_mtfilt_lp_500.png'
        savefig(write_dir, fname)

        # Low pass 500 Hz with larger time window
        band = [-500, 500]
        N = 0.05
        NW = (band[1]-band[0])/2
        f0 = np.mean(band)
        tapers = [N, NW]
        fn = lambda x: precondition.mtfilter(x, tapers, fs=fs, f0=f0)
        test_filter(fn, fs=fs)
        fname = 'test_mtfilt_lp_500.png'
        savefig(write_dir, fname)

        # Narrow band-pass
        band = [575, 625] # signals within band can pass
        N = 0.1
        NW = (band[1]-band[0])/2
        f0 = np.mean(band)
        tapers = [N, NW]
        fn = lambda x: precondition.mtfilter(x, tapers, fs=fs, f0=f0)
        test_filter(fn, fs=fs)
        fname = 'test_mtfilt_bp_500.png'
        savefig(write_dir, fname)

        # High freq band-pass
        band = [1500, 2500] # signals within band can pass
        N = 0.1 # N*sampling_rate is time window you analyze
        NW = (band[1]-band[0])/2
        f0 = np.mean(band)
        tapers = [N, NW]
        fn = lambda x: precondition.mtfilter(x, tapers, fs=fs, f0=f0)
        test_filter(fn, fs=fs)
        fname = 'test_mtfilt_bp_2000.png'
        savefig(write_dir, fname)

        # Test that multichannel filtering works
        t = 2 # seconds
        nch = 8
        x = utils.generate_multichannel_test_signal(t, fs, nch, 300, 2) # 8 channels data
        x_filter = precondition.mtfilter(x, tapers, fs=fs, f0=f0)
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

if __name__ == "__main__":
    unittest.main()

