# we are generating noisy test data using sine and cosine functions with multiple frequencies
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

def plot_filtered_signal(t, x, x_filter, low, high):
    # Plotting noisy test signal and filtered signal
    plt.plot(t, x, label='Noisy signal')
    plt.plot(t, x_filter, label='Filtered signal')
    plt.xlabel('time (seconds)')
    # plt.hlines([-self.a, self.a], 0, self.T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='best')
    plt.show()


def plot_phase_locking(t, a, f0, x_filter):
    # Plotting filtered signal with original signal frequency
    x_f0 = a * 2 * np.cos(2 * np.pi * f0 * t)
    plt.plot(t, x_f0, label='Original signal (%g Hz)' % f0)
    plt.plot(t, x_filter, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.ylabel('amplitude')
    plt.title('Comparison of Original Vs Filtered Signal')
    plt.legend(loc='best')
    plt.show()


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
    plt.show()


def plot_psd(x, x_filter, fs):
    # Plot power spectral density of the signal
    f, psd = precondition.get_psd_welch(x, fs)
    f_filtered, psd_filtered = precondition.get_psd_welch(x_filter, fs)
    plt.semilogy(f, psd, label='test signal')
    plt.semilogy(f_filtered, psd_filtered, label='filtered output')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.title('Power Spectral Density Comparison')
    plt.show()


def plot_db_spectral_estimate(freq, psd, psd_filter, labels):
    psd = 10 * np.log10(psd)
    psd_filter = 10 * np.log10(psd_filter)
    plt.figure()
    precondition.plot_spectral_estimate(freq, psd, (psd_filter,), elabels=(labels,))
    plt.show()

class FilterTests(unittest.TestCase):


    def setUp(self):
        self.T = 0.05
        self.fs = 25000
        self.freq = [600, 312, 2000]
        self.a = 0.02
        # testing generate test_signal
        self.f0 = self.freq[0]
        _x, _t = utils.generate_test_signal(self.T, self.fs, self.freq, [self.a * 2, self.a*0.5, self.a*1.5, self.a*20 ])

        self.x = _x
        self.t = _t

    def test_butterworth(self):
        # Sample rate and desired cutoff frequencies (in Hz).
        # fs = 25000.0
        lowcut = 500.0
        highcut = 1200.0
        tic = time.perf_counter()
        x_filter, f_band = precondition.butterworth_filter_data(self.x, fs = self.fs, bands= [(lowcut, highcut)])
        toc = time.perf_counter()
        print(f" Butterworth filter executed in {toc - tic:0.4f} seconds")

        fname = 'test_signal_filtered_Signal.png'
        plot_filtered_signal(self.t, self.x, x_filter[0], lowcut, highcut)
        plt.show()
        savefig(write_dir, fname)

        fname = 'test_phase_locking.png'
        plot_phase_locking(self.t, self.a, self.f0, x_filter[0])
        plt.show()
        savefig(write_dir, fname)

        fname = 'freq_response_vs_filter_order.png'
        plot_freq_response_vs_filter_order(lowcut, highcut, self.fs)
        plt.show()
        savefig(write_dir, fname)

        fname = 'plot_psd.png'
        plot_psd(self.x, x_filter[0], self.fs)
        plt.show()
        savefig(write_dir, fname)

    def test_multitaper(self):
        f, psd_filter, mu = precondition.get_psd_multitaper(self.x, self.fs)
        psd = precondition.get_psd_welch(self.x, self.fs, np.shape(f)[0])[1]

        fname = 'multitaper_powerspectrum.png'
        plot_db_spectral_estimate(f, psd, psd_filter, 'multitaper')
        plt.show()
        savefig(write_dir, fname)

        bands = [(0, 10), (100, 200), (560, 660), (2000, 2010)]
        lfp = precondition.multitaper_lfp_bandpower(f, psd_filter, bands, 1, False)
        plt.plot(np.arange(len(bands)), np.squeeze(lfp), '-bo')
        plt.xticks(np.arange(len(bands)), bands)
        plt.xlabel('Frequency band (Hz)')
        plt.ylabel('Band Power')
        plt.show()
        fname = 'lfp_bandpower.png'
        savefig(write_dir, fname)

class SpikeDetectionTests(unittest.TestCase):
        
    def test_calc_spike_threshold(self):
        data = np.array(((0,0,1),(4,0,-1),(0,9,-1), (4,9,1)))
        threshold_values = precondition.calc_spike_threshold(data)
        expected_thresh_values = np.array((6,13.5,3))
        np.testing.assert_allclose(threshold_values, expected_thresh_values)

    def test_detect_spikes(self):
        # Test spike time detection
        data = np.array(((0,0,1),(4,0,-1),(0,9,-1), (4,9,1)))
        threshold_values = np.array((0.5, 0.5, 0.5))
        spike_times, wfs = precondition.detect_spikes(data, 10, wf_length=None, threshold=threshold_values)
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

        _, wfs = precondition.detect_spikes(large_data, 300, wf_length=10000, threshold=threshold)
        np.testing.assert_allclose(wfs[0], np.array(((3,4,5),(3,4,5))))
        np.testing.assert_allclose(wfs[1], np.array((8,9,10)).reshape(1,-1))
        np.testing.assert_allclose(wfs[2], np.array((13,14,15)).reshape(1,-1))
        np.testing.assert_allclose(wfs[3], np.array((np.nan,np.nan,np.nan)).reshape(1,-1))

        # Test automatic thresholding
        spike_times, wfs = precondition.detect_spikes(large_data, 100, wf_length=10000, threshold=None)
        np.testing.assert_allclose(spike_times[0], np.array((0.09, 0.19)))
        np.testing.assert_allclose(spike_times[1], np.array((0.18)))
        np.testing.assert_allclose(wfs[0], np.array(((9),(np.nan))).reshape(-1,1))
        np.testing.assert_allclose(wfs[1], np.array((18)))

        # Test speed
        test_speed_data = np.random.normal(size=(250000, 256))
        start = time.time()
        _, _ = spike_times, wfs = precondition.detect_spikes(test_speed_data, 25000, wf_length=1000, threshold=None)
        stop = time.time()

        print('Spike detection on 250,000 samples by 256ch takes ' + str(round(stop-start, 3)) + ' sec')

if __name__ == "__main__":
    unittest.main()

