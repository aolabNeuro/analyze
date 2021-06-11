# we are generating noisy test data using sine and cosine functions with multiple frequencies
import unittest
from aopy.visualization import *
import matplotlib.pyplot as plt
from aopy import precondition
import time

test_dir = os.path.dirname(__file__)
write_dir = os.path.join(test_dir, 'tmp')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

class FilterTests(unittest.TestCase):

    def __init_subclass__(self, **kwargs):
        self.T = 0.05
        self.fs = 25000
        self.freq = [600, 312, 2000]
        self.a = 0.02
        # testing generate test_signal
        _x, _t, _f0 = precondition.generate_test_signal(self.T, self.fs, self.freq, self.a)

    def setUp(self):
        self.T = 0.05
        self.fs = 25000
        self.freq = [600, 312, 2000]
        self.a = 0.02
        # testing generate test_signal
        _x, _t, _f0 = precondition.generate_test_signal(self.T, self.fs, self.freq, self.a)

        self.x = _x
        self.t = _t
        self.f0 = _f0

    def test_butterworth(self):
        # Sample rate and desired cutoff frequencies (in Hz).
        # fs = 25000.0
        lowcut = 500.0
        highcut = 1200.0
        tic = time.perf_counter()
        x_filter = precondition.bandpass_butterworth_filter_data(self.x, lowcut, highcut, self.fs)
        toc = time.perf_counter()
        print(f" Butterworth filter executed in {toc - tic:0.4f} seconds")

        fname = 'test_signal_filtered_Signal.png'
        plot_filtered_signal(self.t, self.x, x_filter, lowcut, highcut)
        plt.show()
        savefig(write_dir, fname)

        fname = 'test_phase_locking.png'
        plot_phase_locking(self.t, self.a, self.f0, x_filter)
        plt.show()
        savefig(write_dir, fname)

        fname = 'freq_response_vs_filter_order.png'
        plot_freq_response_vs_filter_order(self.x, lowcut, highcut, self.fs)
        plt.show()
        savefig(write_dir, fname)

        fname = 'plot_psd.png'
        plot_psd(self.x, x_filter, self.fs)
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

if __name__ == "__main__":
    unittest.main()