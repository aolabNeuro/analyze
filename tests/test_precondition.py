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

    def test_butterworth(self):
        # testing generate test_signal

        self.x, self.t, self.f0 = precondition.generate_test_signal(self.T, self.fs, self.freq, self.a)

        # Sample rate and desired cutoff frequencies (in Hz).
        # fs = 25000.0
        self.lowcut = 500.0
        self.highcut = 1200.0
        tic = time.perf_counter()
        self.x_filter = precondition.bandpass_butterworth_filter_data(self.x, self.lowcut, self.highcut, self.fs)
        toc = time.perf_counter()
        print(f" Butterworth filter executed in {toc - tic:0.4f} seconds")

    def test_plot_filtered_signal(self):
        # Plotting noisy test signal and filtered signal
        fname = 'test_signal_filtered_Signal.png'
        precondition.plot_filtered_signal(self.t, self.x, self.x_filter, self.lowcut, self.highcut)
        plt.show()
        savefig(write_dir, fname)

    def test_plot_phase_locking(self):
        # Plotting filtered signal with original signal frequency
        fname = 'test_phase_locking.png'
        precondition.plot_phase_locking(self.t, self.a, self.f0, self.x_filter)
        plt.show()
        savefig(write_dir, fname)

    def test_plot_freq_response_vs_filter_order(self):
        fname = 'freq_response_vs_filter_order.png'
        precondition.plot_freq_response_vs_filter_order(self.x, self.lowcut, self.highcut, self.fs)
        plt.show()
        savefig(write_dir, fname)

    def test_plot_psd(self):
        fname = 'plot_psd.png'
        precondition.plot_psd(self.x, self.x_filter, self.fs)
        plt.show()
        savefig(write_dir, fname)

    def test_multitaper(self):

        f, psd_filter = precondition.bandpass_multitaper_filter_data(self.x,self.fs)
        psd = precondition.get_psd(self.x,self.fs)
        fname = 'multitaper_powerspectrum.png'
        precondition.plot_db_spectral_estimate(f,psd,psd_filter,'multitaper')
        plt.show()
        savefig(write_dir, fname)