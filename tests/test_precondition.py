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


T = 0.05
fs = 25000
freq = [600, 312, 2000]
a = 0.02
_x, _t, _f0 = precondition.generate_test_signal(T, fs, freq, a)

lowcut = 500.0
highcut = 1200.0

tic = time.perf_counter()
x_filter = precondition.bandpass_butterworth_filter_data(_x, lowcut, highcut, fs)
toc = time.perf_counter()
print(f" Butterworth filter executed in {toc - tic:0.4f} seconds")

# Plotting noisy test signal and filtered signal
# _x, _t, _f0 = precondition.generate_test_signal(T, fs, freq, a)
fname = 'test_signal_filtered_Signal.png'
precondition.plot_filtered_signal(_t, _x, x_filter, lowcut, highcut)
plt.show()
savefig(write_dir, fname)

fname = 'test_phase_locking.png'
precondition.plot_phase_locking(_t, a, _f0, x_filter)
plt.show()
savefig(write_dir, fname)

fname = 'freq_response_vs_filter_order.png'
precondition.plot_freq_response_vs_filter_order(_x, lowcut, highcut, fs)
plt.show()
savefig(write_dir, fname)

fname = 'plot_psd.png'
precondition.plot_psd(_x, x_filter, fs)
plt.show()
savefig(write_dir, fname)

tic = time.perf_counter()
f, psd_filter,mu = precondition.bandpass_multitaper_filter_data(_x, fs, NW = None, BW= None, adaptive = False, jackknife = True, sides = 'default')
toc = time.perf_counter()
print(f" Multitaper filter executed in {toc - tic:0.4f} seconds")

_f, psd = precondition.get_psd(_x, fs, np.shape(f)[0] )
fname = 'multitaper_powerspectrum.png'
precondition.plot_db_spectral_estimate(f, psd ,psd_filter, 'multitaper')
# nitime.viz.plot_spectral_estimate(f, psd_filter,(psd_filter,), elabels='multitaper')
plt.show()
savefig(write_dir, fname)

bands = [(0, 10), (100, 200), (560, 660),(2000, 2010)]
lfp = precondition.multitaper_lfp_bandpower(f,psd_filter, bands, 1, False)
plt.plot(np.arange(len(bands)),np.squeeze(lfp),'-bo')
plt.xticks(np.arange(len(bands)),bands)
plt.xlabel('Frequency band (Hz)')
plt.ylabel('Band Power')
plt.show()
fname = 'lfp_bandpower.png'
savefig(write_dir, fname)


class FilterTests(unittest.TestCase):

    def __init_subclass__(self, **kwargs):
        self.T = 0.05
        self.fs = 25000
        self.freq = [600, 312, 2000]
        self.a = 0.02
        # testing generate test_signal
        _x, _t, _f0 = precondition.generate_test_signal(self.T, self.fs, self.freq, self.a)

    # def test_butterworth(self):
    #     # Sample rate and desired cutoff frequencies (in Hz).
    #     # fs = 25000.0
    #     lowcut = 500.0
    #     highcut = 1200.0
    #     tic = time.perf_counter()
    #     self.x_filter = precondition.bandpass_butterworth_filter_data(_x, lowcut, highcut, fs)
    #     toc = time.perf_counter()
    #     print(f" Butterworth filter executed in {toc - tic:0.4f} seconds")
    #
    # def test_plot_filtered_signal(self):
    #     # Plotting noisy test signal and filtered signal
    #     _x, _t, _f0 = precondition.generate_test_signal(self.T, self.fs, self.freq, self.a)
    #     fname = 'test_signal_filtered_Signal.png'
    #     lowcut = 500.0
    #     highcut = 1200.0
    #     precondition.plot_filtered_signal(_t, _x, self.x_filter, lowcut, highcut)
    #     plt.show()
    #     savefig(write_dir, fname)
    #
    # def test_plot_phase_locking(self):
    #     # Plotting filtered signal with original signal frequency
    #     _x, _t, _f0 = precondition.generate_test_signal(self.T, self.fs, self.freq, self.a)
    #     fname = 'test_phase_locking.png'
    #     precondition.plot_phase_locking(_t, self.a, _f0, self.x_filter)
    #     plt.show()
    #     savefig(write_dir, fname)
    #
    # def test_plot_freq_response_vs_filter_order(self):
    #     fname = 'freq_response_vs_filter_order.png'
    #     precondition.plot_freq_response_vs_filter_order(self.x, self.lowcut, self.highcut, self.fs)
    #     plt.show()
    #     savefig(write_dir, fname)
    #
    # def test_plot_psd(self):
    #     fname = 'plot_psd.png'
    #     precondition.plot_psd(self.x, self.x_filter, self.fs)
    #     plt.show()
    #     savefig(write_dir, fname)
    #
    # def test_multitaper(self):
    #     f, psd_filter = precondition.bandpass_multitaper_filter_data(self.x, self.fs)
    #     psd = precondition.get_psd(self.x, self.fs)
    #     fname = 'multitaper_powerspectrum.png'
    #     precondition.plot_db_spectral_estimate(f, psd, psd_filter, 'multitaper')
    #     plt.show()
    #     savefig(write_dir, fname)


if __name__ == "__main__":
    unittest.main()