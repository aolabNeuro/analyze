# we are generating noisy test data using sine and cosine functions with multiple frequencies

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import freqz
from aopy import precondition


# # Generating a random noise for testing

# test_freq = []
def generate_test_signal(T, fs, freq, a= 0.02):
    '''

    Args:
        T (float): Time period in seconds
        fs (int): Sampling frequency in Hz
        freq (1D array): Frequencies to be mixed in the test signal. main frequency in the first element
        plot_true (bool): boolean value if true, plots the test signal in the provided axes
        ax (plot axis handle): axis to plot the noisy test signal

    Returns:
        x (1D array): cosine wave with multiple frequencies and noise
    '''
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    # a = 0.02
    f0 = freq[0]
    # noise_power = 0.001 * fs / 2

    x = a * np.cos(2 * np.pi * f0 * t) # adding primary frequency
    x += a*0.5 * np.cos(2 * np.pi * freq[1] * t)
    x += a*1.5 * np.cos(2 * np.pi * freq[2] * t)
    x += a*20 * np.sin(2 * np.pi * 1.2 * np.sqrt(t)) # noise
    # x += np.random.normal(scale=np.sqrt(noise_power), size=t.shape)  # noise

    return x, t, f0

# testing generate test_signal
T = 0.05
fs = 25000
freq = [600, 312, 2000]
a = 0.02
x, t, f0= generate_test_signal(T, fs, freq, a)

# Sample rate and desired cutoff frequencies (in Hz).
# fs = 25000.0
lowcut = 500.0
highcut = 1200.0
y = precondition.bandpass_butterworth_filter_data(x, lowcut, highcut, fs)

# Plotting noisy test signal and filtered signal
plt.figure()
plt.clf()
plt.plot(t, x, label='Noisy signal')
plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
plt.xlabel('time (seconds)')
plt.hlines([-a, a], 0, T, linestyles='--')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='best')
plt.show()

# Plotting filtered signal with original signal frequency
plt.figure(1)
plt.clf()
x_f0 = a * np.cos(2 * np.pi * f0 * t)
plt.plot(t, x_f0, label='Original signal (%g Hz)' % f0)
plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
plt.xlabel('time (seconds)')
plt.ylabel('amplitude')
plt.title('Comparison of Original Vs Filtered Signal')
plt.legend(loc='best')
plt.show()

# Plot the frequency response for a few different orders
plt.figure(2)
plt.clf()
for order in [2,3,4,5,6]: # trying  different order of butterworth to see the roll off around cut-off frequencies
    precondition.bandpass_butterworth_filter_data(x, lowcut, highcut, fs, order = order)

    b, a = precondition.bandpass_butterworth_params(lowcut, highcut,fs, order = order)
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')
plt.title('Comparison of Frequency response for Diff. Orders of Butterworth Filter')
plt.show()

# Plot power spectral density of the signal

f, psd = signal.welch(x, fs, average = 'mean')
f_filtered, psd_filterred = signal.welch(y, fs, average='median')
plt.figure(3)
plt.semilogy(f, psd, label='test signal')
plt.semilogy(f_filtered, psd_filterred, label='filtered output')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.legend()
plt.title('Power Spectral Density Comparison')
plt.show()

# Testing Multitaper filter
window_len = 100
std_half_bw = 5/window_len

slepian_seq, slepian_data = precondition.bandpass_multitaper_filter_data(x,lowcut,highcut,fs,window_len,std_half_bw)

fig, ax = plt.subplots(8, 2, figsize = (9,7))
for i in range(8):
    ax[i,0].plot(slepian_seq[i,:], label='slepian sequence')
    ax[i,1].plot(slepian_data[i], '-k', label='slepian seq * test data')
    # ax[i,1].plot(x, '--b', label = 'test data')
plt.tight_layout()
plt.suptitle('Slepian Functions in Time Domain')
plt.show()


fig, ax = plt.subplots(8, 2, figsize = (9,7))
for i in range(8):
    w,h = freqz(slepian_seq[i,:])
    ax[i, 0].plot(w, 20 * np.log10(np.abs(h)), lw=1.0)

    w2,h2 = freqz(slepian_data[i])
    ax[i,1].plot(w2, 20*np.log10(np.abs(h2)), '-k', lw =1.0)
plt.tight_layout()
plt.suptitle('Slepian Functions in Frequency Domain')
plt.show()