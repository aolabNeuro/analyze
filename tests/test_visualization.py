import unittest
from aopy.visualization import *
import numpy as np

write_dir = 'tests/tmp'

class PlottingTests(unittest.TestCase):

    def test_plot_timeseries(self):
        filename = 'timeseries.png'
        data = np.reshape(np.sin(np.arange(1000)/np.pi) + np.sin(2*np.arange(1000)),(1000))
        samplerate = 1000
        plt.figure()
        plot_timeseries(data, samplerate)
        savefig(write_dir, filename)

    def test_plot_freq_domain(self):
        filename = 'freqdomain.png'
        data = np.reshape(np.sin(np.arange(1000)/np.pi) + np.sin(2*np.arange(1000)),(1000))
        freq_data = np.fft.fft(data)
        samplerate = 1000
        plt.figure()
        plot_freq_domain(freq_data, samplerate)
        savefig(write_dir, filename)

    def test_plot_data_on_pos(self):
        filename = 'posmap.png'
        plt.figure()
        data = np.linspace(-1, 1, 100)
        x_pos, y_pos = np.meshgrid(np.arange(0.5,10.5),np.arange(0.5, 10.5))
        missing = [0, 5, 25]
        data_missing = np.delete(data, missing)
        x_missing = np.delete(x_pos, missing)
        y_missing = np.delete(y_pos, missing)
        plot_data_on_pos(data, np.reshape(x_pos,-1), np.reshape(y_pos,-1), [10, 10])
        plt.figure()
        plot_data_on_pos(data_missing, np.reshape(x_missing,-1), np.reshape(y_missing,-1), [10, 10])
        savefig(write_dir, filename)

if __name__ == "__main__":
    unittest.main()