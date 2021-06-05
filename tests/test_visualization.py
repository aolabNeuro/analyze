import unittest
from aopy.visualization import *
import numpy as np
import os

test_dir = os.path.dirname(__file__)
write_dir = os.path.join(test_dir, 'tmp')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

class PlottingTests(unittest.TestCase):

    def test_plot_timeseries(self):
        filename = 'timeseries.png'
        data = np.reshape(np.sin(np.pi*np.arange(1000)/10) + np.sin(2*np.pi*np.arange(1000)/10), (1000))
        samplerate = 1000
        plt.figure()
        plot_timeseries(data, samplerate)
        savefig(write_dir, filename)

    def test_plot_freq_domain_power(self):
        filename = 'freqdomain.png'
        data = np.reshape(np.sin(np.pi*np.arange(1000)/10) + np.sin(2*np.pi*np.arange(1000)/10), (1000))
        samplerate = 1000
        plt.figure()
        plot_freq_domain_power(data, samplerate)
        savefig(write_dir, filename)

    def test_spatial_map(self):
        data = np.linspace(-1, 1, 100)
        x_pos, y_pos = np.meshgrid(np.arange(0.5,10.5),np.arange(0.5, 10.5))
        missing = [0, 5, 25]
        data_missing = np.delete(data, missing)
        x_missing = np.reshape(np.delete(x_pos, missing),-1)
        y_missing = np.reshape(np.delete(y_pos, missing),-1)

        filename = 'posmap.png'
        data_map = get_data_map(data_missing, x_missing, y_missing)
        self.assertEqual(data_map.shape, (10, 10))
        self.assertTrue(np.isnan(data_map[0,0]))
        plt.figure()
        plot_spatial_map(data_map, x_missing, y_missing)
        savefig(write_dir, filename)

        filename = 'posmap_interp.png'
        interp_map = calc_data_map(data_missing, x_missing, y_missing, [10, 10], threshold_dist=0.01)
        self.assertEqual(interp_map.shape, (10, 10))
        self.assertTrue(np.isnan(interp_map[0,0]))
        plot_spatial_map(interp_map, x_missing, y_missing)
        savefig(write_dir, filename)

    def test_animate_events(self):
        events = ["hello", "world", "", "!", ""]
        times = [0., 1.0, 1.5, 2.0, 2.5]
        fps = 10
        filename = os.path.join(write_dir, "animate_test.mp4")
        ani = animate_events(events, times, fps)

        from matplotlib.animation import FFMpegFileWriter
        kwargs = {'transparent': True,}
        writer = FFMpegFileWriter(fps=fps)
        ani.save(filename, dpi=300, writer=writer, savefig_kwargs=kwargs)

    def test_saveanim(self):
        events = ["hello", "world", "", "!", ""]
        times = [0., 1.0, 1.5, 2.0, 2.5]
        fps = 10
        filename = "animate_test_save.mp4"
        ani = animate_events(events, times, fps)
        saveanim(ani, write_dir, filename)

    def test_showanim(self):
        # don't know how to test this. trust me it works :)
        pass

    def test_animate_trajectory_3d(self):
        trajectory = np.zeros((10,3))
        trajectory[:,0] = np.arange(10)
        samplerate = 2
        axis_labels = ['x = Right', 'y = Forwards', 'z = Up']
        ani = animate_trajectory_3d(trajectory, samplerate, history=5, axis_labels=axis_labels)
        filename = "animate_trajectory_test.mp4"
        saveanim(ani, write_dir, filename)

if __name__ == "__main__":
    unittest.main()