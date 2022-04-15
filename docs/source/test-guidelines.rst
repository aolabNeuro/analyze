Writing tests
=============

To run the test modules individually, call
``> python tests/module_name.py``

When reviewing someone's code, you might want to run all the tests at once using
``python -m unittest discover -s tests``

Write tests as you go
---------------------

For example, let's say you are writing a function to format a string to
read "You have ## fish", where ## is an input to the function. You might
start by writing a function shell something like this, in
``aopy/tutorial_functions.py``:

::

    def practice_func_leo(number_of_fish):
        '''
        This function formats a string to say how many fish you have

        Inputs:
            number_of_fish (int): How many fish you have
            
        Outputs:
            str: a string describing how many fish you have
        '''
        pass

Next, you should write a test function in
``tests/test_tutorial_functions.py``. Name your test function ``test_``
followed by the function name:

::

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
            self.assertEqual(output, 'You have 3 fish')

Since I know what the output should be, without programming anything, I
can set up the test to check that the function works.

Now I can implement the function:

::

    def practice_func_leo(number_of_fish):
        '''
        '''
        return 'You have 3 fish'

... and it passes the tests! But wait! This function completely ignores
the ``number_of_fish`` input and always says you have 3 fish. We need to
add a slightly better test. For example:

::

    def test_leo_practice(self):
        output = aopy.tutorial_functions.practice_func_leo(3)
        self.assertEqual(output, 'You have 3 fish')
        
        output = aopy.tutorial_functions.practice_func_leo(0)
        self.assertEqual(output, 'You have 0 fish')

And update the code to pass this test:

::

    def practice_func_leo(number_of_fish):
        '''
        '''
        return 'You have {} fish'.format(number_of_fish)

Of course, this is a contrived example and you wouldn't actually need to
write comprehensive tests on a one-line function.

Testing datasets
----------------

So you want to write a test function, but you don't know what to compare 
it against?

Keep it simple if possible.

For some functions, it is best to explicitly write out a minimal dataset 
that has all the features of a real dataset without actually being real data,
which can be large and hard to interpret. For example,

.. code-block:: python

    analog_data = [[2.9, 0.23], 
                   [1.9, 2.9], 
                   [1.74, 4.76]]
    thresh = 0.5
    expected_digital_array = [[1.0, 0.0], 
                                [0.0, 1.0], 
                                [0.0, 1.0]]
    digital_data = convert_analog_to_digital(analog_data, thresh)
    np.testing.assert_almost_equal(expected_digital_array, digital_data)

In other cases, such as a time series filtering function, it is best to have
synthetic test data that is computer-generated. For example,

.. code-block:: python

    band = [-500, 500] # signals within band can pass
    N = 0.005 # N*sampling_rate is time window you analyze
    NW = (band[1]-band[0])/2
    T = 0.05
    fs = 25000
    nch = 1
    x_312hz = utils.generate_multichannel_test_signal(T, fs, nch, 312, 1.5)
    x_600hz = utils.generate_multichannel_test_signal(T, fs, nch, 600, 0.5)
    f0 = np.mean(band)
    tapers = [N, NW]
    x_mtfilter = precondition.mtfilter(x_312hz + x_600hz, tapers, fs=fs, f0=f0)
    plt.figure()
    plt.plot(x_312hz + x_600hz, label='Original signal (312 Hz + 600 Hz)')
    plt.plot(x_312hz, label='Original signal (312 Hz)')
    plt.plot(x_mtfilter, label='Multitaper-filtered signal')
    plt.xlim([0,500])
    plt.legend()
    fname = 'mtfilter.png'
    savefig(write_dir, fname) # Should have power in 312 Hz but not 600

Finally, it is impractical to generate test data in some cases. For example,

.. code-block:: python

    result_filename = 'test_proc_eyetracking_short.hdf'
    files = {}
    files['ecube'] = '2021-09-30_BMI3D_te2952'
    files['hdf'] = 'beig20210930_02_te2952.hdf'

    # Should fail because no preprocessed experimental data
    if os.path.exists(os.path.join(write_dir, result_filename)):
        os.remove(os.path.join(write_dir, result_filename))
    self.assertRaises(ValueError, lambda: proc_eyetracking(data_dir, files, write_dir, result_filename))

    proc_exp(data_dir, files, write_dir, result_filename)

    # Should fail because not enough trials in this session
    self.assertRaises(ValueError, lambda: proc_eyetracking(data_dir, files, write_dir, result_filename))

    result_filename = 'test_proc_eyetracking.hdf'
    files['ecube'] = '2021-09-29_BMI3D_te2949'
    files['hdf'] = 'beig20210929_02_te2949.hdf'
    if os.path.exists(os.path.join(write_dir, result_filename)):
        os.remove(os.path.join(write_dir, result_filename))
    proc_exp(data_dir, files, write_dir, result_filename)

    # Test that eye calibration is returned, but results are not saved
    eye, meta = proc_eyetracking(data_dir, files, write_dir, result_filename, save_res=False)
    self.assertIsNotNone(eye)
    self.assertIsNotNone(meta)
    self.assertRaises(ValueError, lambda: load_hdf_group(write_dir, result_filename, 'eye_data'))
    self.assertRaises(ValueError, lambda: load_hdf_group(write_dir, result_filename, 'eye_metadata'))

    # Test that eye calibration is saved
    proc_eyetracking(data_dir, files, write_dir, result_filename, save_res=True)
    eye = load_hdf_group(write_dir, result_filename, 'eye_data')
    meta = load_hdf_group(write_dir, result_filename, 'eye_metadata')
    self.assertIsNotNone(eye)
    self.assertIsNotNone(meta)


.. list-table:: List of test datasets included in /tests/data/
   :widths: 25 75
   :header-rows: 1
   
   * - Filename
     - Description
   * - `test20210310_08_te1039.hdf`
     - hdf data for testing sync version 0
   * - `2021-04-07_BMI3D_te1315`
     - eCube data - sync version 2, including Analog and Digital data
   * - `beig20210407_01_te1315.hdf`
     - BMI3D data - sync version 2
   * - `2021-06-14_BMI3D_te1825`
     - eCube data - sync version 4
   * - `beig20210614_07_te1825.hdf`
     - BMI3D data - sync version 4
   * - `2021-09-29_BMI3D_te2949`
     - eCube data - eyetracking test
   * - `beig20210929_02_te2949.hdf`
     - BMI3D data - eyetracking test
   * - `2021-09-30_BMI3D_te2952`
     - eCube data - another eyetracking test
   * - `beig20210930_02_te2952.hdf`
     - BMI3D data - another eyetracking test
   * - `2021-12-13_BMI3D_te3498`
     - eCube data - sync version 7
   * - `test20210330_12_te1254.hdf`
     - BMI3D data - sync version 2
   * - `fake ecube data`
     - eCube data - including fake headstage data with 8 channels from Headstages, plus Analog and Digital data from a test recoridng of BMI3D
   * - `fake_ecube_data_bmi3d.hdf`
     - BMI3D data - from a test recording
   * - `short headstage test`
     - recoring of noise from 64 channel ecube headstage
   * - `244ch_viventi_ecog_elec_to_pos.xlsx`
     - example channel map definition file
   * - `210118_ecog_channel_map.xlsx`
     - example channel mapping file
   * - `example_wfs.npy`
     - waveforms
   * - `matlab_cell_str.mat`
     - cell string saved as a mat file
   * - `Pretend take (1315).csv`
     - optitrack data exported into csv format
   * - `Take 2021-03-10 17_56_55 (1039).csv`
     - optitrack data exported into csv format
   * - `Take 2021-04-06 11_47_54 (1312).csv`
     - optitrack data exported into csv format
   * - `task_codes.yaml`
     - yaml formatted file with some task codes
