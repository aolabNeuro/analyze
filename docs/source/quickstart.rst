Getting Started:
================

.. contents:: :local:

Installation
------------

Clone the repo from github, then install. If you are planning to make changes, use the -e flag to install
in editable mode rather than installing a fixed version.

::

    > git clone https://github.com/aolabNeuro/analyze.git
    > cd analyze
    > pip install -e .

Overview of functions
---------------------

+---------------+-----------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| Module        | contents                                                                                                  | Examples                                                                                                                       |
+===============+===========================================================================================================+================================================================================================================================+
| data          | Directly loading and saving data from bmi3d, peslab, and results                                          | :func:`~aopy.data.load_ecube_data`, :func:`~aopy.data.save_hdf`                                                                |
+---------------+-----------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| preproc       | Reorganize data into a standard format, largely automated for bmi3d                                       | :func:`~aopy.preproc.get_trial_segments`, :func:`~aopy.preproc.proc_exp`                                                       |
+---------------+-----------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| precondition  | Clean and prepare neural data for users to interact with                                                  | :func:`~aopy.precondition.downsample`, :func:`~aopy.precondition.get_psd_multitaper`, :func:`~aopy.precondition.detect_spikes` |
+---------------+-----------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| postproc      | Separating neural features such as LFP bands or spikes binning. And (currently) loading preprocessed data | :func:`~aopy.postproc.extract_mtm_features`, :func:`~aopy.postproc.get_kinematic_segments`                                     |
+---------------+-----------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| analysis      | Compute firing rates, success rates, direction tuning, etc.                                               | :func:`~aopy.analysis.calc_success_rate`, :func:`~aopy.analysis.calc_rms`                                                      |
+---------------+-----------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| visualization | Neural data plotting                                                                                      | :func:`~aopy.visualization.plot_spatial_map`, :func:`~aopy.visualization.plot_raster`, :func:`~aopy.visualization.plot_tfr`    |
+---------------+-----------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| utils         | Helper functions, math, other things that don't really pertain to neural data analysis                    | :func:`~aopy.utils.generate_test_signal`, :func:`~aopy.utils.detect_edges`, :func:`~aopy.utils.derivative`                     |
+---------------+-----------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+

Supported systems
-----------------

Currently aopy supports data from aolab BMI3D and pesaran lab wireless data.

Data from experiments comes from several sources (e.g. experiment hdf files, optitrack csv, binary neural data).
To manage all these different sources of data, aopy has parsing functions that standardize the format and 
contents of these files.

BMI3D
^^^^^

A simple example:

.. code-block:: console

    /data/raw/   
    ├── hdf/
    |   ├── test20210310_08_te1039.hdf
    │   └── ...
    ├── ecube/
    |   ├── 2021-03-10_BMI3D_te1039/
    │   |   ├── AnalogPanel_32_Channels_int16_2021-03-10_10-03-58
    |   |   └── DigitalPanel_64_Channels_bool_masked_uint64_2021-03-10_10-03-58
    │   └── ...

.. code-block:: python

    import aopy
    data_dir = '/data/raw'
    result_dir = '/data/preprocessed/beignet'
    block = 1039
    files = aopy.data.get_filenames_in_dir(data_dir, block)
    result_filename = aopy.data.get_exp_filename(block)
    aopy.preproc.proc_exp(data_dir, files, result_dir, result_filename)

Once preprocessed, you can inspect the hdf file using ``aopy.data.get_hdf_dictionary()``:

::

    preprocessed_te1039.hdf   
    ├── exp_data
    │   ├── task
    │   ├── state
    │   ├── clock
    │   ├── events
    │   ├── trials
    │   └── <raw bmi3d data>
    └── exp_metadata
        ├── source_dir
        ├── source_files
        ├── n_cycles
        ├── n_trials
        ├── bmi3d_start_time
        └── <raw bmi3d metadata>

See :doc:`preproc` for more details on the data format. 
To add mocap and spiking data you would call:

.. code-block:: python

    aopy.preproc.proc_mocap(data_dir, files, result_dir, result_filename)
    aopy.preproc.proc_spikes(data_dir, files, result_dir, result_filename)

The hdf file would now contain:

::

    preprocessed_te1039.hdf   
    ├── exp_data
    │   └── ...
    ├── exp_metadata
    │   └── ...
    ├── mocap_data
    │   └── data
    ├── mocap_metadata
    |   ├── samplerate
    │   ├── source_dir
    |   ├── source_files
    |   └── <raw mocap metadata>
    ├── spikes_data
    │   └── ...
    └── spikes_metadata
        └── ...

(proc_spikes doesn't actually exist as of this writing)

To load a single variable from the preprocessed file, use:

.. code-block:: python

    trials = aopy.data.load_hdf_data(result_dir, result_filename, 'trials', 'exp_data')

Or to load an entire group:

.. code-block:: python

    exp_metadata = aopy.data.load_hdf_group(result_dir, result_filename, 'exp_metadata')

For a more comprehensive example, see the Examples section of this documentation.

Peslab
^^^^^^

Documentation in progress.

.. code-block:: python

    aopy.data.peslab
