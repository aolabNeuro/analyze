Getting Started:
================

Example usage
-------------
Data from experiments comes from several sources (e.g. experiment hdf files, optitrack csv, binary neural data).
To manage all these different sources of data, aopy has parsing functions that standardize the format and 
contents of these files.

A simple example:

::

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

Once preprocessed, you can inspect the hdf file using ``aopy.data.get_hdf_contents()``:

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

See :doc:`preproc.rst` for more details on the data format. 
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


Integrating with BMI3D
----------------------

coming soon!