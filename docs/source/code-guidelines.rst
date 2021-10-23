Writing code
============

Modules
-------

When writing functions, please organize them into the following modules: 

+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| Module        | Purpose                                                                                                                                         | Examples                                                                                 |
+===============+=================================================================================================================================================+==========================================================================================+
| data          | Code for directly loading and saving data (and results)                                                                                         | :func:`aopy.data.load_ecube_data`, :func:`aopy.data.save_hdf`                            |
+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| preproc       | Code for preprocessing neural data (reorganize data into the needed form) including parsing experimental files, trial sorting, and subsampling  | :func:`aopy.preproc.get_trial_segments`, :func:`aopy.preproc.proc_exp`                   |
+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| precondition  | Code for cleaning and preparing neural data for users to interact with, for example: down-sampling, outlier detection, and initial filtering    | :func:`aopy.precondition.get_psd_multitaper`, :func:`aopy.precondition.detect_spikes`    |
+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| postproc      | Code for post-processing neural data, including separating neural features such as LFP bands or spikes detection / binning                      |                                                                                          |
+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| analysis      | Code for neural data analysis; functions here should return interpretable results such as firing rates, success rates, direction tuning, etc.   | :func:`aopy.analysis.calc_success_rate`, :func:`aopy.analysis.calc_rms`                  |
+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| visualization | Code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)                                           | :func:`aopy.visualization.plot_spatial_map`, :func:`aopy.visualization.plot_raster`      |
+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| utils         | Helper functions, math, other things that don't really pertain to neural data analysis                                                          | :func:`aopy.utils.generate_test_signal`, :func:`aopy.utils.detect_edges`                 |
+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------+

Writing code
------------

In general, follow the PEP 8 style guidelines.

Function Checklist
------------------

-  All variables are consistent with the standard naming convention
-  Input and output arguments are in the proper order.
-  Header includes all required information from the template
-  Comments throughout the code fit the fulfilling the requirements

Consistent Variables
--------------------

All variables (input args, output args, local variables) names and
format should be standardized according to the standard convention.

https://docs.google.com/spreadsheets/d/1-zCbqKLzqmr3iDg494yDx2NqNXF254r9CA05bTEyeaM/edit#gid=0

Argument order
--------------

All input and output variables should maintain the following order:

#. Data (arrays, file path strings)
#. Function specifics
#. Plotting information
#. Save information

Data inputs are required and should not be keyworded. Function-specific,
plotting and saving parameters should be keyworded with default values.

General Comments
----------------

Comments should be included whenever any of the following conditions are
met:

-  New local variable is defined. (Include units if applicable)
-  Major analysis section
-  Plotting
-  Saving

