Writing code
============

The purpose of this repository is to facilitate neural data analysis
by providing a platform for sharing and collaborating on code. To 
that end, all code related to the analysis of neural data is welcomed.
However, to avoid excessive clutter in the repository, use the following
guidelines to decide whether to include a piece of code in the repository.

- If you use something more than once in your analysis, it is probably
  a good idea to include it in the repository. This way, you won't be 
  copying and pasting code all the time.
- If you are writing something susceptible to errors, for instance
  functions that compute statistics or perform some kind of
  transformation on data, it is probably a good idea to include it in the
  repository so that tests can confirm that your code is correct and that
  others won't have to redo your efforts to do the same computations.
- If there are existing packages that perform similar analysis, it is 
  generally better to use those packages rather than writing your own. 
  In these cases it is also a good idea to include the package in these
  documentation pages so that others can find it. Feel free to add sections 
  to `preproc`, `analysis`, `visualization`, `utils`, etc. if you have
  links to useful code not included in this repository. One exception
  to this rule is if you regularly use a package that needs some 
  customization, for instance transposing the input data, then including 
  a wrapper function here that does this customization is acceptable.

Modules
-------

When writing functions, please organize them into the following modules: 

+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| Module        | Purpose                                                                                                                                        | Examples                                                                              |
+===============+================================================================================================================================================+=======================================================================================+
| data          | Code for directly loading and saving data (and results)                                                                                        | :func:`aopy.data.load_ecube_data`, :func:`aopy.data.save_hdf`                         |
+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| preproc       | Code for preprocessing neural data (reorganize data into the needed form) including parsing experimental files, trial sorting, and subsampling | :func:`aopy.preproc.get_trial_segments`, :func:`aopy.preproc.proc_exp`                |
+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| precondition  | Code for cleaning and preparing neural data for users to interact with, for example: down-sampling, outlier detection, and initial filtering   | :func:`aopy.precondition.get_psd_multitaper`, :func:`aopy.precondition.detect_spikes` |
+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| postproc      | Code for post-processing neural data, including separating neural features such as LFP bands or spikes binning                                 |                                                                                       |
+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| analysis      | Code for neural data analysis; functions here should return interpretable results such as firing rates, success rates, direction tuning, etc.  | :func:`aopy.analysis.calc_success_rate`, :func:`aopy.analysis.calc_rms`               |
+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| visualization | Code for general neural data plotting (raster plots, multi-channel field potential plots, psth, etc.)                                          | :func:`aopy.visualization.plot_spatial_map`, :func:`aopy.visualization.plot_raster`   |
+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| utils         | Helper functions, math, other things that don't really pertain to neural data analysis                                                         | :func:`aopy.utils.generate_test_signal`, :func:`aopy.utils.detect_edges`              |
+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+

Style
-----

In general, follow the PEP 8 style guidelines.

Function Checklist
------------------

-  Arguments have generic datatypes
-  All variables are consistent with the standard naming convention
-  Input and output arguments are in the proper order.
-  Header includes all required information from the template
-  Comments throughout the code fit the fulfilling the requirements

Generic Datatypes
~~~~~~~~~~~~~~~~~

Function arguments should be as general as possible to allow for 
flexibility of the user. For example, if your analysis loads data 
and metadata in a specific format, try to write functions that load
that data separately from functions that perform analysis on it, so 
that other people who want to do the same analysis can use your
function on their own data format.

Let's say I have a dictionary `trials` that contains my experimental
data, broken down by trial. I want to calculate the average firing rate
for each trial.

.. code-block:: python
   
   def calc_avg_firing_rate(trials):
       avg_firing_rate = []
       for trial in trials:
           avg_firing_rate.append(np.mean(trial['spikes']))
       return avg_firing_rate

This is the wrong way to write this function because it relies on the
data being in a specific format. Instead, I want to write a function
that takes in just the spiking data from each trial instead of the entire
dictionary.

.. code-block:: python
   
   def calc_avg_firing_rate(spikes):
       avg_firing_rate = []
       for trial in spikes:
           avg_firing_rate.append(np.mean(trial))
       return avg_firing_rate

It may be tempting to group datasets together into a single dictionary,
when the list of function inputs starts to get large, but please avoid 
doing so because it puts more burden on the user to format the data in
your special way.

Consistent Variables
~~~~~~~~~~~~~~~~~~~~

All variables (input args, output args, local variables) names and
format should be standardized according to the standard convention.

Timeseries data:

-  always order time in the first dimension and channels in the second
   dimension
-  label with ``_ts``

Keeping track of files:

-  use a separate ``data_dir`` and ``filename`` if your function loads or saves data
-  use the common ``files`` dictionary if your function inputs or outputs file from multiple systems

Plotting functions:

-  take an axis as input so your function can be used on a subplot
-  don't create new figures in a function (in general), plot onto existing ones
-  allow for user-defined settings (e.g. don't fix color, line width, etc.)

Some commonly used variables:

+---------------+-------------+----------------------------------+
| variable name | type        | description                      |
+===============+=============+==================================+
| data_dir      | str         | directory where data is located  |
+---------------+-------------+----------------------------------+
| filename      | str         | basename of a file               |
+---------------+-------------+----------------------------------+
| filepath      | str         | full filepath including diretory |
+---------------+-------------+----------------------------------+
| samplerate    | float       | sampling rate of some data       |
+---------------+-------------+----------------------------------+
| ax            | pyplot.Axes | figure axis                      |
+---------------+-------------+----------------------------------+
| positions     | (nt, 3)     | array of 3d positions in time    |
+---------------+-------------+----------------------------------+
| trajectories  | list        | list of positions timeseries     |
+---------------+-------------+----------------------------------+
| files         | dict        | dictionary of (system, filepath) |
+---------------+-------------+----------------------------------+
| timestamps    | float       | reference values                 |
+---------------+-------------+----------------------------------+

Argument order
~~~~~~~~~~~~~~

All input and output variables should maintain the following order:

#. Data (arrays, file path strings)
#. Function specifics
#. Plotting information
#. Save information

Data inputs are required and should not be keyworded. Function-specific,
plotting and saving parameters should be keyworded with default values.

General Comments
~~~~~~~~~~~~~~~~

Comments should be included whenever any of the following conditions are
met:

-  New local variable is defined. (Include units if applicable)
-  Major analysis section
-  Plotting
-  Saving

Imports
~~~~~~~

Imports should be organized in the following order:

-  standard library imports
-  third party imports
-  local application imports

Imports should be grouped by a single blank line. For example:

.. code-block:: python

   import os
   import sys

   import numpy as np
   import matplotlib.pyplot as plt

   from .. import data as aodata

To avoid circular import errors, do not import functions directly from within aopy. 
Instead, import a module and use the module name to access the function. For example:

.. code-block:: python

   from ..data import load_ecube_data # wrong, will lead to circular imports!
   
   from .. import data as aodata
   aodata.load_ecube_data()

Adding new packages
-------------------

If you need to add a new dependency to the repo, there are a couple items
that must also be completed:

#. Include your new dependency in setup.py, inside the `install_requires = []` list.
   Keep in mind that the name of the package on pypi might be different than what you
   import in python. For example, the PyWavelets package is imported with `import pywt`
   but installed with `pip install PyWavelets`. Here you would list `PyWavelets` as
   the dependency.
#. Include your new dependency in docs/source/conf.py inside the 
   `autodoc_mock_imports = []` list. Here you should type the module name that you import
   in python. For example, `pywt` for PyWavelets.
#. Finally, please notify the `coding` channel on the aolab slack that you have made 
   changes that require users to re-install the package with `pip install -e .`
