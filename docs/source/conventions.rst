Conventions
===========

Timeseries data:

-  always order time in the first dimension and channels in the second
   dimension
-  label with ``_ts``

Keeping track of files:

-  use a separate ``data_dir`` and ``filename`` if your function loads or saves data
-  use the common ``files`` dictionary if your function inputs or outputs file from multiple systems

Plotting functions:

-  take an axis as input your function can be used on a subplot
-  don't create new figures in a function (in general), plot onto existing ones
-  allow for user-defined settings (e.g. color, line width, etc.)

Some commonly used variables:
-----------------------------

+---------------+-------------+---------------------------------------+-------+
| variable name | type        | description                           | count |
+===============+=============+=======================================+=======+
| data_dir      | str         | directory where data is located       | 28    |
+---------------+-------------+---------------------------------------+-------+
| exp_data      | dict        | dictionary containing experiment data | 24    |
+---------------+-------------+---------------------------------------+-------+
| exp_metadata  | dict        | experiment metadata                   | 12    |
+---------------+-------------+---------------------------------------+-------+
| samplerate    | float       | sampling rate of some data            | 11    |
+---------------+-------------+---------------------------------------+-------+
| ax            | pyplot.Axes | figure axis                           | 10    |
+---------------+-------------+---------------------------------------+-------+
| filename      | str         |                                       | 9     |
+---------------+-------------+---------------------------------------+-------+
| filepath      | str         |                                       | 8     |
+---------------+-------------+---------------------------------------+-------+
| files         | dict        |                                       | 8     |
+---------------+-------------+---------------------------------------+-------+
| timestamps    | float       | reference values                      |       |
+---------------+-------------+---------------------------------------+-------+