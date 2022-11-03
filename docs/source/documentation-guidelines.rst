Writing documentation
=====================

Docstrings
----------

All functions are required to have a header placed under the function
definition that is based on the google docstring style. The header
should be enough for someone who has never seen the function before to
be able to use it and understand the outputs.

Examples:
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

The analyze github repo is hooked into
`readthedocs <https://analyze.readthedocs.io/en/latest>`__ so that most
changes to the code are automatically reflected in the documentation.
This documentation is generated from the .rst files located in
``docs/source/``. Each .rst file is named according to a module in
``aopy`` and contains a high-level overview of the module as well as the
API reference for all the functions within that module.

Template
~~~~~~~~

::

    '''
    Broad description of what the function does.

    Example:
        Example of how to use the function.
        Include any images / equations, etc. that would help explain how to use the function.
        
    Args:
        varIn_name1 (dim label1, dim label2): Description
        varIn_name2 (data type): Description
            
    Returns:
        type: description
    '''

Each argument should be named exactly as it appears in the function definition, and labeled as follows:

-  if it is a single value, what is the type (e.g. `float`, `int`, `str`)
-  for an array of values, what is it's size (e.g. `nt` for number of timepoints, `ntr` for number of trials, or `nt, ntr` for a 2d array of time x trials)
-  for any other object, the type (e.g. `pyplot.Axes`)

Example
~~~~~~~

::

   def locate_trials_with_event(trial_events, event_codes, event_columnidx=None):
    '''
    Given an array of trial separated events, this function goes through and finds the event sequences corresponding to the trials
    that include a given event. If an array of event codes are input, the function will find the trials corresponding to
    each event code. 

    Args:
        trial_events (ntr, nevents): Array of trial separated event codes
        event_codes (int, str, list, or 1D array): Event code(s) to find trials for. Can be a list of strings or ints
        event_column (int): Column index to look for events in. Indexing starts at 0. Keep as 'None' if all columns should be analyzed.
        
    Returns:
        tuple: Tuple containing:
            | **list of arrays:** List where each index includes an array of trials containing the event_code corresponding to that index. 
            | **1D Array:** Concatenated indices for which trials correspond to which event code.
                        Can be used as indices to order 'trial_events' by the 'event_codes' input.
    Example:
    ::
        >>> aligned_events_str = np.array([['Go', 'Target 1', 'Target 1'],
                ['Go', 'Target 2', 'Target 2'],
                ['Go', 'Target 4', 'Target 1'],
                ['Go', 'Target 1', 'Target 2'],
                ['Go', 'Target 2', 'Target 1'],
                ['Go', 'Target 3', 'Target 1']])
        >>> split_events, split_events_combined = locate_trials_with_event(aligned_events_str, ['Target 1','Target 2'])
        >>> print(split_events)
        [array([0, 2, 3, 4, 5], dtype=int64), array([1, 3, 4], dtype=int64)]
        >>> print(split_events_combined)
        [0 2 3 4 5 1 3 4]   
        '''   

::

    def plot_raster(data, cue_bin=None, ax=None):
        '''
        Create a raster plot for binary input data and show the relative timing of an event with a vertical red line

        .. image:: _images/raster_plot_example.png

        Args:
            data (ntime, ncolumns): 2D array of data. Typically a time series of spiking events across channels or trials (not spike count- must contain only 0 or 1).
            cue_bin (float): time bin at which an event occurs. Leave as 'None' to only plot data. For example: Use this to indicate 'Go Cue' or 'Leave center' timing.
            ax (plt.Axis): axis to plot raster plot
            
        Returns:
            None: raster plot plotted in appropriate axis
        '''

Tips for vscode
~~~~~~~~~~~~~~~

Use the reStructuredText extension to view a preview of .rst files

You can also use the Python Docstring Generator extension to
automatically generate docstrings for your functions.

Adding images
^^^^^^^^^^^^^

In your docstring or in the .rst file of your choice include the line

::

    .. image:: _images/your-image.png

Then put your image into /docs/source/\_images/

Adding equations
^^^^^^^^^^^^^^^^

You can include any LaTeX equations in the documentation.

In your docstring include:

::

    .. math:: \\frac{ \\sum_{t=0}^{N}f(t,k) }{N}

Notice the double backslash ``\\`` to delimit the escape character in
python.

To add math to an .rst file of your choice:

::

    Math equation on its own line:
    .. math:: 

        \frac{ \sum_{t=0}^{N}f(t,k) }{N}

Math equation on its own line:
    .. math:: 

        \frac{ \sum_{t=0}^{N}f(t,k) }{N}

::

    Inline math: :math:`\theta`

Inline math: :math:`\theta`

Documentation pages
-------------------

In addition to function docstrings, there is also considerable documentation here
on readthedocs. There is this section, :ref:`Contributing:`, as well as a :ref:`Getting started:`
guide, an :ref:`Examples:` page, and many submodules containing general information,
for example the :ref:`Preprocessed data format` page. 

It is recommended to add documentation in this way whenever there is information relevant
to many functions. For example, some documentation should be provided for each submodule
in the repository, i.e. :ref:`Precondition:`, :ref:`Data:`, etc. To add documentation to
these submodule pages, simply edit the relevant `.rst` file and include your prose at the 
top of the file before the block that looks like:

.. code-block:: rst

    API
    ---

    .. automodule:: aopy.postproc
        :members:

It is also recommended to include :ref:`Examples:` of common workflows, for instance
trial-aligning kinematics and neural data. To add an example page, upload your notebook
to `docs/source/examples/`, then add its filename into the `examples.rst` document.
Be sure that your notebook has a title at the beginning, otherwise it won't show up in
the table of contents.
