Analysis:
=========
Functions in this module perform analysis on data and output interpretable results.

Example functions include calculating firing rate, successrate, direction tuning, dimensionality, among others.

.. contents:: :local:

API
---

The analysis module is broken up into the following submodules. The functions and classes within 
`base`, `behavior`, `celltype`, `tuning`, and `kfdecoder` are accessible directly from `aopy.analysis`.
Additional submodules can be accessed with the appropriate `aopy.analysis.<submodule>` import.

Base
^^^^

.. automodule:: aopy.analysis.base
    :members:

Behavior
^^^^^^^^

.. automodule:: aopy.analysis.behavior
    :members:

CellType
^^^^^^^^

.. automodule:: aopy.analysis.celltype
    :members:

Tuning
^^^^^^

.. automodule:: aopy.analysis.tuning
    :members:

KFDecoder
^^^^^^^^^

.. automodule:: aopy.analysis.kfdecoder
    :members:

Latency
^^^^^^

Code for determining the latency of responses to stimulation

AccLLR is based on the paper:

Banerjee A, Dean HL, Pesaran B. A likelihood method for computing selection times 
in spiking and local field potential activity. J Neurophysiol. 2010 Dec;104(6):3705-20. 
doi: 10.1152/jn.00036.2010. Epub 2010 Sep 8. https://pubmed.ncbi.nlm.nih.gov/20884767/

Main function: :func:`~aopy.analysis.accllr.calc_accllr_st`

.. automodule:: aopy.analysis.latency
    :members:

Connectivity
^^^^^^^^^^^^

.. automodule:: aopy.analysis.connectivity
    :members:
