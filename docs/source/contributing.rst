Contributing:
-------------

.. contents:: :local:

The aim of this repository is to maintain high-quality, well-written,
and well-tested code for neural data analysis. To this end, all code
must follow the guidelines laid out in the follwoing sections of this document.

For specific instructions on how to use github, see the `Github SOP <https://docs.google.com/document/d/1JnOoaIXGPXUTmZs_vxThIUNwUKulNUd7_pVAEuDVr_Y/edit?usp=sharing>`_ 

If you are planning to make changes, clone the repo from github, 
then install, then use the -e flag to install in editable mode 
rather than installing a fixed version.

::

    > git clone https://github.com/aolabNeuro/analyze.git
    > cd analyze
    > pip install -e .

If you previously installed `aolab-aopy`
from pip, you can uninstall it with:

::

   > pip uninstall aolab-aopy

.. toctree::

   code-guidelines
   documentation-guidelines
   test-guidelines
   github
   