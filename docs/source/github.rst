Using github
============

See the `Github SOP <https://docs.google.com/document/d/1JnOoaIXGPXUTmZs_vxThIUNwUKulNUd7_pVAEuDVr_Y/edit?usp=sharing>`_ 
for detailed instructions. Here are some tips:

Documentation
-------------

The repo is hooked into
`readthedocs <https://analyze.readthedocs.io/en/latest>`__ so that most
changes to the code are automatically reflected in the documentation.
This documentation is generated from the .rst files located in
``docs/source/``. Each .rst file is named according to a module in
``aopy`` and contains a high-level overview of the module as well as the
API reference for all the functions within that module.

Pull requests
-------------

To make contributions, you will have to use a pull request. Follow the
instructions in the `Github
SOP <https://docs.google.com/document/d/1JnOoaIXGPXUTmZs_vxThIUNwUKulNUd7_pVAEuDVr_Y/edit>`__.

When reviewing someone's code, you should checkout their branch on your
local machine and run all the tests using
``python -m unittest discover -s tests``