Nion Swift STEM Microscope Simulator
====================================

The Nion Swift STEM Microscope Simulator Library (used in Nion Swift)
---------------------------------------------------------------------
A STEM microscope simulator for use with Nion Swift. Used for debugging Nion Swift acquisition and developing acquisition tools, techniques, and apps.

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |linux|
    * - package
      - |version|


.. |linux| image:: https://img.shields.io/travis/nion-software/nionswift-usim/master.svg?label=Linux%20build
   :target: https://travis-ci.org/nion-software/nionswift-usim
   :alt: Travis CI build status (Linux)

.. |version| image:: https://img.shields.io/pypi/v/nionswift-usim.svg
   :target: https://pypi.org/project/nionswift-usim/
   :alt: Latest PyPI version

The STEM microscope simulator can simulate a HAADF detector within a beam scan, a Ronchigram with aberrations, and a 2D EELS detector that can be summed in the zero dimension to produce a 1D EELS spectrum.

The EELS detector produces EELS spectra unique to each beam position within the simulated sample. The Ronchigram is not dependent on beam position.

The simulated sample can be basic elements/vacuum arranged in rectangular regions; or an amorphous sample.

The scan and Ronchigram or EELS detector can be combined to do spectrum imaging.

The simulator has been used to develop many acquisition algorithms and even some tuning and alignment algorithms for use on the Nion STEM instrument.

.. end-badges

More Information
----------------

- `Changelog <https://github.com/nion-software/nionswift-usim/blob/master/CHANGES.rst>`_
