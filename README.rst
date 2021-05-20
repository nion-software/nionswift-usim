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

.. end-badges

The STEM microscope simulator can simulate a HAADF detector within a beam scan, a Ronchigram with aberrations, and a 2D EELS detector that can be summed in the zero dimension to produce a 1D EELS spectrum.

The EELS detector produces EELS spectra unique to each beam position within the simulated sample. The Ronchigram is not dependent on beam position.

The simulated sample can be basic elements/vacuum arranged in rectangular regions; or an amorphous sample.

The scan and Ronchigram or EELS detector can be combined to do spectrum imaging.

The simulator has been used to develop many acquisition algorithms and even some tuning and alignment algorithms for use on the Nion STEM instrument.

Tutorial
--------
Starting with a new or existing project, create a new workspace (**Workspace > New Workspace...**) or clear the existing one. Then split the workspace into a 2x2 layout (**Workspace > Display Panel Split > Split 2x2**).

In the top left panel, right click and choose **uSim Scan (HAADF)**. Press **Scan** at the lower left of that panel. Then press **Stop**.

Open the scan control panel if it is not already open (**Window > uSim Scan Scan Control**). In the scan control panel, click **Positioned**. This will put a probe marker on the scan image. This will only appear when the scan is stopped. You can drag it around.

In the top right panel, right click and choose **uSim EELS Camera**. This shows the 2D EELS detector. Make it display the 1D EELS by clicking the checkbox at the bottom of the panel. Then pres **Play** at the lower left of that panel.

Now drag the probe marker on the scan and the EELS data will change according to the probe position. Press **Pause** on the EELS data when finished.

Open the spectrum imaging panel if it is not already ope (**Window > Spectrum Imaging / 4d Scan Acquisition**). Choose **uSim EELS Camera** at the top left of that panel and choose **Spectra** as the acquisition type. Change the scan width to 24 and the camera exposure time to 50ms. Then click **Acquire**. This will acquire a spectrum image of the scan area and the EELS. You should see the scan from top to bottom and the resulting spectrum image. At the end, you will see the captured HAADF image.

Once the acquisition is finished, right click on the HAADF image which should be in the bottom right panel. Choose **Delete Data Item "Spectrum Image (HAADF)"**. We will not be using that data in this tutorial.

Click on the panel with the spectrum image named "Spectrum Image (uSim EELS Camera)". Then press "p". This will attach a pick region to the spectrum image and display the resulting sum of the spectra within the pick region in a new line plot display. You can drag the display interval to change the energy range displayed on the spectrum image. You can also zoom and otherwise examine the data in the EELS spectra.

Next right click on the spectrum image and choose **Delete Data Item "Spectrum Image (uSim EELS Camera)"**. This will delete the spectrum image and the associated pick line plot.

Now right click on an empty display panel and choose **uSim Ronchigram Camera**. Press **Play**.

Next open the simulator control panel **Window > Simulator Control**. In the simulator panel, you can change various parameters that will affect the Ronchigram. Try changing **C23 X** to 500 nm.

TODO:

    - 4D STEM (Spectrum Imaging using Ronchigram and Images)
    - Change sample type (top of Simulator Control)
    - Change EELS dispersion (set eV/ch to 10 in Simulator Control)
    - Find an edge in EELS data (look in 1200 - 1300 eV range)
    - Background removal (if EELS analysis installed)
    - Scan and subscan, rotation
    - Spectrum imaging with drift correction
    - Multiple shift EELS acquire
    - Multi Acquire
    - Multi Acquire SI
    - 4D STEM with masking
    - Line scan

More Information
----------------

- `Changelog <https://github.com/nion-software/nionswift-usim/blob/master/CHANGES.rst>`_
