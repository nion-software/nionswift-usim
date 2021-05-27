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

Introduction
------------
Microscopes come in many varieties such as light/optical microscopes, electron microscopes, scanning probe microscopes. Within each of those broad categories are many sub-categories such as fluorescence and super resolution optical microscopes, and transmission or scanning electron microscopes.

This simulator is intended to provide a platform and specific simulations for a variety of microscopes.

Currently, however, it is primarily focused on scanning transmission electron microscopy (STEM). A STEM instrument produces electrons which are transmitted through a sample, scattering, deflecting, or changing energy in the process, and then digitized in a detector.

A STEM instrument will include an electron source, electromagnetic lenses to form and control the electron beam, a scanning system to control the lenses, and multiple cameras to detect the resulting electrons. A typical configuration of detectors would include multiple annular dark field detectors, high-angle HAADF, medium-angle MAADF; a Ronchigram detector (an image of the convergent beam diffraction pattern, useful and important in aberration corrected instruments); often an EELS detector (electron energy loss spectrum) to measure energy loss; and video cameras for alignment and adjustment. Other detectors such as optical or x-ray detectors are also sometimes present.

In transmission electron microscopy, as an electron travels through a sample, it may interact with electrons or nuclei, scattering in the process. An ideal sample for transmission electron microscopy will be thin, to limit the number of scattering events. As it scatters, it may transfer some of its energy to the sample (inelastic) or simply be deflected (elastic). When it transfers energy to the sample (inelastic), the amount of energy that the electron loses will be characteristic (on average) of the atom or compound with which it interacts. This may show up in EELS data as phonons (interaction with nuclei), plasmons (interaction with surfaces), or edges (interactions with electrons of specific energies).

.. describe applications

Features
--------
The STEM microscope simulator can simulate a HAADF detector within a beam scan, a Ronchigram with lens aberrations, and a 2D EELS detector that can be summed in the zero dimension to produce a 1D EELS spectrum.

The EELS detector produces EELS spectra unique to each beam position within the simulated sample. The Ronchigram is not dependent on beam position.

The simulated sample can be basic elements/vacuum arranged in rectangular regions; or an amorphous sample.

The scan and Ronchigram or EELS detector can be combined to do spectrum imaging.

The simulator has been used to develop many acquisition algorithms and even some tuning and alignment algorithms for use on the Nion STEM instrument.

Acquisition Modes
-----------------
There are several regularly used modes of acquisition on a STEM instrument.

To begin, an imaging mode formed by continuously scanning the beam in a rectangular pattern with a HAADF detector is often the mode used to find the beam and make rough adjustments. At each point of the scan, a single value is read from the HAADF detector and those values are assembled into an image. The scan size can be adjusted by changing the field of view (FOV), expressed in nm; and also rotated to align with a sample. The pixel time can also be adjusted to account for a weaker or stronger signal, with the tradeoff being the speed of imaging.

Once the beam is centered, the Ronchigram imaging mode is used to focus and correct aberrations in the electromagnetic lenses. The magnification on the Ronchigram can be adjusted using the defocus value, expressed in nm. When the beam has no defocus (0nm), it would appear as an image with little or no contrast, depending on the sample and other alignments. When the beam has a slight defocus (500nm), it forms an image of the sample and can be used for imaging. However, lens aberrations may contribute distortions to the resulting image. This simulator provides simulator aberrations up to 5th order. The default values produce a perfectly aligned instrument; but explicitly introducing aberrations can be useful for learning about aberrations or testing alignment/tuning software.

In a STEM instrument with a spectrometer, an electron energy loss spectrum (EELS) can also be observed. The spectrometer is a special set of electromagnetic lenses that bend the electrons in such a way that the energy of the electron is dispersed in a spatial axis, forming an image where the vertical axis is a non-dimensioned spread of the beam and the horizontal axis is an energy loss for the electron. The energy loss will be characteristic (statistically) of the atom or compound through which the electron travelled.

The simulator provides a 2D EELS spectrum that is calculated based on the beam position (also known as the probe position) on the sample. The 2D EELS spectrum can be summed along the vertical axis to produce a 1D EELS spectrum that can be displayed as a line plot. The probe position can be adjusted by starting a scan and then stopping it, enabling the beam position graphic, and adjusting it.

The scan and EELS spectrum can be combined into a spectrum image. The beam is scanned and at each beam position, an EELS spectrum is acquired. The results are assembled in a data item 2D x 1D where the first two dimensions represent the scan position and the last dimension represents the electron energy loss at that position. Since each EELS spectrum is dependent on the sample at the corresponding beam position, the spectrum image can be used to do things like elemental mapping, where the concentration of one or more elements are formed into images where the intensity represents their relative concentration within the sample.

- 4D STEM: Scanning with Ronchigram detector, combined into 2D x 2D data item
- 4D STEM virtual detectors: Scanning with Ronchigram detector, summed within areas to form one or more 2D images
- Spectrum imaging, short exposure, aligned and summed

.. describe where each of these modes would be used

Samples
-------
The simulator provides two types of samples: a crystal and an amorphous sample.

.. describe each sample and why it is used

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
