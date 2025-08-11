Changelog (nionswift-usim)
==========================

5.4.2 (2025-08-11)
------------------
- Fix issue when running as offline installer plug-in.

5.4.1 (2025-05-29)
------------------
- Maintenance release.

5.4.0 (2025-04-23)
------------------
- Maintenance release.

5.3.0 (2025-01-06)
------------------
- Package restructuring to make it easier to use the simulator in other packages.

5.2.0 (2024-10-27)
------------------
- Use Gaussian noise in place of Poisson noise beyond threshold to improve performance.
- Require Numpy 2. Add Python 3.13 support. Drop Python 3.9, 3.10 support.

5.1.2 (2024-06-14)
------------------
- Fix regression where EELS data was disconnected from probe position.
- Record flyback pixels in metadata.

5.1.1 (2023-10-23)
------------------
- Minor updates for Python 3.12 and typing compatibility.

5.1.0 (2023-09-14)
------------------
- Updates for nionswift-instrumentation 22.0 (sequence buffer).
- Fix issue with data not being updated during synchronized scan.
- Add Python 3.11 support. Drop 3.8.

0.5.0 (2022-12-07)
------------------
- Updates for nionswift-instrumentation 0.21.
- Register scan module (scan device and settings) rather than scan device directly.

0.4.5 (2022-10-03)
------------------
- Add reference setting index and capability to remove controls.

0.4.4 (2022-09-13)
------------------
- Minor bugs for testing in other packages.

0.4.3 (2022-05-28)
------------------
- Add simulated CTS sample.
- Fix issues in instrument panel.
- Implement new axis manager.

0.4.2 (2022-02-18)
------------------
- Improve simulated synchronization behavior for future work.
- Fix minor issues with camera length.

0.4.1 (2021-12-13)
------------------
- Change camera device to implement camera 3 (no prepare methods).

0.4.0 (2021-11-21)
------------------
- Updates for nionswift-instrumentation 0.20 and nionswift 0.16.
- Add support for axis descriptors.
- Add support for camera masks.
- Drop support for Python 3.7.

0.3.0 (2020-08-31)
------------------
- Add support for partial synchronized acquisition.
- Fix handling of probe position in sub-scans.
- Add aperture that can be moved and "distorted" (i.e. dipole and quadrupole effect simulation).
- Add functions to 'Instrument' that facilitate adding new inputs to existing controls.
- Allow input weights for controls to be controls in addition to float.
- Add option to attach a python expression as control input (only one expression per control can be set, but it can be arbitrarily complex, as long as it can be evaluated by 'eval').
- Changed meaning of convergence angle to reflect its real meaning (in the simulator it only controls the size of the aperture on the Ronchigram camera, the effect on the scan is not simulated yet).
- Add 'Variable' class to InstrumentDevice. 'Variables' differ from 'Controls' in that they do not have a local value.

0.2.1 (2019-11-27)
------------------
- Minor changes to be compatible with nionswift-instrumentation.
- Improve 'inform' functionality on Ronchigram controls.

0.2.0 (2019-06-27)
------------------
- Fix some simulated aberration calculations.
- Add option for flake sample (same as previous version) or amorphous sample.
- Allow adding new controls to existing instrument instance.
- Add support for 2D controls and AxisManager.

0.1.7 (2019-04-29)
------------------
- Ensure noise gets added as float32 to ensure good display performance.

0.1.6 (2019-02-27)
------------------
- Fix scaling of spectra to be consistent with beam current, sample thickness, and energy offset.
- Improve performance for cameras.
- Add support for ZLP tare control / inform.
- Add controls used in 4D acquisition.
- Change Ronchigram units to radians.
- Improve/fix reliability with camera running faster than scan.

Contributors: @Brow71189 @cmeyer

0.1.5 (2018-10-04)
------------------
- Fix issue with scan content position (introduced with rotated scans).

0.1.4 (2018-10-03)
------------------
- Fix minor issue with EELS data.

0.1.3 (2018-10-03)
------------------
- Update support for API.
- Add support for rotated scans.

0.1.2 (2018-06-25)
------------------
- Specify lower priorities for all simulator devices.
- Add persistence of camera settings.
- Restructure as a camera module to be parallel with physical camera modules.
- Switch to using calibration controls instead of intrinsic calibrations.

0.1.1 (2018-05-13)
------------------
- Initial version online.
