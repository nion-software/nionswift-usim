Changelog (nionswift-usim)
==========================

0.4.1 (UNRELEASED)
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
