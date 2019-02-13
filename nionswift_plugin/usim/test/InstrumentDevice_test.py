import contextlib
import math
import numpy
import unittest

from nion.data import Calibration
from nion.data import xdata_1_0 as xd
from nion.utils import Geometry

from nion.instrumentation import scan_base
from nionswift_plugin.usim import EELSCameraSimulator
from nionswift_plugin.usim import InstrumentDevice
from nionswift_plugin.usim import RonchigramCameraSimulator


def measure_thickness(d: numpy.ndarray) -> float:
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    mx_pos = numpy.argmax(d)
    mx = d[mx_pos]
    mx_tenth = mx/10
    left_pos = mx_pos - sum(d[:mx_pos] > mx_tenth)
    right_pos = mx_pos + (mx_pos - left_pos)
    s = sum(d[left_pos:right_pos])
    return math.log(sum(d) / s)


class MeasureThickness:
    """Carry out the Thickness measurement and add an interval graphic."""

    def __init__(self, computation, **kwargs):
        """Initialize the computation."""
        self.computation = computation

    def execute(self, src):
        """Execute the computation.

        This method will run in a thread and should not make any modifications to the library.
        """
        self.__data = src.display_xdata.data
        self.__left, self.__right, s = sum_zlp(self.__data)
        self.__thickness  = math.log(sum(self.__data) / s)
        self.__src = src


class TestInstrumentDevice(unittest.TestCase):

    def test_defocus_is_observable(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")

        defocus_m = instrument.defocus_m
        property_changed_count = 0

        def property_changed(key: str) -> None:
            nonlocal defocus_m, property_changed_count
            if key == "defocus_m":
                defocus_m = instrument.defocus_m
                property_changed_count += 1
            if key == "C10":
                property_changed_count += 1

        with contextlib.closing(instrument.property_changed_event.listen(property_changed)):
            # setting defocus should set c10 which should trigger its dependent defocus_m
            instrument.defocus_m += 20 / 1E9
            self.assertAlmostEqual(520 / 1E9, defocus_m)
            # setting c10 should trigger its dependent defocus_m
            instrument.SetVal("C10", 600 / 1E9)
            self.assertAlmostEqual(600 / 1E9, defocus_m)
            # a total of 4 changes should happen; no more, no less
            self.assertEqual(4, property_changed_count)

    def test_ronchigram_handles_dependencies_properly(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        camera = RonchigramCameraSimulator.RonchigramCameraSimulator(instrument, Geometry.IntSize(128, 128), 10, 0.030)
        camera._needs_recalculation = False
        instrument.defocus_m += 10 / 1E9
        self.assertTrue(camera._needs_recalculation)
        camera._needs_recalculation = False
        instrument.C30 += 1E9
        self.assertTrue(camera._needs_recalculation)
        camera._needs_recalculation = False
        instrument.ZLPoffset += 1
        self.assertFalse(camera._needs_recalculation)
        camera._needs_recalculation = False
        instrument.beam_current += 1
        self.assertTrue(camera._needs_recalculation)

    def test_powerlaw(self):
        offset_eV = 100
        onset_eV = 30
        data1 = numpy.zeros((1000, ))
        EELSCameraSimulator.plot_powerlaw(data1, 1.0E6, Calibration.Calibration(), offset_eV, onset_eV)
        data2 = numpy.zeros((1000, ))
        EELSCameraSimulator.plot_powerlaw(data2, 1.0E6, Calibration.Calibration(offset=10), offset_eV, onset_eV)
        data3 = numpy.zeros((1000, ))
        EELSCameraSimulator.plot_powerlaw(data3, 1.0E6, Calibration.Calibration(offset=100), offset_eV, onset_eV)
        self.assertEqual(int(data1[500]), int(data2[500 - 10]))
        self.assertEqual(int(data1[500]), int(data3[500 - 100]))

    def test_norm(self):
        offset_eV = 100
        onset_eV = 30
        data1 = numpy.zeros((1000, ))
        EELSCameraSimulator.plot_norm(data1, 1.0E6, Calibration.Calibration(), offset_eV, onset_eV)
        data2 = numpy.zeros((1000, ))
        EELSCameraSimulator.plot_norm(data2, 1.0E6, Calibration.Calibration(offset=10), offset_eV, onset_eV)
        data3 = numpy.zeros((1000, ))
        EELSCameraSimulator.plot_norm(data3, 1.0E6, Calibration.Calibration(offset=100), offset_eV, onset_eV)
        self.assertEqual(int(data1[500]), int(data2[500 - 10]))
        self.assertEqual(int(data1[500]), int(data3[500 - 100]))

    def test_eels_data_is_consistent_when_energy_offset_changes(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        instrument.get_scan_data(scan_base.ScanFrameParameters({"size": (256, 256), "pixel_time_us": 1, "fov_nm": 10}), 0)
        instrument.validate_probe_position()
        camera = instrument._get_camera_simulator("eels")
        camera_size = camera._camera_shape
        camera.noise.enabled = False
        readout_area = Geometry.IntRect(origin=Geometry.IntPoint(), size=camera_size)
        binning_shape = Geometry.IntSize(1, 1)
        # get the value at 200eV and ZLP offset of 0
        instrument.ZLPoffset = 0
        d = xd.sum(instrument.get_camera_data("eels", readout_area, binning_shape, 0.01), axis=0)
        index200_0 = int(d.dimensional_calibrations[-1].convert_from_calibrated_value(200))
        value200_0 = d.data[index200_0]
        # get the value at 200eV and ZLP offset of 100
        instrument.ZLPoffset = 100
        d = xd.sum(instrument.get_camera_data("eels", readout_area, binning_shape, 0.01), axis=0)
        index200_100 = int(d.dimensional_calibrations[-1].convert_from_calibrated_value(200))
        value200_100 = d.data[index200_100]
        self.assertEqual(int(value200_0 / 100), int(value200_100 / 100))
        # print(f"{index200_0} {index200_100}")
        # print(f"{value200_0} {value200_100}")

    def test_eels_data_is_consistent_when_energy_offset_changes_with_negative_zlp_offset(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        instrument.get_scan_data(scan_base.ScanFrameParameters({"size": (256, 256), "pixel_time_us": 1, "fov_nm": 10}), 0)
        instrument.validate_probe_position()
        camera = instrument._get_camera_simulator("eels")
        camera_size = camera._camera_shape
        camera.noise.enabled = False
        readout_area = Geometry.IntRect(origin=Geometry.IntPoint(), size=camera_size)
        binning_shape = Geometry.IntSize(1, 1)
        # get the value at 200eV and ZLP offset of 0
        instrument.ZLPoffset = -20
        instrument.get_camera_data("eels", readout_area, binning_shape, 0.01)

    def test_eels_data_thickness_is_consistent(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        instrument.get_scan_data(scan_base.ScanFrameParameters({"size": (256, 256), "pixel_time_us": 1, "fov_nm": 10}), 0)
        instrument.validate_probe_position()
        camera = instrument._get_camera_simulator("eels")
        camera_size = camera._camera_shape
        camera.noise.enabled = False
        readout_area = Geometry.IntRect(origin=Geometry.IntPoint(), size=camera_size)
        binning_shape = Geometry.IntSize(1, 1)
        # get the value at 200eV and ZLP offset of 0
        instrument.ZLPoffset = -20
        d = xd.sum(instrument.get_camera_data("eels", readout_area, binning_shape, 0.01), axis=0).data
        # confirm it is a reasonable value
        # print(measure_thickness(d))
        self.assertTrue(0.40 < measure_thickness(d) < 1.00)

    def test_eels_data_camera_current_is_consistent(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        instrument.get_scan_data(scan_base.ScanFrameParameters({"size": (256, 256), "pixel_time_us": 1, "fov_nm": 10}), 0)
        instrument.validate_probe_position()
        camera = instrument._get_camera_simulator("eels")
        camera_size = camera._camera_shape
        camera.noise.enabled = False
        readout_area = Geometry.IntRect(origin=Geometry.IntPoint(), size=camera_size)
        binning_shape = Geometry.IntSize(1, 1)
        # get the value at 200eV and ZLP offset of 0
        instrument.ZLPoffset = -20
        exposure_s = 0.01
        d = xd.sum(instrument.get_camera_data("eels", readout_area, binning_shape, exposure_s), axis=0).data
        # confirm it is a reasonable value
        camera_current_pA = numpy.sum(d) / exposure_s / instrument.counts_per_electron / 6.242e18 * 1e12
        # print(f"current {camera_current_pA :#.2f}pA")
        self.assertTrue(190 < camera_current_pA < 210)
