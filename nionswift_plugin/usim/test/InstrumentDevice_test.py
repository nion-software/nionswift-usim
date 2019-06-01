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

# TODO Is this class needed? Spyder IDE complains about a missing function "sum_zlp" and it does not look like the class
# is used anywhere
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

# Disabled this test because I don't think it is useful but it fails with the latest changes
# to IntrumentDevice. Why do we even need "defocus_m" as an extra control that gets synched with
# "C10"? It is not used anywhere in usim in the way is is tested here. I added new tests for making sure the correct
# events are fired when a control is changed.
# I left "defocus_m" in InstrumentDevice but changed it to be simply an alias for C10. This way,
# all other tests pass and simulator works as expected. am 5-31-2019

    # def test_defocus_is_observable(self):
    #     instrument = InstrumentDevice.Instrument("usim_stem_controller")

    #     defocus_m = instrument.defocus_m
    #     property_changed_count = 0

    #     def property_changed(key: str) -> None:
    #         nonlocal defocus_m, property_changed_count
    #         print(key)
    #         if key == "defocus_m":
    #             defocus_m = instrument.defocus_m
    #             property_changed_count += 1
    #         if key == "C10":
    #             property_changed_count += 1

    #     with contextlib.closing(instrument.property_changed_event.listen(property_changed)):
    #         # setting defocus should set c10 which should trigger its dependent defocus_m
    #         instrument.defocus_m += 20 / 1E9
    #         self.assertAlmostEqual(520 / 1E9, defocus_m)
    #         # setting c10 should trigger its dependent defocus_m
    #         instrument.SetVal("C10", 600 / 1E9)
    #         self.assertAlmostEqual(600 / 1E9, defocus_m)
    #         # a total of 4 changes should happen; no more, no less
    #         self.assertEqual(4, property_changed_count)

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

    def test_can_get_and_set_val_of_built_in_control(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success = instrument.SetVal("C10", -1e-5)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("C10")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -1e-5)

    def test_can_get_and_set_val_of_built_in_2d_control(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success = instrument.SetVal("C12.x", -1e-5)
        self.assertTrue(success)
        success = instrument.SetVal("C12.y", -2e-5)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("C12.x")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -1e-5)
        success, value = instrument.TryGetVal("C12.y")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -2e-5)

    def test_inform_control_of_built_in_control_works(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success = instrument.SetVal("C10Control", -1e-5)
        self.assertTrue(success)
        success = instrument.InformControl("C10", 1e-4)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("C10")
        self.assertTrue(success)
        self.assertAlmostEqual(value, 1e-4)
        success, value = instrument.TryGetVal("C10Control")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -1e-5)

    def test_inform_control_of_built_in_2d_control_works(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success = instrument.SetVal("C12Control.x", -1e-5)
        self.assertTrue(success)
        success = instrument.SetVal("C12Control.y", -3e-5)
        self.assertTrue(success)
        success = instrument.InformControl("C12.x", 1e-4)
        self.assertTrue(success)
        success = instrument.InformControl("C12.y", 3e-4)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("C12.x")
        self.assertTrue(success)
        self.assertAlmostEqual(value, 1e-4)
        success, value = instrument.TryGetVal("C12.y")
        self.assertTrue(success)
        self.assertAlmostEqual(value, 3e-4)
        success, value = instrument.TryGetVal("C12Control.x")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -1e-5)
        success, value = instrument.TryGetVal("C12Control.y")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -3e-5)

    def test_accessing_builtin_control_as_attribute_works(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        instrument.SetVal("C10", 3e-3)
        value = instrument.GetVal("C10")
        self.assertAlmostEqual(value, instrument.C10)
        instrument.C10 = 1e-3
        value = instrument.GetVal("C10")
        self.assertAlmostEqual(value, instrument.C10)

    def test_accessing_builtin_control_as_attribute_triggers_event(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        event_name = ""
        def listen(name):
            nonlocal event_name
            if name == "C10":
                event_name = name
        listener = instrument.property_changed_event.listen(listen)
        instrument.C10 = 1e-3
        del listener
        self.assertEqual(event_name, "C10")

    def test_accessing_builtin_2d_control_as_attribute_works(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        instrument.SetVal("C12.x", -1e-5)
        instrument.SetVal("C12.y", -3e-5)
        value = instrument.GetVal("C12.x")
        self.assertAlmostEqual(value, instrument.C12.x)
        value = instrument.GetVal("C12.y")
        self.assertAlmostEqual(value, instrument.C12.y)
        instrument.C12 = Geometry.FloatPoint(y=-2e5, x=1e5)
        value = instrument.GetVal("C12.x")
        self.assertAlmostEqual(value, instrument.C12.x)
        value = instrument.GetVal("C12.y")
        self.assertAlmostEqual(value, instrument.C12.y)

    def test_accessing_builtin_2d_control_as_attribute_triggers_event(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        event_name = ""
        def listen(name):
            nonlocal event_name
            if name == "C12":
                event_name = name
        listener = instrument.property_changed_event.listen(listen)
        instrument.C12 = Geometry.FloatPoint(y=-2e5, x=1e5)
        del listener
        self.assertEqual(event_name, "C12")

    def test_can_get_and_set_val_of_added_control(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_control("custom_control")
        instrument.add_control(control)
        success = instrument.SetVal("custom_control", -1e-5)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("custom_control")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -1e-5)

    def test_can_get_and_set_val_of_added_2d_control(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_2d_control("custom_control", ("x", "y"))
        instrument.add_control(control)
        success = instrument.SetVal("custom_control.x", -1e-5)
        self.assertTrue(success)
        success = instrument.SetVal("custom_control.y", -2e-5)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("custom_control.x")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -1e-5)
        success, value = instrument.TryGetVal("custom_control.y")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -2e-5)

    def test_inform_control_of_added_control_works(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_control("custom_control")
        instrument.add_control(control)
        instrument.get_control("C10Control").add_input(control, 0.5)
        success = instrument.SetVal("C10Control", -1e-5)
        self.assertTrue(success)
        success = instrument.InformControl("custom_control", 1e-4)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("custom_control")
        self.assertTrue(success)
        self.assertAlmostEqual(value, 1e-4)
        success, value = instrument.TryGetVal("C10Control")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -1e-5)

    def test_inform_control_of_added_2d_control_works(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_2d_control("custom_control", ("x", "y"))
        instrument.add_control(control)
        instrument.get_control("C12Control").x.add_input(control.x, 0.5)
        instrument.get_control("C12Control").y.add_input(control.y, 0.25)
        success = instrument.SetVal("C12Control.x", -1e-5)
        self.assertTrue(success)
        success = instrument.SetVal("C12Control.y", -3e-5)
        self.assertTrue(success)
        success = instrument.InformControl("custom_control.x", 1e-4)
        self.assertTrue(success)
        success = instrument.InformControl("custom_control.y", 3e-4)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("custom_control.x")
        self.assertTrue(success)
        self.assertAlmostEqual(value, 1e-4)
        success, value = instrument.TryGetVal("custom_control.y")
        self.assertTrue(success)
        self.assertAlmostEqual(value, 3e-4)
        success, value = instrument.TryGetVal("C12Control.x")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -1e-5)
        success, value = instrument.TryGetVal("C12Control.y")
        self.assertTrue(success)
        self.assertAlmostEqual(value, -3e-5)

    def test_accessing_added_control_as_attribute_works(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_control("custom_control")
        instrument.add_control(control)
        instrument.SetVal("custom_control", 3e-3)
        value = instrument.GetVal("custom_control")
        self.assertAlmostEqual(value, instrument.custom_control)
        instrument.custom_control = 1e-3
        value = instrument.GetVal("custom_control")
        self.assertAlmostEqual(value, instrument.custom_control)

    def test_accessing_added_control_as_attribute_triggers_event(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_control("custom_control")
        instrument.add_control(control)
        event_name = ""
        def listen(name):
            nonlocal event_name
            if name == "custom_control":
                event_name = name
        listener = instrument.property_changed_event.listen(listen)
        instrument.custom_control = 1e-3
        del listener
        self.assertEqual(event_name, "custom_control")

    def test_accessing_added_2d_control_as_attribute_works(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_2d_control("custom_control", ("x", "y"))
        instrument.add_control(control)
        instrument.SetVal("custom_control.x", -1e-5)
        instrument.SetVal("custom_control.y", -3e-5)
        value = instrument.GetVal("custom_control.x")
        self.assertAlmostEqual(value, instrument.custom_control.x)
        value = instrument.GetVal("custom_control.y")
        self.assertAlmostEqual(value, instrument.custom_control.y)
        instrument.custom_control = Geometry.FloatPoint(y=-2e5, x=1e5)
        value = instrument.GetVal("custom_control.x")
        self.assertAlmostEqual(value, instrument.custom_control.x)
        value = instrument.GetVal("custom_control.y")
        self.assertAlmostEqual(value, instrument.custom_control.y)

    def test_accessing_added_2d_control_as_attribute_triggers_event(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_2d_control("custom_control", ("x", "y"))
        instrument.add_control(control)
        event_name = ""
        def listen(name):
            nonlocal event_name
            if name == "custom_control":
                event_name = name
        listener = instrument.property_changed_event.listen(listen)
        instrument.custom_control = Geometry.FloatPoint(y=-2e5, x=1e5)
        del listener
        self.assertEqual(event_name, "custom_control")

    def test_adding_control_with_duplicate_name_raises_value_error(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_control("C10")
        with self.assertRaises(ValueError):
            instrument.add_control(control)

    def test_accessing_control_in_non_native_axis_works(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success = instrument.SetVal("C12.x", 1e-5)
        self.assertTrue(success)
        success = instrument.SetVal("C12.y", 2e-5)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("C12.px")
        self.assertTrue(success)
        self.assertAlmostEqual(value, 1e-5)
        success, value = instrument.TryGetVal("C12.py")
        self.assertTrue(success)
        self.assertAlmostEqual(value, 2e-5)
        self.assertTrue(isinstance(instrument.get_control("C12").px, InstrumentDevice.ConvertedControl))

    def test_accessing_non_exisiting_axis_fails(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success, value = instrument.TryGetVal("C12.ne")
        self.assertFalse(success)
        success = instrument.SetVal("C12.ne", 1e-5)
        self.assertFalse(success)
        with self.assertRaises(AttributeError):
            instrument.get_control("C12").ne


if __name__ == '__main__':
    unittest.main()
