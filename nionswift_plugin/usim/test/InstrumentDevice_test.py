import contextlib
import math
import numpy
import numpy.typing
import typing
import unittest

from nion.data import Calibration
from nion.data import xdata_1_0 as xd
from nion.utils import Geometry

from nion.instrumentation import scan_base
from nionswift_plugin.usim import EELSCameraSimulator
from nionswift_plugin.usim import InstrumentDevice
from nionswift_plugin.usim import RonchigramCameraSimulator
from nionswift_plugin.usim import ScanDevice
from nionswift_plugin.usim import CameraSimulator

_NDArray = numpy.typing.NDArray[typing.Any]


def measure_thickness(d: _NDArray) -> float:
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    mx_pos = int(numpy.argmax(d))
    mx = d[mx_pos]
    mx_tenth = mx/10
    left_pos = mx_pos - sum(d[:mx_pos] > mx_tenth)
    right_pos = mx_pos + (mx_pos - left_pos)
    s = sum(d[left_pos:right_pos])
    return math.log(sum(d) / s)

def create_camera_and_scan_simulator(instrument: InstrumentDevice.Instrument, camera_type: str) -> typing.Tuple[CameraSimulator.CameraSimulator, ScanDevice.Device]:
    camera_simulator: CameraSimulator.CameraSimulator
    if camera_type == "eels":
        camera_simulator = EELSCameraSimulator.EELSCameraSimulator(instrument, Geometry.IntSize.make(instrument.camera_sensor_dimensions("eels")), instrument.counts_per_electron)
    else:
        camera_simulator = RonchigramCameraSimulator.RonchigramCameraSimulator(instrument, Geometry.IntSize.make(instrument.camera_sensor_dimensions("ronchigram")), instrument.counts_per_electron, instrument.stage_size_nm)

    scan_device = ScanDevice.Device(instrument)

    return camera_simulator, scan_device


class TestInstrumentDevice(unittest.TestCase):

    def test_ronchigram_handles_dependencies_properly(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        camera = RonchigramCameraSimulator.RonchigramCameraSimulator(instrument, Geometry.IntSize(128, 128), 10, 0.030)
        camera._needs_recalculation = False
        instrument.defocus_m += 10 / 1E9
        self.assertTrue(camera._needs_recalculation)
        camera._needs_recalculation = False
        instrument.SetValDelta("C30", 1E9)
        self.assertTrue(camera._needs_recalculation)
        camera._needs_recalculation = False
        instrument.SetValDelta("ZLPoffset", 1)
        self.assertFalse(camera._needs_recalculation)
        camera._needs_recalculation = False
        instrument.SetValDelta("BeamCurrent", 1)
        self.assertTrue(camera._needs_recalculation)

    def test_powerlaw(self) -> None:
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

    def test_norm(self) -> None:
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

    def test_eels_data_is_consistent_when_energy_offset_changes(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        camera_simulator, scan_simulator = create_camera_and_scan_simulator(instrument, "eels")
        scan_simulator.get_scan_data(scan_base.ScanFrameParameters({"size": (256, 256), "pixel_time_us": 1, "fov_nm": 10}), 0)
        instrument.validate_probe_position()
        camera_size = camera_simulator._camera_shape
        typing.cast(EELSCameraSimulator.EELSCameraSimulator, camera_simulator).noise.enabled = False
        readout_area = Geometry.IntRect(origin=Geometry.IntPoint(), size=camera_size)
        binning_shape = Geometry.IntSize(1, 1)
        # get the value at 200eV and ZLP offset of 0
        instrument.SetVal("ZLPoffset", 0)
        d = xd.sum(camera_simulator.get_frame_data(readout_area, binning_shape, 0.01, instrument.scan_context, instrument.probe_position), axis=0)
        index200_0 = int(d.dimensional_calibrations[-1].convert_from_calibrated_value(200))
        value200_0 = d._data_ex[index200_0]
        # get the value at 200eV and ZLP offset of 100
        instrument.SetVal("ZLPoffset", 100)
        d = xd.sum(camera_simulator.get_frame_data(readout_area, binning_shape, 0.01, instrument.scan_context, instrument.probe_position), axis=0)
        index200_100 = int(d.dimensional_calibrations[-1].convert_from_calibrated_value(200))
        value200_100 = d._data_ex[index200_100]
        self.assertEqual(int(value200_0 / 100), int(value200_100 / 100))
        # print(f"{index200_0} {index200_100}")
        # print(f"{value200_0} {value200_100}")

    def test_eels_data_is_consistent_when_energy_offset_changes_with_negative_zlp_offset(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        camera_simulator, scan_simulator = create_camera_and_scan_simulator(instrument, "eels")
        scan_simulator.get_scan_data(scan_base.ScanFrameParameters({"size": (256, 256), "pixel_time_us": 1, "fov_nm": 10}), 0)
        instrument.validate_probe_position()
        camera_size = camera_simulator._camera_shape
        typing.cast(EELSCameraSimulator.EELSCameraSimulator, camera_simulator).noise.enabled = False
        readout_area = Geometry.IntRect(origin=Geometry.IntPoint(), size=camera_size)
        binning_shape = Geometry.IntSize(1, 1)
        # get the value at 200eV and ZLP offset of 0
        instrument.SetVal("ZLPoffset", -20)
        camera_simulator.get_frame_data(readout_area, binning_shape, 0.01, instrument.scan_context, instrument.probe_position)

    def test_eels_data_thickness_is_consistent(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        camera_simulator, scan_simulator = create_camera_and_scan_simulator(instrument, "eels")
        # use the flake sample
        instrument.sample_index = 0
        # set up the scan context; these are here temporarily until the scan context architecture is fully implemented
        instrument._update_scan_context(Geometry.IntSize(256, 256), Geometry.FloatPoint(), 10, 0.0)
        instrument._set_scan_context_probe_position(instrument.scan_context, Geometry.FloatPoint(0.5, 0.5))
        # grab scan data
        scan_simulator.get_scan_data(scan_base.ScanFrameParameters({"size": (256, 256), "pixel_time_us": 1, "fov_nm": 10}), 0)
        instrument.validate_probe_position()
        camera_size = camera_simulator._camera_shape
        typing.cast(EELSCameraSimulator.EELSCameraSimulator, camera_simulator).noise.enabled = False
        readout_area = Geometry.IntRect(origin=Geometry.IntPoint(), size=camera_size)
        binning_shape = Geometry.IntSize(1, 1)
        # get the value at 200eV and ZLP offset of 0
        instrument.SetVal("ZLPoffset", -20)
        d = xd.sum(camera_simulator.get_frame_data(readout_area, binning_shape, 0.01, instrument.scan_context, instrument.probe_position), axis=0)._data_ex
        # confirm it is a reasonable value
        # print(measure_thickness(d))
        self.assertTrue(0.40 < measure_thickness(d) < 1.00)

    def test_eels_data_camera_current_is_consistent(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        camera_simulator, scan_simulator = create_camera_and_scan_simulator(instrument, "eels")
        # set up the scan context; these are here temporarily until the scan context architecture is fully implemented
        instrument._update_scan_context(Geometry.IntSize(256, 256), Geometry.FloatPoint(), 10, 0.0)
        instrument._set_scan_context_probe_position(instrument.scan_context, Geometry.FloatPoint(0.5, 0.5))
        # grab scan data
        scan_simulator.get_scan_data(scan_base.ScanFrameParameters({"size": (256, 256), "pixel_time_us": 1, "fov_nm": 10}), 0)
        instrument.validate_probe_position()
        camera_size = camera_simulator._camera_shape
        typing.cast(EELSCameraSimulator.EELSCameraSimulator, camera_simulator).noise.enabled = False
        readout_area = Geometry.IntRect(origin=Geometry.IntPoint(), size=camera_size)
        binning_shape = Geometry.IntSize(1, 1)
        # get the value at 200eV and ZLP offset of 0
        instrument.SetVal("ZLPoffset", -20)
        exposure_s = 0.01
        d = xd.sum(camera_simulator.get_frame_data(readout_area, binning_shape, 0.01, instrument.scan_context, instrument.probe_position), axis=0)._data_ex
        # confirm it is a reasonable value
        camera_current_pA = numpy.sum(d) / exposure_s / instrument.counts_per_electron / 6.242e18 * 1e12
        # print(f"current {camera_current_pA :#.2f}pA")
        self.assertTrue(190 < camera_current_pA < 210)

    def test_can_get_and_set_val_of_built_in_control(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success = instrument.SetVal("C10", -1e-5)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("C10")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -1e-5)

    def test_can_get_and_set_val_of_built_in_2d_control(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success = instrument.SetVal("C12.x", -1e-5)
        self.assertTrue(success)
        success = instrument.SetVal("C12.y", -2e-5)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("C12.x")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -1e-5)
        success, value = instrument.TryGetVal("C12.y")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -2e-5)

    def test_inform_control_of_built_in_control_works(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success = instrument.SetVal("C10Control", -1e-5)
        self.assertTrue(success)
        success = instrument.InformControl("C10", 1e-4)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("C10")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, 1e-4)
        success, value = instrument.TryGetVal("C10Control")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -1e-5)

    def test_inform_control_of_built_in_2d_control_works(self) -> None:
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
        assert value is not None
        self.assertAlmostEqual(value, 1e-4)
        success, value = instrument.TryGetVal("C12.y")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, 3e-4)
        success, value = instrument.TryGetVal("C12Control.x")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -1e-5)
        success, value = instrument.TryGetVal("C12Control.y")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -3e-5)

    def test_changing_control_triggers_property_changed_event(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        event_name = ""
        def listen(name: str) -> None:
            nonlocal event_name
            if name == "C10":
                event_name = name
        with contextlib.closing(instrument.property_changed_event.listen(listen)):
            instrument.SetVal("C10", 1e-3)
        self.assertEqual(event_name, "C10")

    def test_setting_and_getting_attribute_values_and_2D_values_works(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        C12 = Geometry.FloatPoint(x=-1e-5, y=-3e-5)
        instrument.SetVal("C12.x", C12.x)
        instrument.SetVal("C12.y", C12.y)
        self.assertAlmostEqual(C12.x, instrument.GetVal("C12.x"))
        self.assertAlmostEqual(C12.x, instrument.GetVal2D("C12").x)
        self.assertAlmostEqual(C12.y, instrument.GetVal("C12.y"))
        self.assertAlmostEqual(C12.y, instrument.GetVal2D("C12").y)
        C12 = Geometry.FloatPoint(y=-2e5, x=1e5)
        instrument.SetVal2D("C12", C12)
        self.assertAlmostEqual(C12.x, instrument.GetVal("C12.x"))
        self.assertAlmostEqual(C12.x, instrument.GetVal2D("C12").x)
        self.assertAlmostEqual(C12.y, instrument.GetVal("C12.y"))
        self.assertAlmostEqual(C12.y, instrument.GetVal2D("C12").y)

    def test_setting_2d_control_triggers_event(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        event_name = ""
        def listen(name: str) -> None:
            nonlocal event_name
            if name == "C12":
                event_name = name
        with contextlib.closing(instrument.property_changed_event.listen(listen)):
            instrument.SetVal2D("C12", Geometry.FloatPoint(y=-2e5, x=1e5))
        self.assertEqual(event_name, "C12")

    def test_can_get_and_set_val_of_added_control(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_control("custom_control")
        instrument.add_control(control)
        success = instrument.SetVal("custom_control", -1e-5)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("custom_control")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -1e-5)

    def test_can_get_and_set_val_of_added_2d_control(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_2d_control("custom_control", ("x", "y"))
        instrument.add_control(control)
        success = instrument.SetVal("custom_control.x", -1e-5)
        self.assertTrue(success)
        success = instrument.SetVal("custom_control.y", -2e-5)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("custom_control.x")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -1e-5)
        success, value = instrument.TryGetVal("custom_control.y")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -2e-5)

    def test_inform_control_of_added_control_works(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_control("custom_control")
        instrument.add_control(control)
        typing.cast(InstrumentDevice.Control, instrument.get_control("C10Control")).add_input(control, 0.5)
        success = instrument.SetVal("C10Control", -1e-5)
        self.assertTrue(success)
        success = instrument.InformControl("custom_control", 1e-4)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("custom_control")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, 1e-4)
        success, value = instrument.TryGetVal("C10Control")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -1e-5)

    def test_inform_control_of_added_2d_control_works(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_2d_control("custom_control", ("x", "y"))
        instrument.add_control(control)
        typing.cast(InstrumentDevice.Control2D, instrument.get_control("C12Control")).x.add_input(control.x, 0.5)
        typing.cast(InstrumentDevice.Control2D, instrument.get_control("C12Control")).y.add_input(control.y, 0.25)
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
        assert value is not None
        self.assertAlmostEqual(value, 1e-4)
        success, value = instrument.TryGetVal("custom_control.y")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, 3e-4)
        success, value = instrument.TryGetVal("C12Control.x")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -1e-5)
        success, value = instrument.TryGetVal("C12Control.y")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, -3e-5)

    def test_setting_added_control_triggers_event(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_control("custom_control")
        instrument.add_control(control)
        event_name = ""
        def listen(name: str) -> None:
            nonlocal event_name
            if name == "custom_control":
                event_name = name
        with contextlib.closing(instrument.property_changed_event.listen(listen)):
            instrument.SetVal("custom_control", 1e-3)
        self.assertEqual(event_name, "custom_control")

    def test_setting_and_getting_attribute_values_and_2D_values_works_on_custom_2d_control(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_2d_control("custom_control", ("x", "y"))
        instrument.add_control(control)
        value = Geometry.FloatPoint(x=-1e-5, y=-3e-5)
        instrument.SetVal("custom_control.x", value.x)
        instrument.SetVal("custom_control.y", value.y)
        self.assertAlmostEqual(value.x, instrument.GetVal("custom_control.x"))
        self.assertAlmostEqual(value.x, instrument.GetVal2D("custom_control").x)
        self.assertAlmostEqual(value.y, instrument.GetVal("custom_control.y"))
        self.assertAlmostEqual(value.y, instrument.GetVal2D("custom_control").y)
        value = Geometry.FloatPoint(y=-2e5, x=1e5)
        instrument.SetVal2D("custom_control", value)
        self.assertAlmostEqual(value.x, instrument.GetVal("custom_control.x"))
        self.assertAlmostEqual(value.x, instrument.GetVal2D("custom_control").x)
        self.assertAlmostEqual(value.y, instrument.GetVal("custom_control.y"))
        self.assertAlmostEqual(value.y, instrument.GetVal2D("custom_control").y)

    def test_setting_value_on_custom_2d_control_triggers_property_changed_event(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_2d_control("custom_control", ("x", "y"))
        instrument.add_control(control)
        event_name = ""
        def listen(name: str) -> None:
            nonlocal event_name
            if name == "custom_control":
                event_name = name
        with contextlib.closing(instrument.property_changed_event.listen(listen)):
            instrument.SetVal2D("custom_control", Geometry.FloatPoint(y=-2e5, x=1e5))
        self.assertEqual(event_name, "custom_control")

    def test_setting_attribute_value_on_custom_2d_control_triggers_property_changed_event(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_2d_control("custom_control", ("x", "y"))
        instrument.add_control(control)
        event_name = ""
        def listen(name: str) -> None:
            nonlocal event_name
            if name == "custom_control":
                event_name = name
        with contextlib.closing(instrument.property_changed_event.listen(listen)):
            instrument.SetVal("custom_control.x", 2e5)
        self.assertEqual(event_name, "custom_control")

    def test_adding_control_with_duplicate_name_raises_value_error(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        control = instrument.create_control("C10")
        with self.assertRaises(ValueError):
            instrument.add_control(control)

    def test_accessing_control_in_non_native_axis_works(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success = instrument.SetVal("C12.x", 1e-5)
        self.assertTrue(success)
        success = instrument.SetVal("C12.y", 2e-5)
        self.assertTrue(success)
        success, value = instrument.TryGetVal("C12.px")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, 1e-5)
        success, value = instrument.TryGetVal("C12.py")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, 2e-5)
        self.assertTrue(isinstance(typing.cast(InstrumentDevice.Control2D, instrument.get_control("C12")).px, InstrumentDevice.ConvertedControl))

    def test_accessing_non_exisiting_axis_fails(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success, value = instrument.TryGetVal("C12.ne")
        self.assertFalse(success)
        success = instrument.SetVal("C12.ne", 1e-5)
        self.assertFalse(success)
        with self.assertRaises(AttributeError):
            getattr(instrument.get_control("C12"), "ne")

    def test_get_drive_strength_with_arrow_syntax(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success, value = instrument.TryGetVal("CApertureOffset.x->CAperture.x")
        self.assertTrue(success)
        assert value is not None
        self.assertAlmostEqual(value, 1.0)

    def test_set_drive_strength_with_arrow_syntax(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success = instrument.SetVal("CApertureOffset.x->CAperture.x", 0.5)
        self.assertTrue(success)
        success = instrument.SetVal("CApertureOffset.y->CAperture.y", 0.2)
        self.assertTrue(success)
        value = instrument.GetVal("CApertureOffset.x->CAperture.x")
        self.assertAlmostEqual(value, 0.5)
        value = instrument.GetVal("CApertureOffset.y->CAperture.y")
        self.assertAlmostEqual(value, 0.2)

    def test_get_drive_strength_fails_for_non_existing_drive(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        success, value = instrument.TryGetVal("CApertureOffset.y->CAperture.x")
        self.assertFalse(success)

    def test_use_control_as_input_weight_works(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        weight_control = instrument.create_control("weight_control")
        instrument.add_control(weight_control)
        input_control = instrument.create_control("input_control")
        instrument.add_control(input_control)
        test_control = instrument.create_control("test_control", weighted_inputs=[(input_control, weight_control)])
        instrument.add_control(test_control)
        self.assertAlmostEqual(instrument.GetVal("test_control"), 0)
        self.assertTrue(instrument.SetVal("input_control", 1.0))
        self.assertAlmostEqual(instrument.GetVal("test_control"), 0)
        self.assertTrue(instrument.SetVal("weight_control", 1.0))
        self.assertAlmostEqual(instrument.GetVal("test_control"), 1.0)

    def test_expression(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        weight_control = instrument.create_control("weight_control")
        instrument.add_control(weight_control)
        input_control = instrument.create_control("input_control")
        instrument.add_control(input_control)
        other_control = instrument.create_control("other_control")
        instrument.add_control(other_control)
        test_control = instrument.create_control("test_control")
        instrument.add_control(test_control)
        test_control.set_expression("input_control*weight_control/2 + x",
                                    variables={"input_control": "input_control",
                                               "weight_control": weight_control,
                                               "x": "other_control"},
                                    instrument=instrument)
        self.assertAlmostEqual(instrument.GetVal("test_control"), 0)
        self.assertTrue(instrument.SetVal("input_control", 1.0))
        self.assertAlmostEqual(instrument.GetVal("test_control"), 0)
        self.assertTrue(instrument.SetVal("weight_control", 2.0))
        self.assertAlmostEqual(instrument.GetVal("test_control"), 1.0)

    def test_using_control_in_its_own_expression_raises_value_error(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        test_control = instrument.create_control("test_control")
        instrument.add_control(test_control)
        test_control.add_dependent(test_control)
        with self.assertRaises(ValueError):
            test_control.set_expression('test_control', variables={'test_control': test_control})

    def test_add_input_for_existing_control(self) -> None:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        test_control = instrument.create_control("test_control")
        other_control = instrument.create_control("other_control")
        weight_control = instrument.create_control("weight_control")
        instrument.add_control(test_control)
        instrument.add_control(other_control)
        instrument.add_control(weight_control)
        instrument.add_control_inputs('test_control', [(other_control, weight_control)])
        # Add it a second time to test add existing control
        instrument.add_control_inputs('test_control', [(other_control, weight_control)])


if __name__ == '__main__':
    unittest.main()
