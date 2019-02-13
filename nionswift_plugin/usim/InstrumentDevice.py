"""
Useful references:
    http://www.rodenburg.org/guide/index.html
    http://www.ammrf.org.au/myscope/
"""

# standard libraries
import math
import numpy
import threading
import typing
import re

from nion.data import Core
from nion.data import DataAndMetadata

from nion.utils import Event
from nion.utils import Geometry

from nion.instrumentation import stem_controller

from . import CameraSimulator
from . import EELSCameraSimulator
from . import SampleSimulator
from . import RonchigramCameraSimulator


"""
from nionswift_plugin.usim import InstrumentDevice
from nion.data import Calibration
import scipy.stats
data = numpy.zeros((1000, ))
InstrumentDevice.plot_powerlaw(data, 1, Calibration.Calibration(), 100, 30)
show(data)
data = numpy.zeros((1000, ))
InstrumentDevice.plot_powerlaw(data, 1, Calibration.Calibration(offset=10), 100, 30)
show(data)

from nionswift_plugin.usim import InstrumentDevice
from nion.data import Calibration
import scipy.stats
data = numpy.zeros((1000, ))
InstrumentDevice.plot_norm(data, 1, Calibration.Calibration(), 100, 30)
show(data)
data = numpy.zeros((1000, ))
InstrumentDevice.plot_norm(data, 1, Calibration.Calibration(offset=10), 100, 30)
show(data)

from nionswift_plugin.usim import InstrumentDevice
from nion.data import Calibration
import scipy.stats
data = numpy.zeros((1000, ))
powerlaw = scipy.stats.powerlaw(4, loc=0, scale=4000)
show(powerlaw.pdf(numpy.linspace(1000, 0, data.shape[0])))
show(powerlaw.pdf(numpy.linspace(900, -100, data.shape[0])))

from nionswift_plugin.usim import InstrumentDevice
from nion.data import Calibration
import scipy.stats
data = numpy.zeros((1000, ))
show(scipy.stats.norm(loc=100, scale=30).cdf(numpy.linspace(0, 1000, data.shape[0])))
show(scipy.stats.norm(loc=100, scale=30).cdf(numpy.linspace(100, 1100, data.shape[0])))


from nionswift_plugin.usim import InstrumentDevice
from nion.data import Calibration
import scipy.stats
data = numpy.zeros((1000, ))
powerlaw = scipy.stats.powerlaw(8, loc=0, scale=4000)
show(powerlaw.pdf(numpy.linspace(4000 - 0, 4000 - 1000, data.shape[0])) * scipy.stats.norm(loc=100, scale=30).cdf(numpy.linspace(0, 1000, data.shape[0])))
show(powerlaw.pdf(numpy.linspace(4000 - 100, 4000 - 1100, data.shape[0])) * scipy.stats.norm(loc=100, scale=30).cdf(numpy.linspace(100, 1100, data.shape[0])))
# show(powerlaw.pdf(numpy.linspace(4000 - 0, 4000 - 1000, data.shape[0])))
# show(powerlaw.pdf(numpy.linspace(4000 - 100, 4000 - 1100, data.shape[0])))
"""


class Control:
    """
    Controls keep an output value equal to the weight sum of input values plus a local value.

    TODO: add optional noise (continuous and periodic)
    TODO: add response time to changes
    TODO: add hysteresis
    """

    def __init__(self, name: str, local_value: float = 0.0, weighted_inputs: typing.Optional[typing.List[typing.Tuple["Control", float]]] = None):
        self.name = name
        self.weighted_inputs = weighted_inputs if weighted_inputs else list()
        self.dependents = list()
        self.local_value = float(local_value)
        for input, _ in self.weighted_inputs:
            input.add_dependent(self)
        self.__last_output_value = None
        self.on_changed = None

    def __str__(self):
        return "{}: {} + {} = {}".format(self.name, self.__weighted_input_value, self.local_value, self.output_value)

    @property
    def __weighted_input_value(self) -> float:
        return sum([weight * input.output_value for input, weight in self.weighted_inputs])

    @property
    def output_value(self) -> float:
        return self.__weighted_input_value + self.local_value

    def add_input(self, input: "Control", weight: float) -> None:
        self.weighted_inputs.append((input, weight))
        self._notify_change()

    def add_dependent(self, dependent: "Control") -> None:
        self.dependents.append(dependent)

    def set_local_value(self, value: float) -> None:
        self.local_value = value
        self._notify_change()

    def set_output_value(self, value: float) -> None:
        self.set_local_value(value - self.__weighted_input_value)

    def inform_output_value(self, value: float) -> None:
        # save old dependent output values so they can stay constant
        old_dependent_outputs = [dependent.output_value for dependent in self.dependents]
        # set the output value
        self.set_output_value(value)
        # update dependent output values to old values
        for dependent, dependent_output in zip(self.dependents, old_dependent_outputs):
            dependent.set_output_value(dependent_output)

    def _notify_change(self) -> None:
        output_value = self.output_value
        if output_value != self.__last_output_value:
            self.__last_output_value = output_value
            if callable(self.on_changed):
                self.on_changed(self)
            for dependent in self.dependents:
                dependent._notify_change()


class Instrument(stem_controller.STEMController):
    """
    TODO: add temporal supersampling for cameras (to produce blurred data when things are changing).
    """

    def __init__(self, instrument_id: str):
        super().__init__()
        self.priority = 20
        self.instrument_id = instrument_id
        self.property_changed_event = Event.Event()
        self.__camera_frame_event = threading.Event()
        self.__sample = SampleSimulator.Sample()

        # define the STEM geometry limits
        self.stage_size_nm = 150
        self.max_defocus = 5000 / 1E9

        self.__stage_position_m = Geometry.FloatPoint()
        self.__beam_shift_m = Geometry.FloatPoint()
        self.__convergence_angle_rad = 30 / 1000
        self.__order_1_max_angle = 0.008
        self.__order_2_max_angle = 0.012
        self.__order_3_max_angle = 0.024
        self.__order_1_patch = 0.006
        self.__order_2_patch = 0.006
        self.__order_3_patch = 0.006
        self.__c1_range = 4e-9
        self.__c2_range = 300e-9
        self.__c3_range = 17e-6
        self.__slit_in = False
        self.__energy_per_channel_eV = 0.5
        self.__voltage = 100000
        self.__beam_current = 200E-12  # 200 pA
        self.__blanked = False
        self.__ronchigram_shape = Geometry.IntSize(2048, 2048)
        self.__eels_shape = Geometry.IntSize(256, 1024)
        self.__last_scan_params = None
        self.__live_probe_position = None
        self.__sequence_progress = 0
        self.__lock = threading.Lock()

        zlp_tare_control = Control("ZLPtare")
        zlp_offset_control = Control("ZLPoffset", -20, [(zlp_tare_control, 1.0)])
        c10 = Control("C10", 500 / 1e9)
        c12_x = Control("C12.x")
        c12_y = Control("C12.y")
        c21_x = Control("C21.x")
        c21_y = Control("C21.y")
        c23_x = Control("C23.x")
        c23_y = Control("C23.y")
        c30 = Control("C30")
        c32_x = Control("C32.x")
        c32_y = Control("C32.y")
        c34_x = Control("C34.x")
        c34_y = Control("C34.y")
        c10Control = Control("C10Control", 0.0, [(c10, 1.0)])
        c12Control_x = Control("C12Control.x", 0.0, [(c12_x, 1.0)])
        c12Control_y = Control("C12Control.y", 0.0, [(c12_y, 1.0)])
        c21Control_x = Control("C21Control.x", 0.0, [(c21_x, 1.0)])
        c21Control_y = Control("C21Control.y", 0.0, [(c21_y, 1.0)])
        c23Control_x = Control("C23Control.x", 0.0, [(c23_x, 1.0)])
        c23Control_y = Control("C23Control.y", 0.0, [(c23_y, 1.0)])
        c30Control = Control("C30Control", 0.0, [(c30, 1.0)])
        c32Control_x = Control("C32Control.x", 0.0, [(c32_x, 1.0)])
        c32Control_y = Control("C32Control.y", 0.0, [(c32_y, 1.0)])
        c34Control_x = Control("C34Control.x", 0.0, [(c34_x, 1.0)])
        c34Control_y = Control("C34Control.y", 0.0, [(c34_y, 1.0)])
        # dependent controls
        defocus_m_control = Control("defocus_m", c10.output_value, [(c10, 1.0)])

        self.__controls = {
            "ZLPtare": zlp_tare_control,
            "ZLPoffset": zlp_offset_control,
            "C10": c10,
            "C12.x": c12_x,
            "C12.y": c12_y,
            "C21.x": c21_x,
            "C21.y": c21_y,
            "C23.x": c23_x,
            "C23.y": c23_y,
            "C30": c30,
            "C32.x": c32_x,
            "C32.y": c32_y,
            "C34.x": c34_x,
            "C34.y": c34_y,
            "C10Control": c10Control,
            "C12Control.x": c12Control_x,
            "C12Control.y": c12Control_y,
            "C21Control.x": c21Control_x,
            "C21Control.y": c21Control_y,
            "C23Control.x": c23Control_x,
            "C23Control.y": c23Control_y,
            "C30Control": c30Control,
            "C32Control.x": c32Control_x,
            "C32Control.y": c32Control_y,
            "C34Control.x": c34Control_x,
            "C34Control.y": c34Control_y,
            "defocus_m_control": defocus_m_control,
            }

        def control_changed(control: Control) -> None:
            self.property_changed_event.fire(control.name)

        for control in self.__controls.values():
            control.on_changed = control_changed

        controls_added = []
        for name, control in self.__controls.items():
            if "." in name and not name in controls_added:
                splitname = name.split(".")
                if splitname[1] == "x":
                    x_name = name
                    y_name = splitname[0] + "." + "y"
                elif splitname[1] == "y":
                    y_name = name
                    x_name = splitname[0] + "." + "x"
                else:
                    continue
                # we need to wrap the getter and setter functions into these "creator" functions in order to
                # de-reference the y_name and x_name variables. Otherwise python keeps using the variable names
                # which change with each iteration of the for-loop.
                def make_getter(y_name, x_name):
                    def getter(self):
                        return Geometry.FloatPoint(self.__controls[y_name].output_value,
                                                   self.__controls[x_name].output_value)
                    return getter
                def make_setter(y_name, x_name):
                    def setter(self, value):
                        self.__controls[y_name].set_output_value(value.y)
                        self.__controls[x_name].set_output_value(value.x)
                        self.property_changed_event.fire(y_name.split(".")[0])
                    return setter

                setattr(Instrument, splitname[0], property(make_getter(y_name, x_name),
                                                           make_setter(y_name, x_name)))
                controls_added.append(x_name)
                controls_added.append(y_name)
            elif name not in controls_added:
                def make_getter(name):
                    def getter(self):
                        return self.__controls[name].output_value
                    return getter
                def make_setter(name):
                    def setter(self, value):
                        self.__controls[name].set_output_value(value)
                        self.property_changed_event.fire(name)
                    return setter

                setattr(Instrument, name, property(make_getter(name), make_setter(name)))
                controls_added.append(name)

        self.__cameras = {
            "ronchigram": RonchigramCameraSimulator.RonchigramCameraSimulator(self, self.__ronchigram_shape, self.counts_per_electron, self.__convergence_angle_rad),
            "eels": EELSCameraSimulator.EELSCameraSimulator(self, self.__eels_shape, self.counts_per_electron)
        }

    def close(self):
        for camera in self.__cameras.values():
            camera.close()
        self.__cameras = dict()

    def _get_camera_simulator(self, camera_id: str) -> CameraSimulator:
        return self.__cameras[camera_id]

    @property
    def sample(self) -> SampleSimulator.Sample:
        return self.__sample

    @property
    def live_probe_position(self):
        return self.__live_probe_position

    @live_probe_position.setter
    def live_probe_position(self, position):
        self.__live_probe_position = position
        self.property_changed_event.fire("live_probe_position")

    def get_control(self, control_name: str) -> Control:
        return self.__controls[control_name]

    @property
    def sequence_progress(self):
        with self.__lock:
            return self.__sequence_progress

    @sequence_progress.setter
    def sequence_progress(self, value):
        with self.__lock:
            self.__sequence_progress = value

    def increment_sequence_progress(self):
        with self.__lock:
            self.__sequence_progress += 1

    def trigger_camera_frame(self) -> None:
        self.__camera_frame_event.set()

    def wait_for_camera_frame(self, timeout: float) -> bool:
        result = self.__camera_frame_event.wait(timeout)
        self.__camera_frame_event.clear()
        return result

    def get_scan_data(self, frame_parameters, channel) -> numpy.ndarray:
        size = Geometry.IntSize.make(frame_parameters.subscan_pixel_size if frame_parameters.subscan_pixel_size else frame_parameters.size)
        offset_m = self.stage_position_m - self.beam_shift_m
        fov_size_nm = Geometry.FloatSize.make(frame_parameters.fov_size_nm) if frame_parameters.fov_size_nm else Geometry.FloatSize(frame_parameters.fov_nm, frame_parameters.fov_nm)
        if frame_parameters.subscan_fractional_size:
            subscan_fractional_size = Geometry.FloatSize.make(frame_parameters.subscan_fractional_size)
            fov_size_nm = Geometry.FloatSize(height=fov_size_nm.height * subscan_fractional_size.height,
                                             width=fov_size_nm.width * subscan_fractional_size.width)
        center_nm = Geometry.FloatPoint.make(frame_parameters.center_nm)
        if frame_parameters.subscan_fractional_center:
            subscan_fractional_center = Geometry.FloatPoint.make(frame_parameters.subscan_fractional_center)
            center_nm += Geometry.FloatPoint(y=(subscan_fractional_center.y - 0.5) * fov_size_nm.height,
                                             x=(subscan_fractional_center.x - 0.5) * fov_size_nm.width)
        extra = int(math.ceil(max(size.height * math.sqrt(2) - size.height, size.width * math.sqrt(2) - size.width)))
        extra_nm = Geometry.FloatPoint(y=(extra / size.height) * fov_size_nm[0], x=(extra / size.width) * fov_size_nm[1])
        used_size = size + Geometry.IntSize(height=extra, width=extra)
        data = numpy.zeros((used_size.height, used_size.width), numpy.float32)
        self.__sample.plot_features(data, offset_m, fov_size_nm, extra_nm, center_nm, used_size)
        noise_factor = 0.3
        if frame_parameters.rotation_rad != 0:
            inner_height = size.height / used_size.height
            inner_width = size.width / used_size.width
            inner_bounds = ((1.0 - inner_height) * 0.5, (1.0 - inner_width) * 0.5), (inner_height, inner_width)
            data = Core.function_crop_rotated(DataAndMetadata.new_data_and_metadata(data), inner_bounds, -frame_parameters.rotation_rad).data
            # TODO: data is not always the correct size
        else:
            data = data[extra // 2:extra // 2 + size.height, extra // 2:extra // 2 + size.width]
        data = (data + numpy.random.randn(size.height, size.width) * noise_factor) * frame_parameters.pixel_time_us
        self.__last_scan_params = size, fov_size_nm, center_nm
        return data

    def camera_sensor_dimensions(self, camera_type: str) -> typing.Tuple[int, int]:
        if camera_type == "ronchigram":
            return self.__ronchigram_shape[0], self.__ronchigram_shape[1]
        else:
            return self.__eels_shape[0], self.__eels_shape[1]

    def camera_readout_area(self, camera_type: str) -> typing.Tuple[int, int, int, int]:
        # returns readout area TLBR
        if camera_type == "ronchigram":
            return 0, 0, self.__ronchigram_shape[0], self.__ronchigram_shape[1]
        else:
            return 0, 0, self.__eels_shape[0], self.__eels_shape[1]

    @property
    def counts_per_electron(self):
        return 40

    def get_electrons_per_pixel(self, pixel_count: int, exposure_s: float) -> float:
        beam_current_pa = self.__beam_current * 1E12
        e_per_pa = 6.242E18 / 1E12
        beam_e = beam_current_pa * e_per_pa
        e_per_pixel_per_second = beam_e / pixel_count
        return e_per_pixel_per_second * exposure_s

    def get_camera_data(self, camera_type: str, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float) -> DataAndMetadata.DataAndMetadata:
        return self.__cameras[camera_type].get_frame_data(readout_area, binning_shape, exposure_s, self.__last_scan_params)

    def get_camera_dimensional_calibrations(self, camera_type: str, readout_area: Geometry.IntRect = None, binning_shape: Geometry.IntSize = None):
        return self.__cameras[camera_type].get_dimensional_calibrations(readout_area, binning_shape)

    @property
    def stage_position_m(self) -> Geometry.FloatPoint:
        return self.__stage_position_m

    @stage_position_m.setter
    def stage_position_m(self, value: Geometry.FloatPoint) -> None:
        self.__stage_position_m = value
        self.property_changed_event.fire("stage_position_m")

    @property
    def beam_shift_m(self) -> Geometry.FloatPoint:
        return self.__beam_shift_m

    @beam_shift_m.setter
    def beam_shift_m(self, value: Geometry.FloatPoint) -> None:
        self.__beam_shift_m = value
        self.property_changed_event.fire("beam_shift_m")

    @property
    def defocus_m(self) -> float:
        return self.__controls["C10"].output_value

    @defocus_m.setter
    def defocus_m(self, value: float) -> None:
        self.__controls["C10"].set_output_value(value)

    @property
    def voltage(self) -> float:
        return self.__voltage

    @voltage.setter
    def voltage(self, value: float) -> None:
        self.__voltage = value
        self.property_changed_event.fire("voltage")

    @property
    def beam_current(self) -> float:
        return self.__beam_current

    @beam_current.setter
    def beam_current(self, value: float) -> None:
        self.__beam_current = value
        self.property_changed_event.fire("beam_current")

    @property
    def is_blanked(self) -> bool:
        return self.__blanked

    @is_blanked.setter
    def is_blanked(self, value: bool) -> None:
        self.__blanked = value
        self.property_changed_event.fire("is_blanked")

    @property
    def is_slit_in(self) -> bool:
        return self.__slit_in

    @is_slit_in.setter
    def is_slit_in(self, value: bool) -> None:
        self.__slit_in = value
        self.property_changed_event.fire("is_slit_in")

    @property
    def energy_offset_eV(self) -> float:
        return self.__controls["ZLPoffset"].output_value

    @energy_offset_eV.setter
    def energy_offset_eV(self, value: float) -> None:
        self.__controls["ZLPoffset"].set_output_value(value)
        # TODO: this should be fired whenever ZLPoffset changes; not just when this method is called.
        self.property_changed_event.fire("energy_offset_eV")

    @property
    def energy_per_channel_eV(self) -> float:
        return self.__energy_per_channel_eV

    @energy_per_channel_eV.setter
    def energy_per_channel_eV(self, value: float) -> None:
        self.__energy_per_channel_eV = value
        self.property_changed_event.fire("energy_per_channel_eV")

    @property
    def order_1_max_angle(self) -> float:
        return self.__order_1_max_angle

    @order_1_max_angle.setter
    def order_1_max_angle(self, value: float) -> None:
        self.__order_1_max_angle = value
        self.property_changed_event.fire("order_1_max_angle")

    @property
    def order_2_max_angle(self) -> float:
        return self.__order_2_max_angle

    @order_2_max_angle.setter
    def order_2_max_angle(self, value: float) -> None:
        self.__order_2_max_angle = value
        self.property_changed_event.fire("order_2_max_angle")

    @property
    def order_3_max_angle(self) -> float:
        return self.__order_3_max_angle

    @order_3_max_angle.setter
    def order_3_max_angle(self, value: float) -> None:
        self.__order_3_max_angle = value
        self.property_changed_event.fire("order_3_max_angle")

    @property
    def order_1_patch(self) -> float:
        return self.__order_1_patch

    @order_1_patch.setter
    def order_1_patch(self, value: float) -> None:
        self.__order_1_patch = value
        self.property_changed_event.fire("order_1_patch")

    @property
    def order_2_patch(self) -> float:
        return self.__order_2_patch

    @order_2_patch.setter
    def order_2_patch(self, value: float) -> None:
        self.__order_2_patch = value
        self.property_changed_event.fire("order_2_patch")

    @property
    def order_3_patch(self) -> float:
        return self.__order_3_patch

    @order_3_patch.setter
    def order_3_patch(self, value: float) -> None:
        self.__order_3_patch = value
        self.property_changed_event.fire("order_3_patch")

    @property
    def C1_range(self) -> float:
        return self.__c1_range

    @C1_range.setter
    def C1_range(self, value: float) -> None:
        self.__c1_range = value
        self.property_changed_event.fire("C1_range")

    @property
    def C2_range(self) -> float:
        return self.__c2_range

    @C2_range.setter
    def C2_range(self, value: float) -> None:
        self.__c2_range = value
        self.property_changed_event.fire("C2_range")

    @property
    def C3_range(self) -> float:
        return self.__c3_range

    @C3_range.setter
    def C3_range(self, value: float) -> None:
        self.__c3_range = value
        self.property_changed_event.fire("C3_range")

    def get_autostem_properties(self):
        """Return a new autostem properties (dict) to be recorded with an acquisition.

           * use property names that are lower case and separated by underscores
           * use property names that include the unit attached to the end
           * avoid using abbreviations
           * avoid adding None entries
           * dict must be serializable using json.dumps(dict)

           Be aware that these properties may be used far into the future so take care when designing additions and
           discuss/review with team members.
        """
        return {
            "high_tension_v": self.voltage,
            "defocus_m": self.defocus_m,
        }

    # these are required functions to implement the standard stem controller interface.

    def TryGetVal(self, s: str) -> (bool, float):

        def parse_camera_values(p: str, s: str) -> (bool, float):
            if s == "y_offset":
                return True, self.get_camera_dimensional_calibrations(p)[0].offset
            elif s == "x_offset":
                return True, self.get_camera_dimensional_calibrations(p)[1].offset
            elif s == "y_scale":
                return True, self.get_camera_dimensional_calibrations(p)[0].scale
            elif s == "x_scale":
                return True, self.get_camera_dimensional_calibrations(p)[1].scale
            return False, None

        if s == "EELS_MagneticShift_Offset":
            return True, self.energy_offset_eV
        elif s == "C_Blank":
            return True, 1.0 if self.is_blanked else 0.0
        elif s in self.__controls:
            return True, self.__controls[s].output_value
        # This handles all supported aberration coefficients
        elif re.match("(C[1-3][0-4])(\.[auxbvy]|$)$", s):
            split_s = s.split('.')
            control = getattr(self, split_s[0], None)
            if control is not None:
                if len(split_s) > 1:
                     if split_s[1] in ("aux"):
                         return True, control.x
                     elif split_s[1] in ("bvy"):
                         return True, control.y
                else:
                    return True, control
        # This handles the target values for all supported aberration coefficients
        elif re.match("(\^C[1-3][0-4])(\.[auxbvy]|$)$", s):
            return True, 0.0
        # This handles the "require" values for all supported aberration coefficients
        elif re.match("C[1-3][0-4]?Range$", s):
            value = getattr(self, f"{s[:2]}_range", None)
            if value is not None:
                return True, value
        # This handles the tuning max angles and patch sizes for all supported aberration coefficients
        elif re.match("Order[1-3](MaxAngle|Patch)$", s):
            if s.endswith("MaxAngle"):
                value = getattr(self, f"order_{s[5]}_max_angle", None)
            else:
                value = getattr(self, f"order_{s[5]}_patch", None)
            if value is not None:
                return True, value
        elif s.startswith("ronchigram_"):
            return parse_camera_values("ronchigram", s[len("ronchigram_"):])
        elif s.startswith("eels_"):
            return parse_camera_values("eels", s[len("eels_"):])
        return False, None

    def GetVal(self, s: str, default_value: float=None) -> float:
        good, d = self.TryGetVal(s)
        if not good:
            if default_value is None:
                raise Exception("No element named '{}' exists! Cannot get value.".format(s))
            else:
                return default_value
        return d

    def SetVal(self, s: str, val: float) -> bool:
        if s == "EELS_MagneticShift_Offset":
            self.energy_offset_eV = val
            return True
        elif s == "C_Blank":
            self.is_blanked = val != 0.0
            return True
        elif s in self.__controls:
            self.__controls[s].set_output_value(val)
            return True
        # This handles all supported aberration coefficients
        elif re.match("(C[1-3][0-4])(\.[auxbvy]|$)$", s):
            split_s = s.split('.')
            control = getattr(self, split_s[0], None)
            if control is not None:
                if len(split_s) > 1:
                     if split_s[1] in ("aux"):
                         setattr(self, split_s[0], control.make((control.y, val)))
                         return True
                     elif split_s[1] in ("bvy"):
                         setattr(self, split_s[0], control.make((val, control.x)))
                         return True
                else:
                    setattr(self, split_s[0], val)
                    return True
        return False

    def SetValWait(self, s: str, val: float, timeout_ms: int) -> bool:
        return self.SetVal(s, val)

    def SetValAndConfirm(self, s: str, val: float, tolfactor: float, timeout_ms: int) -> bool:
        return self.SetVal(s, val)

    def SetValDelta(self, s: str, delta: float) -> bool:
        return self.SetVal(s, self.GetVal(s) + delta)

    def InformControl(self, s: str, val: float) -> bool:
        # here we need to check first for the match with aberration coefficients. Otherwise the
        # "property_changed_event" will be fired with the wrong paramter
        # This handles all supported aberration coefficients
        if re.match("(C[1-3][0-4])(\.[auxbvy]|$)$", s):
            split_s = s.split('.')
            if len(split_s) > 1:
                 if split_s[1] in ("aux"):
                     control = self.__controls.get(split_s[0] + ".x")
                     if control:
                         control.inform_output_value(val)
                         self.property_changed_event.fire(split_s[0])
                         return True
                 elif split_s[1] in ("bvy"):
                     control = self.__controls.get(split_s[0] + ".y")
                     if control:
                         control.inform_output_value(val)
                         self.property_changed_event.fire(split_s[0])
                         return True
            else:
                control = self.__controls.get(s)
                if control:
                    control.inform_output_value(val)
                    self.property_changed_event.fire(s)
                    return True
        elif s in self.__controls:
            self.__controls[s].inform_output_value(val)
            self.property_changed_event.fire(s)
            return True
        return self.SetVal(s, val)

    def change_stage_position(self, *, dy: int=None, dx: int=None):
        """Shift the stage by dx, dy (meters). Do not wait for confirmation."""
        self.stage_position_m += Geometry.FloatPoint(y=-dy, x=-dx)

    def change_pmt_gain(self, pmt_type: stem_controller.PMTType, *, factor: float) -> None:
        """Change specified PMT by factor. Do not wait for confirmation."""
        pass
