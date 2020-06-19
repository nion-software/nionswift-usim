"""
Useful references:
    http://www.rodenburg.org/guide/index.html
    http://www.ammrf.org.au/myscope/
"""

# standard libraries
import copy
import math
import time

import numpy
import threading
import typing
import re

from nion.data import Core
from nion.data import DataAndMetadata
from nion.utils import Event
from nion.utils import Geometry
from nion.swift.model import HardwareSource
from nion.swift.model import Utility
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

class Variable:
    """
    Variables evalute an expression plus a sum of input values.

    """

    def __init__(self, name: str,
                 weighted_inputs: typing.Optional[typing.List[typing.Tuple["Variable", typing.Union[float, typing.Callable]]]] = None):
        self.name = name
        self.weighted_inputs = weighted_inputs if weighted_inputs else list()
        self.__variables = dict()
        self.__expression = None
        self.dependents = list()
        for input, _ in self.weighted_inputs:
            input.add_dependent(self)
        self.__last_output_value = None
        self.on_changed = None

    def __str__(self):
        return "{}: {}".format(self.name, self.output_value)

    @property
    def weighted_input_value(self) -> float:
        weighted_input = 0
        for input_, weight in self.weighted_inputs:
            if isinstance(weight, Variable):
                weighted_input += weight.output_value * input_.output_value
            else:
                weighted_input += weight * input_.output_value
        if self.__expression is not None:
            expr = self.__evaluate_expression()
            weighted_input += expr
        return weighted_input

    @property
    def output_value(self) -> float:
        return self.weighted_input_value

    @property
    def variables(self) -> dict:
        return self.__variables

    def get_expression(self) -> str:
        return self.__expression

    def set_expression(self, expression: str, variables: typing.Optional[dict]=None, instrument: typing.Optional["Instrument"]=None) -> None:
        if variables is not None:
            resolved_variables = dict()
            for key, value in variables.items():
                if isinstance(value, str):
                    if instrument is None:
                        raise TypeError("An instrument controller instance is required when using string names as control identifiers.")
                    value = instrument.get_control(value)
                    if value is None:
                        raise ValueError(f"Cannot get value for name {key}.")
                if isinstance(value, Control2D):
                    raise TypeError("2D controls cannot be used in expressions")
                if isinstance(value, Variable):
                    if value  == self:
                        raise ValueError("An expression cannot include the control it is attached to.")
                    value.add_dependent(self)
                resolved_variables[key] = value
            self.__variables = resolved_variables
        self.__expression = expression
        self._notify_change()

    def __evaluate_expression(self):
        variables = dict()
        for key, value in self.__variables.items():
            if isinstance(value, Variable):
                value = value.output_value
            variables[key] = value
        try:
            res = eval(self.__expression, globals(), variables)
        except:
            import traceback
            traceback.print_exc()
            return 0
        else:
            return res

    def add_input(self, input: "Variable", weight: typing.Union[float, "Variable"]) -> None:
        # if input is already in the list of weighted inputs, overwrite it
        inputs = [control for control, _ in self.weighted_inputs]
        if input in inputs:
            input_index = inputs.index(input)
            self.weighted_inputs[input_index] = (input, weight)
        else:
            self.weighted_inputs.append((input, weight))
        # we can always call add dependent because it checks if self is already in input's dependents
        input.add_dependent(self)
        if isinstance(weight, Variable):
            weight.add_dependent(self)
        self._notify_change()

    def add_dependent(self, dependent: "Variable") -> None:
        if dependent not in self.dependents:
            self.dependents.append(dependent)

    def _notify_change(self) -> None:
        output_value = self.output_value
        if output_value != self.__last_output_value:
            self.__last_output_value = output_value
            if callable(self.on_changed):
                self.on_changed(self)
            for dependent in self.dependents:
                dependent._notify_change()

class Control(Variable):
    """
    Controls keep an output value equal to the weight sum of input values plus a local value.

    TODO: add optional noise (continuous and periodic)
    TODO: add response time to changes
    TODO: add hysteresis
    """

    def __init__(self, name: str, local_value: float = 0.0,
                 weighted_inputs: typing.Optional[typing.List[typing.Tuple["Control", typing.Union[float, typing.Callable]]]] = None):
        super().__init__(name, weighted_inputs)
        self.local_value = float(local_value)

    def __str__(self):
        return "{}: {} + {} = {}".format(self.name, self.weighted_input_value, self.local_value, self.output_value)

    @property
    def output_value(self) -> float:
        return self.weighted_input_value + self.local_value

    def set_local_value(self, value: float) -> None:
        self.local_value = value
        self._notify_change()

    def set_output_value(self, value: float) -> None:
        self.set_local_value(value - self.weighted_input_value)

    def inform_output_value(self, value: float) -> None:
        # save old dependent output values so they can stay constant
        old_dependent_outputs = [dependent.output_value for dependent in self.dependents]
        # set the output value
        self.set_output_value(value)
        # update dependent output values to old values
        for dependent, dependent_output in zip(self.dependents, old_dependent_outputs):
            if isinstance(dependent, (Control, ConvertedControl)):
                dependent.set_output_value(dependent_output)


class ConvertedControl:
    """
    This object is returned when accessing a 'sub-control' from a 2D control in another than its native axis.
    It behaves like a normal 'Control' for getting and setting local and output values, but it does not actually
    save any of these in itself. Instead it converts all values to the native axis and saves them in the
    'original' Control.
    It does not allow adding inputs or dependents, because this should only be done in the native axis.
    """

    def __init__(self, control_2d: "Control2D", index: int, axis: stem_controller.AxisType):
        self.__index = index
        self.__control_2d = control_2d
        self.__axis = axis

    @property
    def __weighted_input_value_2d(self) -> typing.Tuple[float, float]:
        value_a = self.__control_2d.controls[0].weighted_input_value
        value_b = self.__control_2d.controls[1].weighted_input_value
        return AxisManager().convert_vector((value_a, value_b), self.__control_2d.native_axis, self.__axis)

    @property
    def weighted_input_value(self) -> float:
        return self.__weighted_input_value_2d[self.__index]

    @property
    def __output_value_2d(self) -> typing.Tuple[float, float]:
        value_a = self.__control_2d.controls[0].output_value
        value_b = self.__control_2d.controls[1].output_value
        return AxisManager().convert_vector((value_a, value_b), self.__control_2d.native_axis, self.__axis)

    @property
    def output_value(self) -> float:
        return self.__output_value_2d[self.__index]

    @property
    def __local_value_2d(self) -> typing.Tuple[float, float]:
        value_a = self.__control_2d.controls[0].local_value
        value_b = self.__control_2d.controls[1].local_value
        return AxisManager().convert_vector((value_a, value_b), self.__control_2d.native_axis, self.__axis)

    @property
    def local_value(self) -> float:
        return self.__local_value_2d[self.__index]

    def set_local_value(self, value: float) -> None:
        value_2d = [self.__local_value_2d[0], self.__local_value_2d[1]]
        value_2d[self.__index] = value
        value_2d = (value_2d[0], value_2d[1])  # doing it this way keeps the type checker happy
        value_2d_native = AxisManager().convert_vector(value_2d, self.__axis, self.__control_2d.native_axis)
        self.__control_2d.controls[0].set_local_value(value_2d_native[0])
        self.__control_2d.controls[1].set_local_value(value_2d_native[1])

    def set_output_value(self, value: float) -> None:
        self.set_local_value(value - self.weighted_input_value)

    def inform_output_value(self, value: float) -> None:
        value_2d = list(self.__output_value_2d)
        value_2d[self.__index] = value
        value_2d = (value_2d[0], value_2d[1])  # doing it this way keeps the type checker happy
        value_2d_native = AxisManager().convert_vector(value_2d, self.__axis, self.__control_2d.native_axis)
        self.__control_2d.controls[0].inform_output_value(value_2d_native[0])
        self.__control_2d.controls[1].inform_output_value(value_2d_native[1])


class AxisManager(metaclass=Utility.Singleton):
    """
    This object keeps track of all supported axis and performs conversions between them.
    Right now, it does not actually do anything. But it already implements the required 'API' that is used by 2D
    controls and converted controls, so it should be enough to change 'convert_vector' to actually do some conversion
    if we need to support axis with different rotations between each other.
    """

    def __init__(self):
        self.__supported_axis_names = [('a', 'b'), ('x', 'y'), ('u', 'v'), ('mx', 'my'), ('px', 'py'), ('sx', 'sy'), ('sa', 'sb')]

    @property
    def supported_axis_names(self):
        return self.__supported_axis_names.copy()

    def convert_vector(self, vector: typing.Tuple[float, float], from_axis: stem_controller.AxisType, to_axis: stem_controller.AxisType) -> typing.Tuple[float, float]:
        return vector


class Control2D:
    """
    This object represents a 2-dimensional control (i.e. a vector) that supports accessing its components in different
    coordinate systems. It mainly bundles two 1d-controls together and keeps track of the relation between them.
    When creating it, `name` should only be the 'basename' of the control, without a dot in the end or any axis names.
    Different axis can be accessed by using the regular attribute access syntax, i.e. for a control 'C12' you can
    access the components of the vector in ('a', 'b')-coordinates by calling C12.a and C12.b, respectively.
    """

    def __init__(self,
                 name: str,
                 native_axis: stem_controller.AxisType,
                 local_values: typing.Tuple[float, float] = (0.0, 0.0),
                 weighted_inputs: typing.Optional[typing.Tuple[typing.List[typing.Tuple["Control", typing.Union[float, typing.Callable]]],
                                                               typing.List[typing.Tuple["Control", typing.Union[float, typing.Callable]]]]] = None):
        self.name = name
        self.native_axis = native_axis
        if weighted_inputs is None:
            weighted_inputs = (None, None)
        # give both 'sub-controls' the same name so that the 'property_changed_event' will be fired with the name of
        # the 'parent' 2d-control
        control_b = Control(name, local_value=local_values[1], weighted_inputs=weighted_inputs[1])
        control_a = Control(name, local_value=local_values[0], weighted_inputs=weighted_inputs[0])

        self.__controls = (control_a, control_b)

    def __str__(self):
        return "{0}.{1[0]}: {2} + {3} = {4}\n{0}.{1[1]}: {5} + {6} = {7}\n".format(self.name,
                                                                                   self.native_axis,
                                                                                   self.__controls[0].weighted_input_value,
                                                                                   self.__controls[0].local_value,
                                                                                   self.__controls[0].output_value,
                                                                                   self.__controls[1].weighted_input_value,
                                                                                   self.__controls[1].local_value,
                                                                                   self.__controls[1].output_value)

    @property
    def controls(self) -> typing.Tuple["Control", "Control"]:
        return self.__controls

    def __getattr__(self, attr):
        if attr in self.native_axis:
            return self.__controls[self.native_axis.index(attr)]
        axis_names = AxisManager().supported_axis_names
        for axis in axis_names:
            if attr in axis:
                converted = ConvertedControl(self, axis.index(attr), axis)
                return converted
        raise AttributeError(f"'{self.__class__.__name__}' has not attribute '{attr}'")


class DriftController:

    def __init__(self):
        self.__start_time = time.time()

    @property
    def offset_m(self) -> Geometry.FloatPoint:
        # note: positive values will move data down/right
        max_drift_y_m = 15 * 1E-9 * 0
        max_drift_x_m = 10 * 1E-9 * 0
        period_y_s = 60 * 4
        period_x_s = 90 * 4
        phase_y_rad = 7
        phase_x_rad = 10
        return Geometry.FloatPoint(y=max_drift_y_m * math.cos((time.time() - self.__start_time - phase_y_rad) * 2 * math.pi / period_y_s),
                                   x=max_drift_x_m * math.sin((time.time() - self.__start_time + phase_x_rad) * 2 * math.pi / period_x_s))


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
        self.__camera_frame_event_ack = threading.Event()

        # define the STEM geometry limits
        self.stage_size_nm = 1000
        self.max_defocus = 5000 / 1E9

        # define the samples
        self.__samples = [SampleSimulator.RectangleFlakeSample(self.stage_size_nm), SampleSimulator.AmorphousSample()]
        self.__sample_index = 0

        self.__stage_position_m = Geometry.FloatPoint()
        self.__drift_controller = DriftController()
        self.__slit_in = False
        self.__energy_per_channel_eV = 0.5
        self.__beam_current = 200E-12  # 200 pA
        self.__blanked = False
        self.__ronchigram_shape = Geometry.IntSize(2048, 2048)
        self.__eels_shape = Geometry.IntSize(256, 1024)
        self.__scan_context = stem_controller.ScanContext()
        self.__probe_position = None
        self.__live_probe_position = None
        self.__sequence_progress = 0
        self._is_synchronized = False
        self.__lock = threading.Lock()
        self.__controls = dict()

        built_in_controls = self.__create_built_in_controls()
        for control in built_in_controls:
            self.add_control(control)
        # We need to set the expressions after adding the controls to InstrumentDevice
        self.__set_expressions()

        self.__cameras = {
            "ronchigram": RonchigramCameraSimulator.RonchigramCameraSimulator(self, self.__ronchigram_shape, self.counts_per_electron, self.stage_size_nm),
            "eels": EELSCameraSimulator.EELSCameraSimulator(self, self.__eels_shape, self.counts_per_electron)
        }

    def close(self):
        for camera in self.__cameras.values():
            camera.close()
        self.__cameras = dict()

    def _get_camera_simulator(self, camera_id: str) -> CameraSimulator:
        return self.__cameras[camera_id]

    def __create_built_in_controls(self) -> typing.List[typing.Union[Variable, Control2D]]:
        zlp_tare_control = Control("ZLPtare")
        zlp_offset_control = Control("ZLPoffset", -20, [(zlp_tare_control, 1.0)])
        stage_position_m = Control2D("stage_position_m", ("x", "y"))
        beam_current = Control("BeamCurrent", 200e-12)
        # monochromator controls
        mc_exists = Control("S_MC_InsideColumn", local_value=1) # Used by tuning to check if scope has a monochromator
        slit_tilt = Control2D("SlitTilt", ("x", "y"))
        slit_C10 = Control("Slit_C10")
        slit_C12 = Control2D("Slit_C12", ("x", "y"))
        slit_C21 = Control2D("Slit_C21", ("x", "y"))
        slit_C23 = Control2D("Slit_C23", ("x", "y"))
        slit_C30 = Control("Slit_C30")
        slit_C32 = Control2D("Slit_C32", ("x", "y"))
        slit_C34 = Control2D("Slit_C34", ("x", "y"))
        # VOA controls
        c_aperture_offset = Control2D("CApertureOffset", ("x", "y"))
        c_aperture = Control2D("CAperture", ("x", "y"), weighted_inputs=([(c_aperture_offset.x, 1.0), (slit_tilt.x, 1.0)],
                                                                         [(c_aperture_offset.y, 1.0), (slit_tilt.y, 1.0)]))
        aperture_round = Control2D("ApertureRound", ("x", "y"))
        s_voa = Control("S_VOA")
        s_moa = Control("S_MOA")
        convergence_angle = Control("ConvergenceAngle", 0.04)
        voltage = Control("EHT", 100000)
        c10 = Control("C10", 500 / 1e9)
        c12 = Control2D("C12", ("x", "y"))
        c21 = Control2D("C21", ("x", "y"))
        c23 = Control2D("C23", ("x", "y"))
        c30 = Control("C30")
        c32 = Control2D("C32", ("x", "y"))
        c34 = Control2D("C34", ("x", "y"))
        c10Control = Control("C10Control", 0.0, [(c10, 1.0)])
        c12Control = Control2D("C12Control", ("x", "y"), weighted_inputs=([(c12.x, 1.0)], [(c12.y, 1.0)]))
        c21Control = Control2D("C21Control", ("x", "y"), weighted_inputs=([(c21.x, 1.0)], [(c21.y, 1.0)]))
        c23Control = Control2D("C23Control", ("x", "y"), weighted_inputs=([(c23.x, 1.0)], [(c23.y, 1.0)]))
        c30Control = Control("C30Control", 0.0, [(c30, 1.0)])
        c32Control = Control2D("C32Control", ("x", "y"), weighted_inputs=([(c32.x, 1.0)], [(c32.y, 1.0)]))
        c34Control = Control2D("C34Control", ("x", "y"), weighted_inputs=([(c34.x, 1.0)], [(c34.y, 1.0)]))
        csh = Control2D("CSH", ("x", "y"))
        drift = Control2D("Drift", ("x", "y"))
        # tuning parameters
        order_1_max_angle = Variable("Order1MaxAngle")
        order_2_max_angle = Variable("Order2MaxAngle")
        order_3_max_angle = Variable("Order3MaxAngle")
        c1_range = Variable("C1Range")
        c2_range = Variable("C2Range")
        c3_range = Variable("C3Range")
        rsq_seconds = Variable("RSquareC2s")
        rsq_thirds = Variable("RSquareC3s")
        
        # dependent controls
        beam_shift_m_control = Control2D("beam_shift_m", ("x", "y"), (csh.x.output_value, csh.y.output_value), ([(csh.x, 1.0)], [(csh.y, 1.0)]))
        # AxisConverter is commonly used to convert between axis without affecting any hardware
        axis_converter = Control2D("AxisConverter", ("x", "y"))
        return [stage_position_m, zlp_tare_control, zlp_offset_control, c10, c12, c21, c23, c30, c32, c34, c10Control,
                c12Control, c21Control, c23Control, c30Control, c32Control, c34Control, csh, drift, beam_current,
                beam_shift_m_control, order_1_max_angle, order_2_max_angle, order_3_max_angle, c1_range, c2_range,
                c3_range, c_aperture, aperture_round, s_voa, s_moa, c_aperture_offset, mc_exists, slit_tilt, slit_C10,
                slit_C12, slit_C21, slit_C23, slit_C30, slit_C32, slit_C34, convergence_angle, axis_converter,
                rsq_seconds, rsq_thirds, voltage]
    
    def __set_expressions(self):
        self.get_control("RSquareC2s").set_expression("(((C21_a**2+C21_b**2)/1296+(C23_a**2+C23_b**2)/144)/lamb**2)*6.283**2*MaxApertureAngle**6",
                                                      variables={"C21_a": "C21.x", "C21_b": "C21.y",
                                                                 "C23_a": "C23.x", "C23_b": "C23.y",
                                                                 "lamb": 3.7e-12, "MaxApertureAngle": 0.03},
                                                      instrument=self)
        self.get_control("RSquareC3s").set_expression("((C30**2/5760+(C32_a**2+C32_b**2)/5120+(C34_a**2+C34_b**2)/320)/lamb**2)*6.283**2*MaxApertureAngle**8",
                                                      variables={"C30": "C30",
                                                                 "C32_a": "C32.x", "C32_b": "C32.y",
                                                                 "C34_a": "C34.x", "C34_b": "C34.y",
                                                                 "lamb": 3.7e-12, "MaxApertureAngle": 0.03},
                                                      instrument=self)
        self.get_control("Order1MaxAngle").set_expression("-1")
        self.get_control("Order2MaxAngle").set_expression("-1")
        self.get_control("Order3MaxAngle").set_expression("-1")

    @property
    def sample(self) -> SampleSimulator.Sample:
        return self.__samples[self.__sample_index]

    @property
    def sample_titles(self) -> typing.List[str]:
        return [sample.title for sample in self.__samples]

    @property
    def sample_index(self) -> int:
        return self.__sample_index

    @sample_index.setter
    def sample_index(self, value: int) -> None:
        self.__sample_index = value

    @property
    def live_probe_position(self):
        return self.__live_probe_position

    @live_probe_position.setter
    def live_probe_position(self, position):
        self.__live_probe_position = position
        self.property_changed_event.fire("live_probe_position")

    @property
    def drift_offset_m(self) -> Geometry.FloatPoint:
        return self.__drift_controller.offset_m

    def _set_scan_context_probe_position(self, scan_context: stem_controller.ScanContext, probe_position: Geometry.FloatPoint) -> None:
        self.__scan_context = copy.deepcopy(scan_context)
        self.__probe_position = probe_position

    def control_changed(self, control: Variable) -> None:
        self.property_changed_event.fire(control.name)
        
    def create_variable(self, name: str, weighted_inputs: typing.Optional[typing.List[typing.Tuple[Control, typing.Union[float, Control]]]] = None) -> Variable:
        return Variable(name, weighted_inputs)

    def create_control(self, name: str, local_value: float = 0.0, weighted_inputs: typing.Optional[typing.List[typing.Tuple[Control, typing.Union[float, Control]]]] = None) -> Control:
        return Control(name, local_value, weighted_inputs)

    def create_2d_control(self, name: str, native_axis: stem_controller.AxisType, local_values: typing.Tuple[float, float] = (0.0, 0.0), weighted_inputs: typing.Optional[typing.Tuple[typing.List[typing.Tuple[Control, typing.Union[float, Control]]], typing.List[typing.Tuple[Control, typing.Union[float, Control]]]]] = None):
        return Control2D(name, native_axis, local_values, weighted_inputs)

    def add_control(self, control: typing.Union[Variable, Control2D]) -> None:
        if control.name in self.__controls:
            raise ValueError(f"A control with name {control.name} already exists.")
        if isinstance(control, Control2D):
            control.controls[0].on_changed = self.control_changed
            control.controls[1].on_changed = self.control_changed
        else:
            control.on_changed = self.control_changed
        self.__controls[control.name] = control

    def get_control(self, control_name: str) -> typing.Union[Variable, Control2D, None]:
        if "." in control_name:
            split_name = control_name.split(".")
            control = self.__controls.get(split_name[0])
            if isinstance(control, Control2D):
                control = getattr(control, split_name[1], None)
        else:
            control = self.__controls.get(control_name)
        return control

    def add_control_inputs(self, control_name: str,
                           weighted_inputs: typing.List[typing.Tuple[Variable, typing.Union[float, typing.Callable]]]) -> None:
        control = self.get_control(control_name)
        assert isinstance(control, Variable)
        for input, weight in weighted_inputs:
            control.add_input(input, weight)

    def set_input_weight(self, control_name: str, input_name: str, new_weight: typing.Union[float, Variable]) -> None:
        control = self.get_control(control_name)
        assert isinstance(control, Variable)
        input_control = self.get_control(input_name)
        assert isinstance(input_control, Variable)
        inputs = [control_ for control_, _ in control.weighted_inputs]
        if input_control not in inputs:
            raise ValueError(f"{input_name} is not an input for {control_name}. Please add it first before attempting to change its strength.")
        control.add_input(input_control, new_weight)

    def get_input_weight(self, control_name: str, input_name: str) -> float:
        control = self.get_control(control_name)
        assert isinstance(control, Variable)
        input_control = self.get_control(input_name)
        assert isinstance(input_control, Variable), f"{input_name} is not of type 'Variable' but {type(input_control)}."
        inputs = [control_ for control_, _ in control.weighted_inputs]
        if input_control not in inputs:
            raise ValueError(f"{input_name} is not an input for {control_name}. Please add it first before attempting to get its strength.")
        weight = control.weighted_inputs[inputs.index(input_control)][1]
        if isinstance(weight, Variable):
            return weight.output_value
        return weight

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

    def _enter_synchronized_state(self, scan_controller: HardwareSource.HardwareSource, *, camera: HardwareSource.HardwareSource=None) -> None:
        self._is_synchronized = True

    def _exit_synchronized_state(self, scan_controller: HardwareSource.HardwareSource, *, camera: HardwareSource.HardwareSource=None) -> None:
        self._is_synchronized = False

    def wait_for_camera_ack(self, cancel_event: threading.Event) -> None:
        if self._is_synchronized:
            for _ in range(100):
                if self.__camera_frame_event_ack.wait(5.0 / 100) or cancel_event.is_set():
                    self.__camera_frame_event_ack.clear()
                    return
            print("ACK TIMEOUT")
        self.__camera_frame_event_ack.clear()

    def trigger_camera_frame(self) -> None:
        self.__camera_frame_event.set()

    def wait_for_camera_frame(self, timeout: float) -> bool:
        self.__camera_frame_event_ack.set()
        result = self.__camera_frame_event.wait(timeout)
        self.__camera_frame_event.clear()
        return result

    def get_scan_data(self, frame_parameters, channel) -> numpy.ndarray:
        size = Geometry.IntSize.make(frame_parameters.subscan_pixel_size if frame_parameters.subscan_pixel_size else frame_parameters.size)
        offset_m = self.actual_offset_m  # stage position - beam shift + drift
        fov_size_nm = Geometry.FloatSize.make(frame_parameters.fov_size_nm) if frame_parameters.fov_size_nm else Geometry.FloatSize(frame_parameters.fov_nm, frame_parameters.fov_nm)
        if frame_parameters.subscan_fractional_size:
            subscan_fractional_size = Geometry.FloatSize.make(frame_parameters.subscan_fractional_size)
            used_fov_size_nm = Geometry.FloatSize(height=fov_size_nm.height * subscan_fractional_size.height,
                                                  width=fov_size_nm.width * subscan_fractional_size.width)
        else:
            used_fov_size_nm = fov_size_nm
        center_nm = Geometry.FloatPoint.make(frame_parameters.center_nm)
        if frame_parameters.subscan_fractional_center:
            subscan_fractional_center = Geometry.FloatPoint.make(frame_parameters.subscan_fractional_center) - Geometry.FloatPoint(y=0.5, x=0.5)
            fc = subscan_fractional_center.rotate(frame_parameters.rotation_rad)
            center_nm += Geometry.FloatPoint(y=fc.y * fov_size_nm.height, x=fc.x * fov_size_nm.width)
        extra = int(math.ceil(max(size.height * math.sqrt(2) - size.height, size.width * math.sqrt(2) - size.width)))
        extra_nm = Geometry.FloatPoint(y=(extra / size.height) * used_fov_size_nm[0], x=(extra / size.width) * used_fov_size_nm[1])
        used_size = size + Geometry.IntSize(height=extra, width=extra)
        data = numpy.zeros((used_size.height, used_size.width), numpy.float32)
        self.sample.plot_features(data, offset_m, used_fov_size_nm, extra_nm, center_nm, used_size)
        noise_factor = 0.3
        total_rotation = frame_parameters.rotation_rad
        if frame_parameters.subscan_rotation:
            total_rotation -= frame_parameters.subscan_rotation
        if total_rotation != 0:
            inner_height = size.height / used_size.height
            inner_width = size.width / used_size.width
            inner_bounds = ((1.0 - inner_height) * 0.5, (1.0 - inner_width) * 0.5), (inner_height, inner_width)
            data = Core.function_crop_rotated(DataAndMetadata.new_data_and_metadata(data), inner_bounds, -total_rotation).data
        else:
            data = data[extra // 2:extra // 2 + size.height, extra // 2:extra // 2 + size.width]
        return (data + numpy.random.randn(size.height, size.width) * noise_factor) * frame_parameters.pixel_time_us

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
        beam_current_pa = self.GetVal("BeamCurrent") * 1E12
        e_per_pa = 6.242E18 / 1E12
        beam_e = beam_current_pa * e_per_pa
        e_per_pixel_per_second = beam_e / pixel_count
        return e_per_pixel_per_second * exposure_s

    def get_camera_data(self, camera_type: str, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float) -> DataAndMetadata.DataAndMetadata:
        return self.__cameras[camera_type].get_frame_data(readout_area, binning_shape, exposure_s, self.__scan_context, self.__probe_position)

    def get_camera_dimensional_calibrations(self, camera_type: str, readout_area: Geometry.IntRect = None, binning_shape: Geometry.IntSize = None):
        return self.__cameras[camera_type].get_dimensional_calibrations(readout_area, binning_shape)

    @property
    def actual_offset_m(self) -> Geometry.FloatPoint:
        return self.stage_position_m - self.GetVal2D("beam_shift_m") + self.__drift_controller.offset_m

    @property
    def stage_position_m(self) -> Geometry.FloatPoint:
        return self.GetVal2D("stage_position_m")

    @stage_position_m.setter
    def stage_position_m(self, value: Geometry.FloatPoint) -> None:
        self.SetVal2D("stage_position_m", value)

    @property
    def defocus_m(self) -> float:
        return self.GetVal("C10")

    @defocus_m.setter
    def defocus_m(self, value: float) -> None:
        self.SetVal("C10", value)

    @property
    def voltage(self) -> float:
        return self.GetVal("EHT")

    @voltage.setter
    def voltage(self, value: float) -> None:
        self.SetVal("EHT", value)
        self.property_changed_event.fire("voltage")

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
            "high_tension": self.voltage,
            "defocus": self.defocus_m,
        }

    # these are required functions to implement the standard stem controller interface.

    def __resolve_control_name(self, s: str, set_val: typing.Optional[float]=None) -> typing.Tuple[bool, typing.Optional[float]]:
        if "->" in s:
            input_name, control_name = s.split("->")
            if set_val is not None:
                try:
                    self.set_input_weight(control_name, input_name, set_val)
                except (ValueError, AssertionError):
                    return False, None
                else:
                    return True, None
            else:
                try:
                    value = self.get_input_weight(control_name, input_name)
                except (ValueError, AssertionError):
                    return False, None
                else:
                    return True, value
        else:
            control = self.get_control(s)
            if isinstance(control, (Variable, ConvertedControl)):
                if set_val is not None:
                    control.set_output_value(set_val)
                    return True, None
                else:
                    return True, control.output_value
            return False, None

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
        # This handles the target values for all aberration coefficients up to 5th order (needed for tuning)
        elif re.match("(\^C[1-5][0-6])(\.[auxbvy]|$)$", s):
            return True, 0.0
        elif s.startswith("ronchigram_"):
            return parse_camera_values("ronchigram", s[len("ronchigram_"):])
        elif s.startswith("eels_"):
            return parse_camera_values("eels", s[len("eels_"):])
        else:
            return self.__resolve_control_name(s)

    def GetVal(self, s: str, default_value: float=None) -> float:
        good, d = self.TryGetVal(s)
        if not good:
            if default_value is None:
                raise Exception(f"No element named '{s}' exists! Cannot get value.")
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
        else:
            return self.__resolve_control_name(s, set_val=val)[0]

    def SetValWait(self, s: str, val: float, timeout_ms: int) -> bool:
        return self.SetVal(s, val)

    def SetValAndConfirm(self, s: str, val: float, tolfactor: float, timeout_ms: int) -> bool:
        return self.SetVal(s, val)

    def SetValDelta(self, s: str, delta: float) -> bool:
        return self.SetVal(s, self.GetVal(s) + delta)

    def SetValDeltaAndConfirm(self, s: str, delta: float, tolfactor: float, timeout_ms: int) -> bool:
        return self.SetValAndConfirm(s, self.GetVal(s) + delta, tolfactor, timeout_ms)

    def InformControl(self, s: str, val: float) -> bool:
        if "." in s:
            split_s = s.split('.')
            control = self.get_control(split_s[0]) # get the 2d control
            if control is not None:
                axis = getattr(control, split_s[1], None) # get the control that holds the value for the right axis
                if axis is not None:
                    axis.inform_output_value(val) # inform the actual value
                    return True
        else:
            control = self.get_control(s)
            if control is not None:
                control.inform_output_value(val)
                return True
        return self.SetVal(s, val)

    def GetVal2D(self, s: str, default_value: Geometry.FloatPoint=None, *, axis: stem_controller.AxisType=None) -> Geometry.FloatPoint:
        control = self.__controls.get(s)
        if isinstance(control, Control2D):
            axis = axis if axis is not None else control.native_axis
            return Geometry.FloatPoint(getattr(control, axis[1]).output_value, getattr(control, axis[0]).output_value)
        if default_value is None:
            raise Exception(f"No 2D element named '{s}' exists! Cannot get value.")
        else:
            return default_value

    def SetVal2D(self, s: str, value: Geometry.FloatPoint, *, axis: stem_controller.AxisType=None) -> bool:
        control = self.__controls.get(s)
        if isinstance(control, Control2D):
            axis = axis if axis is not None else control.native_axis
            getattr(control, axis[0]).set_output_value(value.x)
            getattr(control, axis[1]).set_output_value(value.y)
            return True
        return False

    def SetVal2DAndConfirm(self, s: str, value: Geometry.FloatPoint, tolfactor: float, timeout_ms: int, *, axis: stem_controller.AxisType) -> bool:
        return self.SetVal2D(s, value, axis=axis)

    def SetVal2DDelta(self, s: str, delta: Geometry.FloatPoint, *, axis: stem_controller.AxisType) -> bool:
        return self.SetVal2D(s, self.GetVal2D(s, axis=axis) + delta, axis=axis)

    def SetVal2DDeltaAndConfirm(self, s: str, delta: Geometry.FloatPoint, tolfactor: float, timeout_ms: int, *, axis: stem_controller.AxisType) -> bool:
        return self.SetVal2DAndConfirm(s, self.GetVal2D(s, axis=axis) + delta, tolfactor, timeout_ms, axis=axis)

    def InformControl2D(self, s: str, value: Geometry.FloatPoint, *, axis: stem_controller.AxisType) -> bool:
        control = self.__controls.get(s)
        if isinstance(control, Control2D):
            axis = axis if axis is not None else control.native_axis
            getattr(control, axis[0]).inform_output_value(value.x)
            getattr(control, axis[1]).inform_output_value(value.y)
            return True
        return False

    def HasValError(self, s: str) -> bool:
        return False

    def change_stage_position(self, *, dy: int=None, dx: int=None):
        """Shift the stage by dx, dy (meters). Do not wait for confirmation."""
        self.stage_position_m += Geometry.FloatPoint(y=-dy, x=-dx)

    def change_pmt_gain(self, pmt_type: stem_controller.PMTType, *, factor: float) -> None:
        """Change specified PMT by factor. Do not wait for confirmation."""
        pass
