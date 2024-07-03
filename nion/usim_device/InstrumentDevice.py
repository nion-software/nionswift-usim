"""
Useful references:
    http://www.rodenburg.org/guide/index.html
    http://www.ammrf.org.au/myscope/
"""

from __future__ import annotations

# standard libraries
import math
import re
import time
import typing

import numpy.typing

from nion.data import DataAndMetadata
from nion.data import Core
from nion.device_kit import CameraDevice
from nion.device_kit import InstrumentDevice
from nion.device_kit import ScanDevice
from nion.instrumentation import stem_controller
from nion.swift.model import Utility
from nion.usim_device import SampleSimulator
from nion.utils import Geometry
from nion.utils import Observable
from nion.utils import ReferenceCounting


_NDArray = numpy.typing.NDArray[typing.Any]

WeightedInput = typing.Tuple["Variable", typing.Union[float, "Variable"]]


class ControlProvider(typing.Protocol):
    def get_control(self, control_name: str) -> typing.Union[Variable, Control2D, None]:
        ...


class Variable:
    """Evaluate an expression plus a sum of input values."""

    def __init__(self, name: str, weighted_inputs: typing.Optional[typing.List[WeightedInput]] = None):
        self.name = name
        self.weighted_inputs = weighted_inputs if weighted_inputs else list()
        self.__variables: typing.Dict[str, typing.Any] = dict()
        self.__expression: typing.Optional[str] = None
        self.dependents: typing.List[Variable] = list()
        for input, _ in self.weighted_inputs:
            input.add_dependent(self)
        self.__last_output_value: typing.Optional[float] = None
        self.on_changed: typing.Optional[typing.Callable[[Variable], None]] = None

    def __str__(self) -> str:
        return "{}: {}".format(self.name, self.output_value)

    @property
    def weighted_input_value(self) -> float:
        weighted_input = 0.0
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
    def variables(self) -> typing.Mapping[str, typing.Any]:
        return self.__variables

    def get_expression(self) -> typing.Optional[str]:
        return self.__expression

    def set_expression(self, expression: str, variables: typing.Optional[typing.Mapping[str, typing.Any]] = None, instrument: typing.Optional[ControlProvider] = None) -> None:
        if variables is not None:
            resolved_variables = dict()
            for key, value in variables.items():
                if isinstance(value, str):
                    if instrument is None:
                        raise TypeError("An value_manager instance is required when using string names as control identifiers.")
                    value = instrument.get_control(value)
                    if value is None:
                        raise ValueError(f"Cannot get value for name {key}.")
                if isinstance(value, Control2D):
                    raise TypeError("2D controls cannot be used in expressions")
                if isinstance(value, Variable):
                    if value == self:
                        raise ValueError("An expression cannot include the control it is attached to.")
                    value.add_dependent(self)
                resolved_variables[key] = value
            self.__variables = resolved_variables
        self.__expression = expression
        self._notify_change()

    def __evaluate_expression(self) -> float:
        variables = dict()
        for key, value in self.__variables.items():
            if isinstance(value, Variable):
                value = value.output_value
            variables[key] = value
        try:
            assert self.__expression
            res = typing.cast(float, eval(self.__expression, globals(), variables))
        except Exception:
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

    def set_output_value(self, value: float) -> None:
        raise NotImplementedError()

    def inform_output_value(self, value: float) -> None:
        raise NotImplementedError()


class Control(Variable):
    """
    Controls keep an output value equal to the weight sum of input values plus a local value.

    TODO: add optional noise (continuous and periodic)
    TODO: add response time to changes
    TODO: add hysteresis
    """

    def __init__(self, name: str, local_value: float = 0.0, weighted_inputs: typing.Optional[typing.List[WeightedInput]] = None):
        super().__init__(name, weighted_inputs)
        self.local_value = float(local_value)
        # For some tests we need to simulate settings controls which have a reference index. Don't make a difference
        # between normal and settings controls for now since it is not needed.
        self.reference_index = 0

    def __str__(self) -> str:
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
        axis_description: typing.Optional[AxisDescription] = None
        for axis_description in AxisManager().supported_axis_descriptions:
            if axis_description.axis_type == axis:
                break
        assert axis_description
        self.__axis_description = axis_description

    @property
    def __weighted_input_value_2d(self) -> typing.Tuple[float, float]:
        value_a = self.__control_2d.controls[0].weighted_input_value
        value_b = self.__control_2d.controls[1].weighted_input_value
        return AxisManager().axis_transform_point(Geometry.FloatPoint(x=value_a, y=value_b), self.__control_2d.native_axis_description, self.__axis_description).as_tuple()[::-1]

    @property
    def weighted_input_value(self) -> float:
        return self.__weighted_input_value_2d[self.__index]

    @property
    def __output_value_2d(self) -> typing.Tuple[float, float]:
        value_a = self.__control_2d.controls[0].output_value
        value_b = self.__control_2d.controls[1].output_value
        return AxisManager().axis_transform_point(Geometry.FloatPoint(x=value_a, y=value_b), self.__control_2d.native_axis_description, self.__axis_description).as_tuple()[::-1]

    @property
    def output_value(self) -> float:
        return self.__output_value_2d[self.__index]

    @property
    def __local_value_2d(self) -> typing.Tuple[float, float]:
        value_a = self.__control_2d.controls[0].local_value
        value_b = self.__control_2d.controls[1].local_value
        return AxisManager().axis_transform_point(Geometry.FloatPoint(x=value_a, y=value_b), self.__control_2d.native_axis_description, self.__axis_description).as_tuple()[::-1]

    @property
    def local_value(self) -> float:
        return self.__local_value_2d[self.__index]

    def set_local_value(self, value: float) -> None:
        value_2d = [self.__local_value_2d[0], self.__local_value_2d[1]]
        value_2d[self.__index] = value
        value_2d_native = AxisManager().axis_transform_point(Geometry.FloatPoint(x=value_2d[0], y=value_2d[1]), self.__axis_description, self.__control_2d.native_axis_description)
        self.__control_2d.controls[0].set_local_value(value_2d_native.x)
        self.__control_2d.controls[1].set_local_value(value_2d_native.y)

    def set_output_value(self, value: float) -> None:
        self.set_local_value(value - self.weighted_input_value)

    def inform_output_value(self, value: float) -> None:
        value_2d = list(self.__output_value_2d)
        value_2d[self.__index] = value
        value_2d_native = AxisManager().axis_transform_point(Geometry.FloatPoint(x=value_2d[0], y=value_2d[1]), self.__axis_description, self.__control_2d.native_axis_description)
        self.__control_2d.controls[0].inform_output_value(value_2d_native.x)
        self.__control_2d.controls[1].inform_output_value(value_2d_native.y)


class AxisDescription(stem_controller.AxisDescription):
    def __init__(self, axis_id: str, axis1: str, axis2: str, display_name: str, searchable_name: str):
        self.__axis_id = axis_id
        self.__axis_type = (axis1, axis2)
        self.__display_name = display_name
        self.searchable_name = searchable_name

    def __str__(self) -> str:
        return self.display_name

    @property
    def axis_id(self) -> str:
        return self.__axis_id

    @property
    def axis_type(self) -> typing.Tuple[str, str]:
        return self.__axis_type

    @property
    def display_name(self) -> str:
        return self.__display_name


class AxisManager(metaclass=Utility.Singleton):
    """
    This object keeps track of all supported axis and performs conversions between them.
    Right now, it does not actually do anything. But it already implements the required 'API' that is used by 2D
    controls and converted controls, so it should be enough to change 'convert_vector' to actually do some conversion
    if we need to support axis with different rotations between each other.
    """

    def __init__(self) -> None:
        self.__supported_axis_descriptions: typing.List[AxisDescription] = [
            AxisDescription("correctoraxis", "a", "b", "CorrectorAxis (a, b)", "CorrectorAxis"),
            AxisDescription("tv", "x", "y", "TV (x, y)", "TV"),
            AxisDescription("scan", "u", "v", "Scan (u, v)", "Scan"),
            AxisDescription("mc", "mx", "my", "MC (mx, my)", "MC"),
            AxisDescription("postsample", "px", "py", "PostSample (px, py)", "PostSample"),
            AxisDescription("stageaxis", "sx", "sy", "StageAxis (sx, sy)", "StageAxis"),
            AxisDescription("stagetiltaxis", "sa", "sb", "StageTiltAxis (sa, sb)", "StageTiltAxis"),
        ]

    @property
    def supported_axis_names(self) -> typing.Sequence[stem_controller.AxisType]:
        return [axis_description.axis_type for axis_description in self.__supported_axis_descriptions]

    @property
    def supported_axis_descriptions(self) -> typing.Sequence[AxisDescription]:
        return self.__supported_axis_descriptions.copy()

    def axis_transform_point(self, point: Geometry.FloatPoint, from_axis: stem_controller.AxisDescription, to_axis: stem_controller.AxisDescription) -> Geometry.FloatPoint:
        return point


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
                 weighted_inputs: typing.Optional[typing.Tuple[typing.List[WeightedInput], typing.List[WeightedInput]]] = None):
        self.name = name
        self.native_axis = native_axis
        axis_description: typing.Optional[AxisDescription] = None
        for axis_description in AxisManager().supported_axis_descriptions:
            if axis_description.axis_type == native_axis:
                break
        assert axis_description
        self.native_axis_description = axis_description
        # give both 'sub-controls' the same name so that the 'property_changed_event' will be fired with the name of
        # the 'parent' 2d-control
        control_b = Control(name, local_value=local_values[1], weighted_inputs=weighted_inputs[1] if weighted_inputs else None)
        control_a = Control(name, local_value=local_values[0], weighted_inputs=weighted_inputs[0] if weighted_inputs else None)

        self.__controls = (control_a, control_b)

    def __str__(self) -> str:
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

    def __getattr__(self, attr: str) -> typing.Any:
        if attr in self.native_axis:
            return self.__controls[self.native_axis.index(attr)]
        axis_names = AxisManager().supported_axis_names
        for axis in axis_names:
            if attr in axis:
                converted = ConvertedControl(self, axis.index(attr), axis)
                return converted
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{attr}'")


class DriftController:

    def __init__(self) -> None:
        self.__start_time = time.perf_counter()

    @property
    def offset_m(self) -> Geometry.FloatPoint:
        # note: positive values will move data down/right
        max_drift_y_m = 15 * 1E-9 * 0
        max_drift_x_m = 10 * 1E-9 * 0
        period_y_s = 60 * 4
        period_x_s = 90 * 4
        phase_y_rad = 7
        phase_x_rad = 10
        return Geometry.FloatPoint(y=max_drift_y_m * math.cos((time.perf_counter() - self.__start_time - phase_y_rad) * 2 * math.pi / period_y_s),
                                   x=max_drift_x_m * math.sin((time.perf_counter() - self.__start_time + phase_x_rad) * 2 * math.pi / period_x_s))


class ValueManager(Observable.Observable, InstrumentDevice.ValueManagerLike):
    def __init__(self) -> None:
        super().__init__()
        self.__stage_position_m = Geometry.FloatPoint()
        self.__drift_controller = DriftController()
        self.__slit_in = False
        self.__energy_per_channel_eV = 0.5
        self.__beam_current = 200E-12  # 200 pA
        self.__blanked = False
        self.__ronchigram_shape = Geometry.IntSize(2048, 2048)
        self.__eels_shape = Geometry.IntSize(256, 1024)
        self.__probe_position: typing.Optional[Geometry.FloatPoint] = None
        self.__live_probe_position: typing.Optional[Geometry.FloatPoint] = None
        self._is_synchronized = False
        self.__controls: typing.Dict[str, typing.Union[Control2D, Variable]] = dict()

        built_in_controls = self.__create_built_in_controls()
        for control in built_in_controls:
            self.add_control(control)
        # We need to set the expressions after adding the controls to InstrumentDevice
        self.__set_expressions()

        # these need to be set if they are used
        self.ronchigram_camera: typing.Optional[CameraDevice.Camera] = None
        self.eels_camera: typing.Optional[CameraDevice.Camera] = None

    def __create_built_in_controls(self) -> typing.List[typing.Union[Variable, Control2D]]:
        zlp_tare_control = Control("ZLPtare")
        zlp_offset_control = Control("ZLPoffset", -20, [(zlp_tare_control, 1.0)])
        stage_position_m = Control2D("stage_position_m", ("x", "y"))
        beam_current = Control("BeamCurrent", 200e-12)
        # monochromator controls
        mc_exists = Control("S_MC_InsideColumn", local_value=8)  # Used by tuning to check if scope has a monochromator
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
        order_1_patch = Variable("Order1Patch")
        order_2_patch = Variable("Order2Patch")
        order_3_patch = Variable("Order3Patch")
        c1_range = Variable("C1Range")
        c2_range = Variable("C2Range")
        c3_range = Variable("C3Range")
        rsq_seconds = Variable("RSquareC2s")
        rsq_thirds = Variable("RSquareC3s")

        # dependent controls
        beam_shift_m_control = Control2D("beam_shift_m", ("x", "y"), (csh.x.output_value, csh.y.output_value), ([(csh.x, 1.0)], [(csh.y, 1.0)]))
        # AxisConverter is commonly used to convert between axis without affecting any hardware
        axis_converter = Control2D("AxisConverter", ("x", "y"))
        # This is needed in the dectris plugins
        ht_kv = Variable("SuperFEG_HTkV")
        ht_kv.set_expression("EHT * 0.001", variables={"EHT": voltage})
        return [stage_position_m, zlp_tare_control, zlp_offset_control, c10, c12, c21, c23, c30, c32, c34, c10Control,
                c12Control, c21Control, c23Control, c30Control, c32Control, c34Control, csh, drift, beam_current,
                beam_shift_m_control, order_1_max_angle, order_2_max_angle, order_3_max_angle, c1_range, c2_range,
                c3_range, c_aperture, aperture_round, s_voa, s_moa, c_aperture_offset, mc_exists, slit_tilt, slit_C10,
                slit_C12, slit_C21, slit_C23, slit_C30, slit_C32, slit_C34, convergence_angle, axis_converter,
                rsq_seconds, rsq_thirds, voltage, order_1_patch, order_2_patch, order_3_patch, ht_kv]

    def __set_expressions(self) -> None:
        typing.cast(Variable, self.get_control("RSquareC2s")).set_expression(
            "(((C21_a**2+C21_b**2)/1296+(C23_a**2+C23_b**2)/144)/lamb**2)*6.283**2*MaxApertureAngle**6",
            variables={"C21_a": "C21.x", "C21_b": "C21.y",
                       "C23_a": "C23.x", "C23_b": "C23.y",
                       "lamb": 3.7e-12, "MaxApertureAngle": 0.03},
            instrument=self)
        typing.cast(Variable, self.get_control("RSquareC3s")).set_expression(
            "((C30**2/5760+(C32_a**2+C32_b**2)/5120+(C34_a**2+C34_b**2)/320)/lamb**2)*6.283**2*MaxApertureAngle**8",
            variables={"C30": "C30",
                       "C32_a": "C32.x", "C32_b": "C32.y",
                       "C34_a": "C34.x", "C34_b": "C34.y",
                       "lamb": 3.7e-12, "MaxApertureAngle": 0.03},
            instrument=self)
        typing.cast(Variable, self.get_control("Order1MaxAngle")).set_expression("-1")
        typing.cast(Variable, self.get_control("Order2MaxAngle")).set_expression("-1")
        typing.cast(Variable, self.get_control("Order3MaxAngle")).set_expression("-1")

    @property
    def drift_offset_m(self) -> Geometry.FloatPoint:
        return self.__drift_controller.offset_m

    @property
    def actual_offset_m(self) -> Geometry.FloatPoint:
        return self.stage_position_m - self.get_value_2d("beam_shift_m") + self.__drift_controller.offset_m

    @property
    def stage_position_m(self) -> Geometry.FloatPoint:
        return self.get_value_2d("stage_position_m")

    @stage_position_m.setter
    def stage_position_m(self, value: Geometry.FloatPoint) -> None:
        self.set_value_2d("stage_position_m", value)

    def control_changed(self, control: Variable) -> None:
        self.property_changed_event.fire(control.name)

    def create_variable(self, name: str, weighted_inputs: typing.Optional[typing.List[WeightedInput]] = None) -> Variable:
        return Variable(name, weighted_inputs)

    def create_control(self, name: str, local_value: float = 0.0, weighted_inputs: typing.Optional[typing.List[WeightedInput]] = None) -> Control:
        return Control(name, local_value, weighted_inputs)

    def create_2d_control(self, name: str, native_axis: stem_controller.AxisType, local_values: typing.Tuple[float, float] = (0.0, 0.0), weighted_inputs: typing.Optional[typing.Tuple[typing.List[WeightedInput], typing.List[WeightedInput]]] = None) -> Control2D:
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

    def remove_control(self, control: typing.Union[Variable, Control2D]) -> None:
        if not control.name in self.__controls:
            raise ValueError(f"No control with name {control.name} exists.")
        self.__controls.pop(control.name)

    def get_control(self, control_name: str) -> typing.Union[Variable, Control2D, None]:
        if "." in control_name:
            split_name = control_name.split(".")
            control = self.__controls.get(split_name[0])
            if isinstance(control, Control2D):
                control = getattr(control, split_name[1], None)
        else:
            control = self.__controls.get(control_name)
        return control

    def add_control_inputs(self, control_name: str, weighted_inputs: typing.List[WeightedInput]) -> None:
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
    def energy_offset_eV(self) -> float:
        return self.__controls["ZLPoffset"].output_value

    @energy_offset_eV.setter
    def energy_offset_eV(self, value: float) -> None:
        control = self.__controls["ZLPoffset"]
        assert isinstance(control, Control)
        control.set_output_value(value)
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

    def __resolve_control_name(self, s: str, set_val: typing.Optional[float] = None) -> typing.Tuple[bool, typing.Optional[float]]:
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

    def get_value(self, name: str) -> typing.Optional[float]:

        def parse_camera_values(camera_device: CameraDevice.Camera, p: str, s: str) -> typing.Optional[float]:
            if s == "y_offset":
                return camera_device.get_dimensional_calibrations(None, None)[0].offset
            elif s == "x_offset":
                return camera_device.get_dimensional_calibrations(None, None)[1].offset
            elif s == "y_scale":
                return camera_device.get_dimensional_calibrations(None, None)[0].scale
            elif s == "x_scale":
                return camera_device.get_dimensional_calibrations(None, None)[1].scale
            return None

        if name == "EELS_MagneticShift_Offset":
            return self.energy_offset_eV
        elif name == "C_Blank":
            return 1.0 if self.is_blanked else 0.0
        # This handles the target values for all aberration coefficients up to 5th order (needed for tuning)
        elif re.match(r"(\^C[1-5][0-6])(\.[auxbvy]|$)$", name):
            return 0.0
        elif name.startswith("ronchigram_"):
            ronchigram_camera = self.ronchigram_camera
            if ronchigram_camera:
                return parse_camera_values(ronchigram_camera, "ronchigram", name[len("ronchigram_"):])
            return None
        elif name.startswith("eels_"):
            eels_camera = self.eels_camera
            if eels_camera:
                return parse_camera_values(eels_camera, "eels", name[len("eels_"):])
            return None
        else:
            success, value = self.__resolve_control_name(name)
            return value if success else None

    def set_value(self, name: str, value: float) -> bool:
        if name == "EELS_MagneticShift_Offset":
            self.energy_offset_eV = value
            return True
        elif name == "C_Blank":
            self.is_blanked = value != 0.0
            return True
        else:
            return self.__resolve_control_name(name, set_val=value)[0]

    def inform_value(self, name: str, value: float) -> bool:
        if "." in name:
            split_s = name.split('.')
            control = self.get_control(split_s[0])  # get the 2d control
            if control is not None:
                axis = getattr(control, split_s[1], None)  # get the control that holds the value for the right axis
                if axis is not None:
                    axis.inform_output_value(value)  # inform the actual value
                    return True
        else:
            control = self.get_control(name)
            if control is not None:
                control.inform_output_value(value)
                return True
        return self.set_value(name, value)

    def get_value_2d(self, name: str, default_value: typing.Optional[Geometry.FloatPoint] = None, *, axis: typing.Optional[stem_controller.AxisType] = None) -> Geometry.FloatPoint:
        control = self.__controls.get(name)
        if isinstance(control, Control2D):
            axis = axis if axis is not None else control.native_axis
            return Geometry.FloatPoint(getattr(control, axis[1]).output_value, getattr(control, axis[0]).output_value)
        if default_value is None:
            raise Exception(f"No 2D element named '{name}' exists! Cannot get value.")
        else:
            return default_value


    def set_value_2d(self, name: str, value: Geometry.FloatPoint, *, axis: typing.Optional[stem_controller.AxisType] = None) -> bool:
        control = self.__controls.get(name)
        if isinstance(control, Control2D):
            axis = axis if axis is not None else control.native_axis
            getattr(control, axis[0]).set_output_value(value.x)
            getattr(control, axis[1]).set_output_value(value.y)
            return True
        return False

    def inform_control_2d(self, name: str, value: Geometry.FloatPoint, *, axis: stem_controller.AxisType) -> bool:
        control = self.__controls.get(name)
        if isinstance(control, Control2D):
            axis = axis if axis is not None else control.native_axis
            getattr(control, axis[0]).inform_output_value(value.x)
            getattr(control, axis[1]).inform_output_value(value.y)
            return True
        return False

    def get_reference_setting_index(self, settings_control: str) -> int:
        if self.get_value(settings_control) is None:
            raise ValueError(f"Cannot obtain information about control {settings_control}. Does the control exist?")
        control = self.get_control(settings_control)
        # Settings controls can only be 1D controls
        assert isinstance(control, Control)
        return control.reference_index


class Instrument(InstrumentDevice.Instrument):
    def __init__(self, instrument_id: str, value_manager: ValueManager, axis_manager: InstrumentDevice.AxisManagerLike, scan_data_generator: ScanDevice.ScanDataGeneratorLike) -> None:
        super().__init__(instrument_id, value_manager, axis_manager)
        self.__usim_value_manager = value_manager  # for some reason this fails typing if it overrides the super __value_manager (mypy 114)
        self.__property_changed_listener = value_manager.property_changed_event.listen(ReferenceCounting.weak_partial(Instrument.__on_property_changed, self))
        self.__scan_data_generator = scan_data_generator

    @property
    def scan_data_generator(self) -> ScanDevice.ScanDataGeneratorLike:
        return self.__scan_data_generator

    def __on_property_changed(self, name: str) -> None:
        self.property_changed_event.fire(name)

    def get_value(self, name: str) -> typing.Optional[float]:
        return self.value_manager.get_value(name)

    def set_value(self, name: str, value: float) -> bool:
        return self.value_manager.set_value(name, value)

    def inform_value(self, name: str, value: float) -> bool:
        return self.value_manager.inform_value(name, value)

    def get_value_2d(self, s: str, default_value: typing.Optional[Geometry.FloatPoint] = None, *, axis: typing.Optional[stem_controller.AxisType] = None) -> Geometry.FloatPoint:
        return self.value_manager.get_value_2d(s, default_value=default_value, axis=axis)

    def set_value_2d(self, s: str, value: Geometry.FloatPoint, *, axis: typing.Optional[stem_controller.AxisType] = None) -> bool:
        return self.value_manager.set_value_2d(s, value, axis=axis)

    def inform_control_2d(self, s: str, value: Geometry.FloatPoint, *, axis: stem_controller.AxisType) -> bool:
        return self.value_manager.inform_control_2d(s, value, axis=axis)

    def get_reference_setting_index(self, settings_control: str) -> int:
        return self.value_manager.get_reference_setting_index(settings_control)

    # needed for analytical tests (e.g. alignment)

    def get_control(self, control_name: str) -> typing.Union[Variable, Control2D, None]:
        return self.__usim_value_manager.get_control(control_name)

    def add_control(self, control: typing.Union[Variable, Control2D]) -> None:
        self.__usim_value_manager.add_control(control)

    def add_control_inputs(self, control_name: str, weighted_inputs: typing.List[WeightedInput]) -> None:
        self.__usim_value_manager.add_control_inputs(control_name, weighted_inputs)

    def create_control(self, name: str, local_value: float = 0.0, weighted_inputs: typing.Optional[typing.List[WeightedInput]] = None) -> Control:
        return self.__usim_value_manager.create_control(name, local_value, weighted_inputs)

    def remove_control(self, control: typing.Union[Variable, Control2D]) -> None:
        self.__usim_value_manager.remove_control(control)

    def create_2d_control(self, name: str, native_axis: stem_controller.AxisType, local_values: typing.Tuple[float, float] = (0.0, 0.0), weighted_inputs: typing.Optional[typing.Tuple[typing.List[WeightedInput], typing.List[WeightedInput]]] = None) -> Control2D:
        return self.__usim_value_manager.create_2d_control(name, native_axis, local_values, weighted_inputs)


class ScanDataGenerator(Observable.Observable, ScanDevice.ScanDataGeneratorLike):
    def __init__(self, *, sample_index: int = 0) -> None:
        super().__init__()
        self.stage_size_nm = 1000
        # define the samples
        self.__samples = [SampleSimulator.RectangleFlakeSample(self.stage_size_nm), SampleSimulator.AmorphousSample(self.stage_size_nm), SampleSimulator.CombinedTestSample(self.stage_size_nm)]
        self.__sample_index = sample_index

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

    def generate_scan_data(self, instrument: InstrumentDevice.Instrument, scan_frame_parameters: ScanDevice.ScanFrameParameters) -> numpy.typing.NDArray[numpy.float32]:
        size = scan_frame_parameters.size
        fov_size_nm = scan_frame_parameters.fov_size_nm
        rotation = scan_frame_parameters.rotation_rad
        center_nm = scan_frame_parameters.center_nm
        pixel_time_us = scan_frame_parameters.pixel_time_us
        assert fov_size_nm
        # Add some margin in case we need to rotate the data later
        extra = int(math.ceil(max(size.height * math.sqrt(2) - size.height, size.width * math.sqrt(2) - size.width)))
        extra_nm = Geometry.FloatPoint(y=(extra / size.height) * fov_size_nm.height, x=(extra / size.width) * fov_size_nm.width)
        used_size = size + Geometry.IntSize(height=extra, width=extra)
        data: numpy.typing.NDArray[numpy.float32] = numpy.zeros(tuple(used_size), numpy.float32)
        # Now get the data from the sample simulator
        value_manager = typing.cast(ValueManager, instrument.value_manager)
        offset_m = value_manager.actual_offset_m  # stage position - beam shift + drift
        self.sample.plot_features(data, offset_m, fov_size_nm, extra_nm, center_nm, used_size)
        noise_factor = 0.3
        if rotation != 0:
            inner_height = size.height / used_size.height
            inner_width = size.width / used_size.width
            inner_bounds = ((1.0 - inner_height) * 0.5, (1.0 - inner_width) * 0.5), (inner_height, inner_width)
            rotated_xdata = Core.function_crop_rotated(DataAndMetadata.new_data_and_metadata(data), inner_bounds, -rotation)
            assert rotated_xdata
            rotated_data = rotated_xdata.data
            assert rotated_data is not None
            data = rotated_data
        else:
            data = data[extra // 2:extra // 2 + size.height, extra // 2:extra // 2 + size.width]
        return typing.cast(_NDArray, (data + numpy.random.randn(size.height, size.width) * noise_factor) * pixel_time_us)
