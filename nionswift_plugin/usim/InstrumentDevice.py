"""
Useful references:
    http://www.rodenburg.org/guide/index.html
    http://www.ammrf.org.au/myscope/
"""

# standard libraries
import math
import numpy
import random
import scipy.ndimage.interpolation
import scipy.stats
import threading
import typing
import re

from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata

from nion.utils import Event
from nion.utils import Geometry

from nion.instrumentation import stem_controller


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


def plot_powerlaw(data: numpy.ndarray, multiplier: float, energy_calibration: Calibration.Calibration, offset_eV: float, onset_eV: float) -> None:
    # calculate the range
    # 1 represents 0eV, 0 represents 4000eV
    # TODO: sub-pixel accuracy
    energy_range_eV = [energy_calibration.convert_to_calibrated_value(0), energy_calibration.convert_to_calibrated_value(data.shape[0])]
    envelope = scipy.stats.norm(loc=offset_eV, scale=onset_eV).cdf(numpy.linspace(energy_range_eV[0], energy_range_eV[1], data.shape[0]))
    max_ev = 4000
    powerlaw_dist = scipy.stats.powerlaw(8, loc=0, scale=max_ev)  # this is an increasing function; must be reversed below; 8 is arbitrary but looks good
    powerlaw = powerlaw_dist.pdf(numpy.linspace(max_ev - energy_range_eV[0], max_ev - energy_range_eV[1], data.shape[0]))
    data += envelope * multiplier * powerlaw


def plot_norm(data: numpy.ndarray, multiplier: float, energy_calibration: Calibration.Calibration, energy_eV: float, energy_width_eV: float) -> None:
    # calculate the range
    # 1 represents 0eV, 0 represents 4000eV
    # TODO: sub-pixel accuracy
    data_range = [0, data.shape[0]]
    energy_range_eV = [energy_calibration.convert_to_calibrated_value(data_range[0]), energy_calibration.convert_to_calibrated_value(data_range[1])]
    norm = scipy.stats.norm(loc=energy_eV, scale=energy_width_eV)
    data += multiplier * norm.pdf(numpy.linspace(energy_range_eV[0], energy_range_eV[1], data_range[1])) / norm.pdf(energy_eV)


class Feature:

    def __init__(self, position_m, size_m, edges, plasmon_eV, plurality):
        self.position_m = position_m
        self.size_m = size_m
        self.edges = edges
        self.plasmon_eV = plasmon_eV
        self.plurality = plurality

    def get_scan_rect_m(self, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint) -> Geometry.FloatRect:
        scan_size_m = Geometry.FloatSize(height=fov_nm.height, width=fov_nm.width) / 1E9
        scan_rect_m = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint.make(center_nm) / 1E9, scan_size_m)
        scan_rect_m -= offset_m
        return scan_rect_m

    def get_feature_rect_m(self) -> Geometry.FloatRect:
        return Geometry.FloatRect.from_center_and_size(self.position_m, self.size_m)

    def intersects(self, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, probe_position: Geometry.FloatPoint) -> bool:
        scan_rect_m = self.get_scan_rect_m(offset_m, fov_nm, center_nm)
        feature_rect_m = self.get_feature_rect_m()
        probe_position_m = Geometry.FloatPoint(y=probe_position.y * scan_rect_m.height + scan_rect_m.top, x=probe_position.x * scan_rect_m.width + scan_rect_m.left)
        return scan_rect_m.intersects_rect(feature_rect_m) and feature_rect_m.contains_point(probe_position_m)

    def plot(self, data: numpy.ndarray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> int:
        # TODO: how does center_nm interact with stage position?
        # TODO: take into account feature angle
        # TODO: take into account frame parameters angle
        # TODO: expand features to other shapes than rectangle
        scan_rect_m = self.get_scan_rect_m(offset_m, fov_nm, center_nm)
        feature_rect_m = self.get_feature_rect_m()
        sum = 0
        if scan_rect_m.intersects_rect(feature_rect_m):
            feature_rect_top_px = int(shape[0] * (feature_rect_m.top - scan_rect_m.top) / scan_rect_m.height)
            feature_rect_left_px = int(shape[1] * (feature_rect_m.left - scan_rect_m.left) / scan_rect_m.width)
            feature_rect_height_px = int(shape[0] * feature_rect_m.height / scan_rect_m.height)
            feature_rect_width_px = int(shape[1] * feature_rect_m.width / scan_rect_m.width)
            if feature_rect_top_px < 0:
                feature_rect_height_px += feature_rect_top_px
                feature_rect_top_px = 0
            if feature_rect_left_px < 0:
                feature_rect_width_px += feature_rect_left_px
                feature_rect_left_px = 0
            if feature_rect_top_px + feature_rect_height_px > shape[0]:
                feature_rect_height_px = shape[0] - feature_rect_top_px
            if feature_rect_left_px + feature_rect_width_px > shape[1]:
                feature_rect_width_px = shape[1] - feature_rect_left_px
            feature_rect_origin_px = Geometry.IntPoint(y=feature_rect_top_px, x=feature_rect_left_px)
            feature_rect_size_px = Geometry.IntSize(height=feature_rect_height_px, width=feature_rect_width_px)
            feature_rect_px = Geometry.IntRect(feature_rect_origin_px, feature_rect_size_px)
            data[feature_rect_px.top:feature_rect_px.bottom, feature_rect_px.left:feature_rect_px.right] += 1.0
            sum += (feature_rect_px.bottom - feature_rect_px.top) * (feature_rect_px.right - feature_rect_px.left)
        return sum

    def plot_spectrum(self, data: numpy.ndarray, multiplier: float, energy_calibration: Calibration.Calibration) -> None:
        for edge_eV, onset_eV in self.edges:
            # print(f"edge_eV {edge_eV} onset_eV {onset_eV}")
            strength = multiplier * 0.1
            plot_powerlaw(data, strength, energy_calibration, edge_eV, onset_eV)
        for n in range(1, self.plurality + 1):
            plot_norm(data, multiplier / math.factorial(n), energy_calibration, self.plasmon_eV * n, math.sqrt(self.plasmon_eV))


class AberrationsController:
    """Track aberrations and apply them.

    All values are SI.

    Derived from code by Juan-Carlos Idrobo and Andy Lupini.
    """
    coefficient_names = (
        "c0a", "c0b", "c10", "c12a", "c12b", "c21a", "c21b", "c23a", "c23b", "c30", "c32a", "c32b", "c34a", "c34b",
        "c41a", "c41b", "c43a", "c43b" , "c45a", "c45b", "c50", "c52a", "c52b", "c54a", "c54b", "c56a", "c56b", "c70"
    )

    def __init__(self, height, width, theta, max_defocus, defocus):
        self.__height = height
        self.__width = width
        self.__theta = theta
        self.__max_defocus = max_defocus
        self.__coefficients = dict()
        self.__intermediates = dict()  # functions of height/width/theta
        self.__chis = dict()  # chi's, functions of intermediate and coefficients
        self.__coefficients["c10"] = defocus
        self.__chi = None
        self.__c = None

    def apply(self, aberrations, data):
        import time
        t0 = time.perf_counter()

        height = aberrations["height"]
        width = aberrations["width"]
        theta = aberrations["theta"]

        if theta != self.__theta or width != self.__width or height != self.__height:
            self.__width = width
            self.__height = height
            self.__theta = theta
            self.__intermediates = dict()
            self.__chis = dict()
            self.__chi = None
            self.__c = None

        for coefficient_name in self.coefficient_names:
            if self.__coefficients.get(coefficient_name) != aberrations.get(coefficient_name):
                # print(f"changed {coefficient_name}")
                self.__coefficients[coefficient_name] = aberrations[coefficient_name]
                self.__chis.pop(coefficient_name, None)
                self.__chi = None
                self.__c = None

        # below: the tedious part...

        def get_i0ab():
            i0a = self.__intermediates.get("c0a")
            i0b = self.__intermediates.get("c0b")
            if i0a is None or i0b is None:
                i0a, i0b = numpy.meshgrid(numpy.linspace(-theta, theta, width), numpy.linspace(-theta, theta, height))
                self.__intermediates["c0a"] = i0a
                self.__intermediates["c0b"] = i0b
            return i0a, i0b

        def get_i0a():
            return get_i0ab()[0]

        def get_i0b():
            return get_i0ab()[1]

        def get_i0a_squared():
            i0a_squared = self.__intermediates.get("c0a_squared")
            if i0a_squared is None:
                i0a_squared = get_i0a() ** 2
                self.__intermediates["c0a_squared"] = i0a_squared
            return i0a_squared

        def get_i0b_squared():
            i0b_squared = self.__intermediates.get("c0b_squared")
            if i0b_squared is None:
                i0b_squared = get_i0b() ** 2
                self.__intermediates["c0b_squared"] = i0b_squared
            return i0b_squared

        def get_iradius():
            ir = self.__intermediates.get("ir")
            if ir is None:
                ir = get_i0a_squared() + get_i0b_squared()
                self.__intermediates["ir"] = ir
            return ir

        def get_idiff_sq():
            ids = self.__intermediates.get("ids")
            if ids is None:
                ids = get_i0a_squared() - get_i0b_squared()
                self.__intermediates["ids"] = ids
            return ids

        def get_intermediate(coefficient_name):
            intermediate = self.__intermediates.get(coefficient_name)
            if intermediate is None:
                if coefficient_name == "c0a":
                    intermediate = get_i0a()
                elif coefficient_name == "c0b":
                    intermediate = get_i0b()
                elif coefficient_name == "c10":
                    intermediate = get_iradius() / 2
                elif coefficient_name == "c12a":
                    intermediate = get_idiff_sq() / 2
                elif coefficient_name == "c12b":
                    intermediate = get_i0a() * get_i0b()
                elif coefficient_name == "c21a":
                    intermediate = get_i0a() * (get_i0a_squared() + get_i0b_squared()) / 3
                elif coefficient_name == "c21b":
                    intermediate = get_i0b() * (get_i0a_squared() + get_i0b_squared()) / 3
                elif coefficient_name == "c23a":
                    intermediate = get_i0a() * (get_i0a_squared() - 3 * get_i0b_squared()) / 3
                elif coefficient_name == "c23b":
                    intermediate = get_i0b() * (3 * get_idiff_sq()) / 3
                elif coefficient_name == "c30":
                    intermediate = get_intermediate("c10") ** 2
                elif coefficient_name == "c32a":
                    intermediate = get_intermediate("c10") * get_intermediate("c12a")
                elif coefficient_name == "c32b":
                    intermediate = get_intermediate("c10") * get_intermediate("c12b")
                elif coefficient_name == "c34a":
                    intermediate = (get_i0a_squared() ** 2 - 6 * get_i0a_squared() * get_i0b_squared() + get_i0b_squared() ** 2) / 4
                elif coefficient_name == "c34b":
                    intermediate = get_i0a() ** 3 * get_i0b() - get_i0a() * get_i0b() ** 3
                elif coefficient_name == "c41a":
                    intermediate = 4 * get_i0a() * get_intermediate("c10") ** 2 / 5
                elif coefficient_name == "c41b":
                    intermediate = 4 * get_i0b() * get_intermediate("c10") ** 2 / 5
                elif coefficient_name == "c43a":
                    intermediate = get_iradius() * (get_i0a() * get_idiff_sq()) / 5 - 2 * get_i0a() * get_i0b() ** 2
                elif coefficient_name == "c43b":
                    intermediate = get_iradius() * (get_i0b() * get_idiff_sq()) / 5 + 2 * get_i0b() * get_i0a() ** 2
                elif coefficient_name == "c45a":
                    intermediate = (get_i0a() * get_idiff_sq() ** 2 - 4 * get_i0a() * get_idiff_sq() * get_i0b() ** 2 - 4 * get_i0a() ** 3 * get_i0b() ** 2) / 5
                elif coefficient_name == "c45b":
                    intermediate = (get_i0b() * get_idiff_sq() ** 2 + 4 * get_i0b() * get_idiff_sq() * get_i0a() ** 2 - 4 * get_i0a() ** 2 * get_i0b() ** 3) / 5
                elif coefficient_name == "c50":
                    intermediate = 8 * get_intermediate("c10") ** 3 / 6
                elif coefficient_name == "c52a":
                    intermediate = get_iradius() ** 2 * get_idiff_sq() / 6
                elif coefficient_name == "c52b":
                    intermediate = get_iradius() ** 2 * get_intermediate("c12b") / 3
                elif coefficient_name == "c54a":
                    intermediate = (get_iradius() * (get_idiff_sq() ** 2) / 6 - 2 * get_intermediate("c12b")) ** 2
                elif coefficient_name == "c54b":
                    intermediate = 2 * get_iradius() * get_idiff_sq() * get_intermediate("c12b") / 3
                elif coefficient_name == "c56a":
                    intermediate = get_idiff_sq() ** 3 / 6 - 2 * get_i0a() ** 2 * get_i0b() ** 2 * get_idiff_sq()
                elif coefficient_name == "c56b":
                    intermediate = get_intermediate("c12b") * get_idiff_sq() ** 2 - (4 * get_i0a() ** 3 * get_i0b() ** 3) / 3
                elif coefficient_name == "c70":
                    intermediate = get_intermediate("c10") ** 4
            return intermediate

        def get_chi(coefficient_name):
            chi = self.__chis.get(coefficient_name)
            if chi is None:
                coefficient = self.__coefficients.get(coefficient_name, 0.0)
                if coefficient != 0.0:
                    chi = coefficient * get_intermediate(coefficient_name)
                if chi is not None:
                    self.__chis[coefficient_name] = chi
                else:
                    self.__chis.pop(coefficient_name, None)
            return chi

        if self.__chi is None:
            # print("recalculating chi")
            for coefficient_name in self.coefficient_names:
                partial_chi = get_chi(coefficient_name)
                if partial_chi is not None:
                    if self.__chi is None:
                        # print(f"0 {coefficient_name}")
                        self.__chi = numpy.copy(partial_chi)
                    else:
                        # print(f"+ {coefficient_name}")
                        self.__chi += partial_chi
            self.__c = None

        if self.__c is None and self.__chi is not None:
            # print("recalculating grad chi")
            grad_chi = numpy.gradient(self.__chi)
            max_chi0 = self.__max_defocus * theta  * theta
            max_chi1 = self.__max_defocus * theta * theta * ((1 - 1 / width) * (1 - 1 / width) + (1 - 1 / height) * (1 - 1 / height)) / 2
            max_chi = max_chi0 - max_chi1
            scale_y = height / 2 / max_chi
            scale_x = width / 2 / max_chi
            self.__c = [scale_y * grad_chi[0] + height/2, scale_x * grad_chi[1] + width/2]

        # note, the scaling factor of 2pi/wavelength has been removed from chi since it cancels out.

        if self.__c is not None:
            # scale the offsets so that at max defocus, the coordinates cover the entire area of data.
            t1 = time.perf_counter()
            r = scipy.ndimage.interpolation.map_coordinates(data, self.__c)
            t2 = time.perf_counter()
            # print(f"elapsed {t1 - t0} {t2 - t1}")
            return r

        return numpy.zeros((height, width))


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


class CameraSimulator:

    depends_on = list() # subclasses should define the controls and attributes they depend on here

    def __init__(self, instrument: "Instrument", camera_type: str, sensor_dimensions: Geometry.IntSize, counts_per_electron: int):
        self.__instrument = instrument
        self._camera_type = camera_type
        self._sensor_dimensions = sensor_dimensions
        self._counts_per_electron = counts_per_electron
        self._needs_recalculation = True
        self._last_frame_settings = [Geometry.IntRect((0, 0), (0, 0)), Geometry.IntSize(), 0.0, None]

        def property_changed(name):
            if name in self.depends_on:
                self._needs_recalculation = True

        self.__property_changed_event_listener = instrument.property_changed_event.listen(property_changed)

        # we also need to inform the cameras about changes to the (parked) probe position
        def probe_state_changed(probe_state, probe_position):
            property_changed("probe_state")
            property_changed("probe_position")

        self.__probe_state_changed_event_listener = instrument.probe_state_changed_event.listen(probe_state_changed)

    def close(self):
        self.__property_changed_event_listener.close()
        self.__property_changed_event_listener = None
        self.__probe_state_changed_event_listener.close()
        self.__probe_state_changed_event_listener = None

    @property
    def _camera_shape(self) -> Geometry.IntSize:
        return self._sensor_dimensions

    @property
    def instrument(self) -> "Instrument":
        return self.__instrument

    def __getattr__(self, attr):
        if attr in self.depends_on:
            return getattr(self.__instrument, attr)
        raise AttributeError(attr)

    def get_dimensional_calibrations(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize):
        """
        Subclasses should override this method
        """
        return [{}, {}]

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, last_scan_params=None) -> DataAndMetadata.DataAndMetadata:
        """
        Subclasses must override this method
        """
        raise NotImplementedError

    def get_total_counts(self, exposure_s: float) -> float:
        beam_current_pa = self.beam_current * 1E12
        e_per_pa = 6.242E18 / 1E12
        return beam_current_pa * e_per_pa * exposure_s * self._counts_per_electron

    def _get_binned_data(self, data: numpy.ndarray, binning_shape: Geometry.IntSize) -> numpy.ndarray:
        if binning_shape.height > 1:
            # do binning by taking the binnable area, reshaping last dimension into bins, and taking sum of those bins.
            data_T = data.T
            data = data_T[:(data_T.shape[0] // binning_shape.height) * binning_shape.height].reshape(data_T.shape[0], -1, binning_shape.height).sum(axis=-1).T
            if binning_shape.width > 1:
                data = data[:(data.shape[-1] // binning_shape.width) * binning_shape.width].reshape(data.shape[0], -1, binning_shape.width).sum(axis=-1)
        return data


class PoissonNoise:

    def __init__(self):
        self.enabled = True
        self.poisson_level = None

    def apply(self, input: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
        if self.enabled and self.poisson_level:
            rs = numpy.random.RandomState()  # use this to avoid blocking other calls to poisson
            return input + (rs.poisson(self.poisson_level, size=input.data.shape) - self.poisson_level)
        return input


class RonchigramCameraSimulator(CameraSimulator):
    depends_on = ["C10", "C12", "C21", "C23", "C30", "C32", "C34", "C34", "stage_position_m", "probe_state",
                  "probe_position", "live_probe_position", "features", "beam_shift_m", "is_blanked", "beam_current"]

    def __init__(self, instrument: "Instrument", ronchigram_shape: Geometry.IntSize, counts_per_electron: int, convergence_angle: float):
        super().__init__(instrument, "ronchigram", ronchigram_shape, counts_per_electron)
        self.__cached_frame = None
        max_defocus = instrument.max_defocus
        tv_pixel_angle = math.asin(instrument.stage_size_nm / (max_defocus * 1E9)) / ronchigram_shape.height
        self.__tv_pixel_angle = tv_pixel_angle
        self.__convergence_angle_rad = convergence_angle
        self.__max_defocus = max_defocus
        self.__data_scale = 1.0
        theta = tv_pixel_angle * ronchigram_shape.height / 2  # half angle on camera
        defocus_m = instrument.defocus_m
        self.__aberrations_controller = AberrationsController(ronchigram_shape.height, ronchigram_shape.width, theta, max_defocus, defocus_m)
        self.noise = PoissonNoise()

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, last_scan_params=None) -> DataAndMetadata.DataAndMetadata:
        # check if one of the arguments has changed since last call
        new_frame_settings = [readout_area, binning_shape, exposure_s, last_scan_params]
        if new_frame_settings != self._last_frame_settings:
            self._needs_recalculation = True
        self._last_frame_settings = new_frame_settings

        if self._needs_recalculation or self.__cached_frame is None:
            #print("recalculating frame")
            height = readout_area.height
            width = readout_area.width
            offset_m = self.stage_position_m
            full_fov_nm = abs(self.__max_defocus) * math.sin(self.__convergence_angle_rad) * 1E9
            fov_size_nm = Geometry.FloatSize(full_fov_nm * height / self._sensor_dimensions.height, full_fov_nm * width / self._sensor_dimensions.width)
            center_nm = Geometry.FloatPoint(full_fov_nm * (readout_area.center.y / self._sensor_dimensions.height- 0.5), full_fov_nm * (readout_area.center.x / self._sensor_dimensions.width - 0.5))
            size = Geometry.IntSize(height, width)
            data = numpy.zeros((height, width), numpy.float32)
            # features will be positive values; thickness can be simulated by subtracting the features from the
            # vacuum value. the higher the vacuum value, the thinner (i.e. less contribution from features).
            thickness_param = 100
            if not self.is_blanked:
                for feature in self.instrument.features:
                    feature.plot(data, offset_m, fov_size_nm, center_nm, size)
                data = thickness_param - data
            data = self._get_binned_data(data, binning_shape)

            if not self.is_blanked:
                probe_position = Geometry.FloatPoint(0.5, 0.5)
                if self.probe_state == "scanning":
                    probe_position = self.live_probe_position
                elif self.probe_state == "parked" and self.probe_position is not None:
                    probe_position = self.probe_position

                scan_offset = Geometry.FloatPoint()
                if last_scan_params is not None and probe_position is not None:
                    scan_size, scan_fov_size_nm, scan_center_nm = last_scan_params  # get these from last scan
                    scan_offset = Geometry.FloatPoint.make((probe_position[0]*scan_fov_size_nm[0] - scan_fov_size_nm[0]/2,
                                                            probe_position[1]*scan_fov_size_nm[1] - scan_fov_size_nm[1]/2))
                    scan_offset = scan_offset*1e-9

                theta = self.__tv_pixel_angle * self._sensor_dimensions.height / 2  # half angle on camera
                aberrations = dict()
                aberrations["height"] = data.shape[0]
                aberrations["width"] = data.shape[1]
                aberrations["theta"] = theta
                aberrations["c0a"] = self.beam_shift_m[1] + scan_offset[1]
                aberrations["c0b"] = self.beam_shift_m[0] + scan_offset[0]
                aberrations["c10"] = self.C10
                aberrations["c12a"] = self.C12[1]
                aberrations["c12b"] = self.C12[0]
                aberrations["c21a"] = self.C21[1]
                aberrations["c21b"] = self.C21[0]
                aberrations["c23a"] = self.C23[1]
                aberrations["c23b"] = self.C23[0]
                aberrations["c30"] = self.C30
                aberrations["c32a"] = self.C32[1]
                aberrations["c32b"] = self.C32[0]
                aberrations["c34a"] = self.C34[1]
                aberrations["c34b"] = self.C34[0]
                data = self.__aberrations_controller.apply(aberrations, data)

            intensity_calibration = Calibration.Calibration(units="counts")
            dimensional_calibrations = self.get_dimensional_calibrations(readout_area, binning_shape)

            self.__cached_frame = DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)
            self.__data_scale = self.get_total_counts(exposure_s) / (data.shape[0] * data.shape[1] * thickness_param)
            self._needs_recalculation = False

        self.noise.poisson_level = self.__data_scale
        return self.noise.apply(self.__cached_frame * self.__data_scale)

    def get_dimensional_calibrations(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize):
        height = readout_area.height if readout_area else self._sensor_dimensions[0]
        width = readout_area.width if readout_area else self._sensor_dimensions[1]
        scale_y = self.__tv_pixel_angle
        scale_x = self.__tv_pixel_angle
        offset_y = -scale_y * height * 0.5
        offset_x = -scale_x * width * 0.5
        dimensional_calibrations = [
            Calibration.Calibration(offset=offset_y, scale=scale_y, units="rad"),
            Calibration.Calibration(offset=offset_x, scale=scale_x, units="rad")
        ]
        return dimensional_calibrations


class EELSCameraSimulator(CameraSimulator):
    depends_on = ["is_slit_in", "probe_state", "probe_position", "live_probe_position", "is_blanked", "ZLPoffset",
                  "stage_position_m", "beam_shift_m", "features", "energy_offset_eV", "energy_per_channel_eV",
                  "beam_current"]

    def __init__(self, instrument: "Instrument", sensor_dimensions: Geometry.IntSize, counts_per_electron: int):
        super().__init__(instrument, "eels", sensor_dimensions, counts_per_electron)
        self.__cached_frame = None
        self.__data_scale = 1.0
        self.noise = PoissonNoise()

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, last_scan_params=None):
        """
        Features at the probe position will add plasmons and edges in addition to a ZLP.

        There are two inputs to this model: the beam current and the T/l (thickness / mean free path).

        The sum of the spectrum data should add up to the beam current (using counts per electron and conversion from
        electrons to amps).

        The natural log of the ratio of the sum of the spectrum to the sum of the ZLP should equal thickness / mean free
        path.

        The strategy is to have low level routines for adding the shapes of the ZLP (gaussian normal) and plasmons and
        edges (power law multiplied by integrated gaussian normal) and then scaling these shapes such that they satisfy
        the conditions above.

        A complication of this is that the specified energy range may not include the ZLP. So two spectrums are built:
        the one for caller and the one for reference. The reference one is used for calculating the scaling of the ZLP
        and edges, which are then applied to the spectrum for the caller.

        If we define the following values:
            z = sum/integration of unscaled ZLP gaussian
            f = sum/integration of unscaled plasmons/edges
            P = target count such that P / counts_per_electron matches beam current
            T = thickness (nm)
            L = lambda (mean_free_path_nm)
            T/l = thickness / lambda (mean free path)
        then we can solve for two unknowns:
            A = scale of ZLP
            B = scale of plasmons/edges
        using the two equations:
            Az + Bf = P (beam current)
            ln(P / Az) = T/l => P / Az = exp(T/l) (thickness = natural log of ratio of total counts to ZLP counts)
        solving:
            A = P / exp(T/l) / z
            B = (P - Az) / f
        """

        # check if one of the arguments has changed since last call
        new_frame_settings = [readout_area, binning_shape, exposure_s, last_scan_params]
        if new_frame_settings != self._last_frame_settings:
            self._needs_recalculation = True
        self._last_frame_settings = new_frame_settings

        if self._needs_recalculation or self.__cached_frame is None:
            data = numpy.zeros(tuple(self._sensor_dimensions), numpy.float)
            slit_attenuation = 10 if self.is_slit_in else 1
            intensity_calibration = Calibration.Calibration(units="counts")
            dimensional_calibrations = self.get_dimensional_calibrations(readout_area, binning_shape)
            probe_position = Geometry.FloatPoint(0.5, 0.5)
            if self.is_blanked:
                probe_position = None
            elif self.probe_state == "scanning":
                probe_position = self.live_probe_position
            elif self.probe_state == "parked" and self.probe_position is not None:
                probe_position = self.probe_position

            # typical thickness over mean free path (T/l) will be 0.5
            mean_free_path_nm = 100  # nm. (lambda values from back of Edgerton)
            thickness_per_layer_nm = 30  # nm

            # this is the number of pixel counts expected if the ZLP is visible in vacuum for the given exposure
            # and beam current (in get_total_counts).
            target_pixel_count = self.get_total_counts(exposure_s) / data.shape[0]

            # grab the specific calibration for the energy direction and offset by ZLPoffset
            used_calibration = dimensional_calibrations[1]
            used_calibration.offset = self.instrument.get_control("ZLPoffset").local_value

            if last_scan_params is not None and probe_position is not None:

                # make a buffer for the spectrum
                spectrum = numpy.zeros((data.shape[1], ), numpy.float)

                # configure a calibration for the reference spectrum. then plot the ZLP on the reference data. sum it to
                # get the zlp_pixel_count and the zlp_scale. this is the value to multiple zlp data by to scale it so
                # that it will produce the target pixel count. since we will be storing the spectra in a 2d array,
                # divide by the height of that array so that when it is summed, the value comes out correctly.
                zlp0_calibration = Calibration.Calibration(scale=used_calibration.scale, offset=-20)
                spectrum_ref = numpy.zeros((int(zlp0_calibration.convert_from_calibrated_value(-20 + 1000) - zlp0_calibration.convert_from_calibrated_value(-20)), ), numpy.float)
                plot_norm(spectrum_ref, 1.0, Calibration.Calibration(scale=used_calibration.scale, offset=-20), 0, 0.5 / slit_attenuation)
                zlp_ref_pixel_count = float(numpy.sum(spectrum_ref))

                # build the spectrum and reference spectrum by adding the features. the data is unscaled.
                spectrum_ref = numpy.zeros((int(zlp0_calibration.convert_from_calibrated_value(-20 + 1000) - zlp0_calibration.convert_from_calibrated_value(-20)), ), numpy.float)
                size, fov_size_nm, center_nm = last_scan_params  # get these from last scan
                offset_m = self.stage_position_m - self.beam_shift_m  # get this from current values
                feature_layer_count = 0
                for index, feature in enumerate(self.instrument.features):
                    if feature.intersects(offset_m, fov_size_nm, center_nm, Geometry.FloatPoint.make(probe_position)):
                        feature.plot_spectrum(spectrum, 1.0, used_calibration)
                        feature.plot_spectrum(spectrum_ref, 1.0, zlp0_calibration)
                        feature_layer_count += 1
                feature_pixel_count = max(numpy.sum(spectrum_ref), 0.01)

                # make the calculations for A, B (zlp_scale and feature_scale).
                thickness_factor = feature_layer_count * thickness_per_layer_nm / mean_free_path_nm
                zlp_scale = target_pixel_count / math.exp(thickness_factor) / zlp_ref_pixel_count
                feature_scale = (target_pixel_count - (target_pixel_count / math.exp(thickness_factor))) / feature_pixel_count
                # print(f"thickness_factor {thickness_factor}")

                # apply the scaling. spectrum holds the features at this point, but not the ZLP. just multiple by
                # feature_scale to make the feature part of the spectrum final. then plot the ZLP scaled by zlp_scale.
                spectrum *= feature_scale
                # print(f"sum {numpy.sum(spectrum) * data.shape[0]}")
                # print(f"zlp_ref_pixel_count {zlp_ref_pixel_count} feature_pixel_count {feature_pixel_count}")
                # print(f"zlp_scale {zlp_scale} feature_scale {feature_scale}")
                plot_norm(spectrum, zlp_scale, used_calibration, 0, 0.5 / slit_attenuation)
                # print(f"sum {numpy.sum(spectrum) * data.shape[0]}")
                # print(f"target_pixel_count {target_pixel_count}")

                # finally, store the spectrum into each row of the data
                data[:, ...] = spectrum

                # spectrum_pixel_count = float(numpy.sum(spectrum)) * data.shape[0]
                # print(f"z0 {zlp_ref_pixel_count * data.shape[0]} / {used_calibration.offset}")
                # print(f"beam current {self.instrument.beam_current * 1e12}pA")
                # print(f"current {spectrum_pixel_count / exposure_s / self.instrument.counts_per_electron / 6.242e18 * 1e12:#.2f}pA")
                # print(f"target {target_pixel_count}  actual {spectrum_pixel_count}")
                # print(f"s {spectrum_pixel_count} z {zlp_ref_pixel_count * zlp_scale * data.shape[0]}")
                # print(f"{math.log(spectrum_pixel_count / (zlp_ref_pixel_count * zlp_scale * data.shape[0]))} {thickness_factor}")

            data = self._get_binned_data(data, binning_shape)

            self.__cached_frame = DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)
            self.__data_scale = self.get_total_counts(exposure_s) / target_pixel_count / slit_attenuation / self._sensor_dimensions[0]
            self._needs_recalculation = False

        self.noise.poisson_level = self.__data_scale
        return self.noise.apply(self.__cached_frame)

    def get_dimensional_calibrations(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize):
        energy_offset_eV = self.energy_offset_eV
        # energy_offset_eV += random.uniform(-1, 1) * self.__energy_per_channel_eV * 5
        dimensional_calibrations = [
            Calibration.Calibration(),
            Calibration.Calibration(offset=energy_offset_eV, scale=self.energy_per_channel_eV, units="eV")
        ]
        return dimensional_calibrations


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
        self.__features = list()

        # define the STEM geometry limits
        self.stage_size_nm = 150
        self.max_defocus = 5000 / 1E9

        sample_size_m = Geometry.FloatSize(height=20, width=20) / 1E9
        feature_percentage = 0.3
        random_state = random.getstate()
        random.seed(1)
        energies = [[(68, 30), (855, 50), (872, 50)], [(29, 15), (1217, 50), (1248, 50)], [(1839, 5), (99, 50)]]  # Ni, Ge, Si
        plasmons = [20, 16.2, 16.8]
        for i in range(100):
            position_m = Geometry.FloatPoint(y=(2 * random.random() - 1.0) * sample_size_m.height, x=(2 * random.random() - 1.0) * sample_size_m.width)
            size_m = feature_percentage * Geometry.FloatSize(height=random.random() * sample_size_m.height, width=random.random() * sample_size_m.width)
            self.__features.append(Feature(position_m, size_m, energies[i%len(energies)], plasmons[i%len(plasmons)], 4))
        random.setstate(random_state)
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
            "ronchigram": RonchigramCameraSimulator(self, self.__ronchigram_shape, self.counts_per_electron, self.__convergence_angle_rad),
            "eels": EELSCameraSimulator(self, self.__eels_shape, self.counts_per_electron)
        }

    def close(self):
        for camera in self.__cameras.values():
            camera.close()
        self.__cameras = dict()

    def _get_camera_simulator(self, camera_id: str) -> CameraSimulator:
        return self.__cameras[camera_id]

    @property
    def features(self):
        return self.__features

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
        for feature in self.__features:
            feature.plot(data, offset_m, fov_size_nm + extra_nm, center_nm, used_size)
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
