"""
Useful references:
    http://www.rodenburg.org/guide/index.html
    http://www.ammrf.org.au/myscope/
"""

# standard libraries
import math
import numpy
import os
import random
import scipy.ndimage.interpolation
import scipy.stats
import threading
import typing

from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata

from nion.utils import Event
from nion.utils import Geometry

from nion.instrumentation import stem_controller


def plot_powerlaw(data: numpy.ndarray, multiplier: float, energy_calibration: Calibration.Calibration, offset_eV: float, onset_eV: float) -> None:
    # calculate the range
    # 1 represents 0eV, 0 represents 4000eV
    # TODO: sub-pixel accuracy
    data_range = [0, data.shape[0]]
    energy_range_eV = [energy_calibration.convert_to_calibrated_value(data_range[0]), energy_calibration.convert_to_calibrated_value(data_range[1])]
    envelope = scipy.stats.norm(loc=offset_eV, scale=onset_eV).cdf(numpy.linspace(energy_range_eV[0], energy_range_eV[1], data.shape[0]))
    powerlaw = scipy.stats.powerlaw(4, loc=0, scale=4000)
    if energy_range_eV[1] - offset_eV > 0:
        data += envelope * multiplier * powerlaw.pdf(numpy.linspace(energy_range_eV[1], energy_range_eV[0], data_range[1])) / powerlaw.pdf(energy_range_eV[1] - offset_eV)


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

    def plot(self, data: numpy.ndarray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> None:
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


def _relativeFile(filename):
    dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(dir, filename)


class Instrument(stem_controller.STEMController):

    def __init__(self, instrument_id: str):
        super().__init__()
        self.priority = 20
        self.instrument_id = instrument_id
        self.__camera_frame_event = threading.Event()
        self.__features = list()
        stage_size_nm = 150
        sample_size_m = Geometry.FloatSize(height=20, width=20) / 1E9
        feature_percentage = 0.3
        random_state = random.getstate()
        random.seed(1)
        energies = [[(68, 30), (855, 50), (872, 50)], [(29, 15), (1217, 50), (1248, 50)], [(1839, 5), (99, 50)]]  # Ni, Ge, Si
        plasmons = [20, 16.2, 16.8]
        for i in range(100):
            position_m = Geometry.FloatPoint(y=(2 * random.random() - 1.0) * sample_size_m.height, x=(2 * random.random() - 1.0) * sample_size_m.width)
            size_m = feature_percentage * Geometry.FloatSize(height=random.random() * sample_size_m.height, width=random.random() * sample_size_m.width)
            self.__features.append(Feature(position_m, size_m, energies[i%len(energies)], plasmons[i%len(energies)], 4))
        random.setstate(random_state)
        self.__stage_position_m = Geometry.FloatPoint()
        self.__beam_shift_m = Geometry.FloatPoint()
        self.__convergence_angle_rad = 30 / 1000
        self.__defocus_m = 500 / 1E9
        self.__c12 = Geometry.FloatPoint()
        self.__c21 = Geometry.FloatPoint()
        self.__c23 = Geometry.FloatPoint()
        self.__c30 = 0.0
        self.__c32 = Geometry.FloatPoint()
        self.__c34 = Geometry.FloatPoint()
        self.__slit_in = False
        self.__energy_offset_eV = -20
        self.__energy_per_channel_eV = 0.5
        self.__voltage = 100000
        self.__beam_current = 200E-12  # 200 pA
        self.__blanked = False
        self.property_changed_event = Event.Event()
        self.__ronchigram_shape = Geometry.IntSize(2048, 2048)
        self.__max_defocus = 5000 / 1E9
        self.__tv_pixel_angle = math.asin(stage_size_nm / (self.__max_defocus * 1E9)) / self.__ronchigram_shape.height
        self.__eels_shape = Geometry.IntSize(256, 1024)
        self.__last_scan_params = None
        self.live_probe_position = None
        theta = self.__tv_pixel_angle * self.__ronchigram_shape.height / 2  # half angle on camera
        self.__aberrations_controller = AberrationsController(self.__ronchigram_shape[0], self.__ronchigram_shape[1], theta, self.__max_defocus, self.__defocus_m)

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
        used_size = size + Geometry.IntSize(height=extra, width=extra)
        data = numpy.zeros((used_size.height, used_size.width), numpy.float32)
        for feature in self.__features:
            feature.plot(data, offset_m, fov_size_nm, center_nm, used_size)
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

    def __get_binned_data(self, data: numpy.ndarray, binning_shape: Geometry.IntSize) -> numpy.ndarray:
        if binning_shape.height > 1:
            # do binning by taking the binnable area, reshaping last dimension into bins, and taking sum of those bins.
            data_T = data.T
            data = data_T[:(data_T.shape[0] // binning_shape.height) * binning_shape.height].reshape(data_T.shape[0], -1, binning_shape.height).sum(axis=-1).T
            if binning_shape.width > 1:
                data = data[:(data.shape[-1] // binning_shape.width) * binning_shape.width].reshape(data.shape[0], -1, binning_shape.width).sum(axis=-1)
        return data

    @property
    def counts_per_electron(self):
        return 40

    def get_electrons_per_pixel(self, pixel_count: int, exposure_s: float) -> float:
        beam_current_pa = self.__beam_current * 1E12
        e_per_pa = 6.242E18 / 1E12
        beam_e = beam_current_pa * e_per_pa
        e_per_pixel_per_second = beam_e / pixel_count
        return e_per_pixel_per_second * exposure_s

    def get_total_counts(self, exposure_s: float) -> float:
        beam_current_pa = self.__beam_current * 1E12
        e_per_pa = 6.242E18 / 1E12
        return beam_current_pa * e_per_pa * exposure_s * self.counts_per_electron

    def get_camera_data(self, camera_type: str, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float) -> DataAndMetadata.DataAndMetadata:
        if camera_type == "ronchigram":
            height = readout_area.height
            width = readout_area.width
            offset_m = self.stage_position_m
            full_fov_nm = abs(self.__max_defocus) * math.sin(self.__convergence_angle_rad) * 1E9
            fov_size_nm = Geometry.FloatSize(full_fov_nm * height / self.__ronchigram_shape.height, full_fov_nm * width / self.__ronchigram_shape.width)
            center_nm = Geometry.FloatPoint(full_fov_nm * (readout_area.center.y / self.__ronchigram_shape.height- 0.5), full_fov_nm * (readout_area.center.x / self.__ronchigram_shape.width - 0.5))
            size = Geometry.IntSize(height, width)
            data = numpy.zeros((height, width), numpy.float32)
            feature_pixel_count = 0
            for feature in self.__features:
                feature_pixel_count += feature.plot(data, offset_m, fov_size_nm, center_nm, size)
            # features will be positive values; thickness can be simulated by subtracting the features from the
            # vacuum value. the higher the vacuum value, the thinner (i.e. less contribution from features).
            thickness_param = 100
            data = thickness_param - data
            data = self.__get_binned_data(data, binning_shape)
            data_scale = self.get_total_counts(exposure_s) / (data.shape[0] * data.shape[1] * thickness_param)
            rs = numpy.random.RandomState()  # use this to avoid blocking other calls to poisson
            data = data * data_scale + rs.poisson(data_scale, size=data.shape) - data_scale

            theta = self.__tv_pixel_angle * self.__ronchigram_shape.height / 2  # half angle on camera
            aberrations = dict()
            aberrations["height"] = data.shape[0]
            aberrations["width"] = data.shape[1]
            aberrations["theta"] = theta
            aberrations["c0a"] = self.beam_shift_m[1]
            aberrations["c0b"] = self.beam_shift_m[0]
            aberrations["c10"] = self.__defocus_m
            aberrations["c12a"] = self.__c12[1]
            aberrations["c12b"] = self.__c12[0]
            aberrations["c21a"] = self.__c21[1]
            aberrations["c21b"] = self.__c21[0]
            aberrations["c23a"] = self.__c23[1]
            aberrations["c23b"] = self.__c23[0]
            aberrations["c30"] = self.__c30
            aberrations["c32a"] = self.__c32[1]
            aberrations["c32b"] = self.__c32[0]
            aberrations["c34a"] = self.__c34[1]
            aberrations["c34b"] = self.__c34[0]
            data = self.__aberrations_controller.apply(aberrations, data)

            intensity_calibration = Calibration.Calibration(units="counts")
            dimensional_calibrations = self.get_camera_dimensional_calibrations(camera_type, readout_area, binning_shape)

            return DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)
        else:
            data = numpy.zeros(tuple(self.__eels_shape), numpy.float)
            slit_attenuation = 10 if self.__slit_in else 1
            intensity_calibration = Calibration.Calibration(units="counts")
            dimensional_calibrations = self.get_camera_dimensional_calibrations(camera_type, readout_area, binning_shape)
            probe_position = self.probe_position
            if self.__blanked:
                probe_position = None
            elif self.probe_state == "parked":
                pass
            elif self.probe_state == "scanning":
                probe_position = self.live_probe_position
            if probe_position is not None:
                spectrum = numpy.zeros((data.shape[1], ), numpy.float)
                plot_norm(spectrum, 1.0, dimensional_calibrations[1], 0, 0.5 / slit_attenuation)
                size, fov_size_nm, center_nm = self.__last_scan_params  # get these from last scan
                offset_m = self.stage_position_m - self.beam_shift_m  # get this from current values
                mean_free_path = 100  # nm. (lambda values from back of Edgerton)
                thickness = 50  # nm
                # T/lambda = 0.25 0.5 typical values OLK
                # ZLP to first plasmon (areas, total count) is T/lambda.
                # Each plasmon is also reduce by T/L
                for index, feature in enumerate(self.__features):
                    if feature.intersects(offset_m, fov_size_nm, center_nm, Geometry.FloatPoint.make(probe_position)):
                        feature.plot_spectrum(spectrum, 1.0 / 10, dimensional_calibrations[1])
                feature_pixel_count = max(numpy.sum(spectrum), 0.01)
                data[:, ...] = spectrum
            else:
                feature_pixel_count = 1
            data = self.__get_binned_data(data, binning_shape)
            data_scale = self.get_total_counts(exposure_s) / feature_pixel_count / slit_attenuation / self.__eels_shape[0]
            poisson_level = data_scale + 5  # camera noise
            rs = numpy.random.RandomState()  # use this to avoid blocking other calls to poisson
            data = data * data_scale + (rs.poisson(poisson_level, size=data.shape) - poisson_level)
            return DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)

    def get_camera_dimensional_calibrations(self, camera_type: str, readout_area: Geometry.IntRect = None, binning_shape: Geometry.IntSize = None):
        if camera_type == "ronchigram":
            binning_shape = binning_shape if binning_shape else Geometry.IntSize(1, 1)
            height = readout_area.height if readout_area else self.__ronchigram_shape[0]
            width = readout_area.width if readout_area else self.__ronchigram_shape[1]
            full_fov_nm = abs(self.__defocus_m) * math.sin(self.__tv_pixel_angle * self.__ronchigram_shape.height) * 1E9
            fov_nm = Geometry.FloatSize(full_fov_nm * height / self.__ronchigram_shape.height, full_fov_nm * width / self.__ronchigram_shape.width)
            scale_y = binning_shape[0] * fov_nm[0] / height
            scale_x = binning_shape[1] * fov_nm[1] / width
            offset_y = -scale_y * height * 0.5
            offset_x = -scale_x * width * 0.5
            dimensional_calibrations = [
                Calibration.Calibration(offset=offset_y, scale=scale_y, units="nm"),
                Calibration.Calibration(offset=offset_x, scale=scale_x, units="nm")
            ]
            return dimensional_calibrations
        if camera_type == "eels":
            dimensional_calibrations = [
                Calibration.Calibration(),
                Calibration.Calibration(offset=self.__energy_offset_eV, scale=self.__energy_per_channel_eV, units="eV")
            ]
            return dimensional_calibrations
        return [{}, {}]

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
        return self.__defocus_m

    @defocus_m.setter
    def defocus_m(self, value: float) -> None:
        self.__defocus_m = value
        self.property_changed_event.fire("defocus_m")

    @property
    def c12(self) -> Geometry.FloatPoint:
        return self.__c12

    @c12.setter
    def c12(self, value: Geometry.FloatPoint) -> None:
        self.__c12 = value
        self.property_changed_event.fire("c12")

    @property
    def c21(self) -> Geometry.FloatPoint:
        return self.__c21

    @c21.setter
    def c21(self, value: Geometry.FloatPoint) -> None:
        self.__c21 = value
        self.property_changed_event.fire("c21")

    @property
    def c23(self) -> Geometry.FloatPoint:
        return self.__c23

    @c23.setter
    def c23(self, value: Geometry.FloatPoint) -> None:
        self.__c23 = value
        self.property_changed_event.fire("c23")

    @property
    def c30(self) -> float:
        return self.__c30

    @c30.setter
    def c30(self, value: float) -> None:
        self.__c30 = value
        self.property_changed_event.fire("c30")

    @property
    def c32(self) -> Geometry.FloatPoint:
        return self.__c32

    @c32.setter
    def c32(self, value: Geometry.FloatPoint) -> None:
        self.__c32 = value
        self.property_changed_event.fire("c32")

    @property
    def c34(self) -> Geometry.FloatPoint:
        return self.__c34

    @c34.setter
    def c34(self, value: Geometry.FloatPoint) -> None:
        self.__c34 = value
        self.property_changed_event.fire("c34")

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
        return self.__energy_offset_eV

    @energy_offset_eV.setter
    def energy_offset_eV(self, value: float) -> None:
        self.__energy_offset_eV = value
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
        elif s == "C10":
            return True, self.defocus_m
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
        elif s == "C10":
            self.defocus_m = val
            return True
        return False

    def SetValWait(self, s: str, val: float, timeout_ms: int) -> bool:
        return self.SetVal(s, val)

    def SetValAndConfirm(self, s: str, val: float, tolfactor: float, timeout_ms: int) -> bool:
        return self.SetVal(s, val)

    def SetValDelta(self, s: str, delta: float) -> bool:
        return self.SetVal(s, self.GetVal(s) + delta)

    def InformControl(self, s: str, val: float) -> bool:
        return self.SetVal(s, val)

    def change_stage_position(self, *, dy: int=None, dx: int=None):
        """Shift the stage by dx, dy (meters). Do not wait for confirmation."""
        self.stage_position_m += Geometry.FloatPoint(y=-dy, x=-dx)

    def change_pmt_gain(self, pmt_type: stem_controller.PMTType, *, factor: float) -> None:
        """Change specified PMT by factor. Do not wait for confirmation."""
        pass
