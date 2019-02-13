# standard libraries
import math
import numpy
import scipy.ndimage.interpolation
import scipy.stats

from nion.data import Calibration
from nion.data import DataAndMetadata

from nion.utils import Geometry

from . import CameraSimulator
from . import Noise


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


class RonchigramCameraSimulator(CameraSimulator.CameraSimulator):
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
        self.noise = Noise.PoissonNoise()

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
                for feature in self.instrument.sample.features:
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
