from __future__ import annotations

# standard libraries
import copy
import typing
import math
import numpy
import numpy.typing
import scipy.ndimage
import scipy.stats

from nion.data import Calibration
from nion.data import DataAndMetadata

from nion.utils import Geometry

from . import CameraSimulator
from . import Noise

if typing.TYPE_CHECKING:
    from . import InstrumentDevice
    from nion.instrumentation import stem_controller

_NDArray = numpy.typing.NDArray[typing.Any]


class AberrationsController:
    """Track aberrations and apply them.

    All values are SI.

    Derived from code by Juan-Carlos Idrobo and Andy Lupini.
    """
    coefficient_names = (
        "c0a", "c0b", "c10", "c12a", "c12b", "c21a", "c21b", "c23a", "c23b", "c30", "c32a", "c32b", "c34a", "c34b",
        "c41a", "c41b", "c43a", "c43b", "c45a", "c45b", "c50", "c52a", "c52b", "c54a", "c54b", "c56a", "c56b", "c70"
    )

    def __init__(self, height: int, width: int, theta: float, max_defocus: float, defocus: float) -> None:
        self.__height = height
        self.__width = width
        self.__theta = theta
        self.__max_defocus = max_defocus
        self.__coefficients: typing.Dict[str, float] = dict()
        self.__intermediates: typing.Dict[str, _NDArray] = dict()  # functions of height/width/theta
        self.__chis: typing.Dict[str, _NDArray] = dict()  # chi's, functions of intermediate and coefficients
        self.__coefficients["c10"] = defocus
        self.__chi: typing.Optional[_NDArray] = None
        self.__c: typing.Optional[typing.List[float]] = None

    def apply(self, aberrations: typing.Mapping[str, typing.Union[int, float]], data: numpy.typing.NDArray[numpy.float32]) -> _NDArray:
        height = int(aberrations["height"])
        width = int(aberrations["width"])
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

        def get_i0ab() -> typing.Tuple[_NDArray, _NDArray]:
            i0a = self.__intermediates.get("c0a")
            i0b = self.__intermediates.get("c0b")
            if i0a is None or i0b is None:
                i0a, i0b = numpy.meshgrid(numpy.linspace(-theta, theta, width), numpy.linspace(-theta, theta, height))
                self.__intermediates["c0a"] = i0a
                self.__intermediates["c0b"] = i0b
            return i0a, i0b

        def get_i0a() -> _NDArray:
            return get_i0ab()[0]

        def get_i0b() -> _NDArray:
            return get_i0ab()[1]

        def get_i0a_squared() -> _NDArray:
            i0a_squared = self.__intermediates.get("c0a_squared")
            if i0a_squared is None:
                i0a_squared = get_i0a() ** 2
                self.__intermediates["c0a_squared"] = i0a_squared
            return i0a_squared

        def get_i0b_squared() -> _NDArray:
            i0b_squared = self.__intermediates.get("c0b_squared")
            if i0b_squared is None:
                i0b_squared = get_i0b() ** 2
                self.__intermediates["c0b_squared"] = i0b_squared
            return i0b_squared

        def get_iradius() -> _NDArray:
            ir = self.__intermediates.get("ir")
            if ir is None:
                ir = get_i0a_squared() + get_i0b_squared()
                self.__intermediates["ir"] = ir
            return ir

        def get_idiff_sq() -> _NDArray:
            ids = self.__intermediates.get("ids")
            if ids is None:
                ids = get_i0a_squared() - get_i0b_squared()
                self.__intermediates["ids"] = ids
            return ids

        def get_intermediate(coefficient_name: str) -> _NDArray:
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
                    intermediate = get_i0b() * (3 * get_i0a_squared() - get_i0b_squared()) / 3
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
                    intermediate = get_iradius() * (get_i0a() * get_idiff_sq() - 2 * get_i0a() * get_i0b() ** 2) / 5
                elif coefficient_name == "c43b":
                    intermediate = get_iradius() * (get_i0b() * get_idiff_sq() + 2 * get_i0b() * get_i0a() ** 2) / 5
                elif coefficient_name == "c45a":
                    intermediate = (get_i0a() * get_idiff_sq() ** 2 - 4 * get_i0a() * get_idiff_sq() * get_i0b() ** 2 - 4 * get_i0a() ** 3 * get_i0b() ** 2) / 5
                elif coefficient_name == "c45b":
                    # this type: ignore is inexplicably necessary; seems like bug in numpy.typing or mypy. try removing to see if it passes typing tests.
                    intermediate = (get_i0b() * get_idiff_sq() ** 2 + 4 * get_i0b() * get_idiff_sq() * get_i0a() ** 2 - 4 * get_i0a() ** 2 * get_i0b() ** 3) / 5  # type: ignore
                elif coefficient_name == "c50":
                    intermediate = 8 * get_intermediate("c10") ** 3 / 6
                elif coefficient_name == "c52a":
                    intermediate = get_iradius() ** 2 * get_idiff_sq() / 6
                elif coefficient_name == "c52b":
                    intermediate = get_iradius() ** 2 * get_intermediate("c12b") / 3
                elif coefficient_name == "c54a":
                    intermediate = get_iradius() * (get_idiff_sq() ** 2 - 4 * get_intermediate("c12b") ** 2) / 6
                elif coefficient_name == "c54b":
                    intermediate = 2 * get_iradius() * get_idiff_sq() * get_intermediate("c12b") / 3
                elif coefficient_name == "c56a":
                    intermediate = get_idiff_sq() ** 3 / 6 - 2 * get_i0a() ** 2 * get_i0b() ** 2 * get_idiff_sq()
                elif coefficient_name == "c56b":
                    intermediate = get_intermediate("c12b") * get_idiff_sq() ** 2 - (4 * get_i0a() ** 3 * get_i0b() ** 3) / 3
                elif coefficient_name == "c70":
                    intermediate = 2 * get_intermediate("c10") ** 4
            assert intermediate is not None
            return intermediate

        def get_chi(coefficient_name: str) -> typing.Optional[_NDArray]:
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
            max_chi0 = self.__max_defocus * theta * theta
            max_chi1 = self.__max_defocus * theta * theta * ((1 - 1 / width) * (1 - 1 / width) + (1 - 1 / height) * (1 - 1 / height)) / 2
            max_chi = max_chi0 - max_chi1
            scale_y = height / 2 / max_chi
            scale_x = width / 2 / max_chi
            self.__c = [scale_y * grad_chi[0] + height/2, scale_x * grad_chi[1] + width/2]

        # note, the scaling factor of 2pi/wavelength has been removed from chi since it cancels out.

        if self.__c is not None:
            # scale the offsets so that at max defocus, the coordinates cover the entire area of data.
            return scipy.ndimage.map_coordinates(data, self.__c, order=1)  # type: ignore

        return numpy.zeros((height, width))


def ellipse_radius(polar_angle: typing.Union[float, _NDArray], a: float, b: float, rotation: float) -> typing.Union[float, _NDArray]:
    """
    Returns the radius of a point lying on an ellipse with the given parameters. The ellipse is described in polar
    coordinates here, which makes it easy to incorporate a rotation.

    Parameters
    -----------
    polar_angle : float or _NDArray
                  Polar angle of a point to which the corresponding radius should be calculated (rad).
    a : float
        Length of the major half-axis of the ellipse.
    b : float
        Length of the minor half-axis of the ellipse.
    rotation : Rotation of the ellipse with respect to the x-axis (rad). Counter-clockwise is positive.

    Returns
    --------
    radius : float or _NDArray
             Radius of a point lying on an ellipse with the given parameters.
    """

    return a * b / numpy.sqrt((b * numpy.cos(polar_angle + rotation)) ** 2 + (a * numpy.sin(polar_angle + rotation)) ** 2)  # type: ignore


def draw_ellipse(image: _NDArray, ellipse: typing.Tuple[float, float, float, float, float], *, color: typing.Any = 1.0) -> None:
    """
    Draws an ellipse on a 2D-array.

    Parameters
    ----------
    image : array
            The array on which the ellipse will be drawn. Note that the data will be modified in place.
    ellipse : tuple
              A tuple describing an ellipse with the same moments as the aperture. The values must be (in this order):
              [0] The y-coordinate of the center.
              [1] The x-coordinate of the center.
              [2] The length of the major half-axis
              [3] The length of the minor half-axis
              [4] The rotation of the ellipse in rad.
    color : optional
            The color to which the pixels inside the given ellipse will be set. Note that `color` will be cast to the
            type of `image` automatically. If this is not possible, an exception will be raised. The default is 1.0.

    Returns
    --------
    None
    """
    shape = image.shape
    assert len(shape) == 2, 'Can only draw an ellipse on a 2D-array.'
    # coords = np.mgrid[-shape[0]/2:shape[0]/2:shape[0]*1j, -shape[1]/2:shape[1]/2:shape[1]*1j]
    top = max(int(ellipse[0] - ellipse[2]), 0)
    left = max(int(ellipse[1] - ellipse[2]), 0)
    bottom = min(int(ellipse[0] + ellipse[2]) + 1, shape[0])
    right = min(int(ellipse[1] + ellipse[2]) + 1, shape[1])
    coords = numpy.mgrid[top - ellipse[0]:bottom - ellipse[0], left - ellipse[1]:right - ellipse[1]]  # type: ignore
    # coords[0] -= ellipse[0]
    # coords[1] -= ellipse[1]
    radii = numpy.sqrt(numpy.sum(coords**2, axis=0))
    polar_angles = numpy.arctan2(coords[0], coords[1])
    ellipse_radii = ellipse_radius(polar_angles, *ellipse[2:])
    image[top:bottom, left:right][radii < ellipse_radii] = color


class RonchigramCameraSimulator(CameraSimulator.CameraSimulator):
    depends_on = ["C10Control", "C12Control", "C21Control", "C23Control", "C30Control", "C32Control", "C34Control",
                  "C34Control", "stage_position_m", "probe_state", "probe_position", "features",
                  "beam_shift_m", "is_blanked", "BeamCurrent", "CAperture", "ApertureRound", "S_VOA", "ConvergenceAngle"]

    def __init__(self, instrument: InstrumentDevice.Instrument, ronchigram_shape: Geometry.IntSize, counts_per_electron: int, stage_size_nm: float) -> None:
        super().__init__(instrument, "ronchigram", ronchigram_shape, counts_per_electron)
        self.__cached_frame: typing.Optional[DataAndMetadata.DataAndMetadata] = None
        max_defocus = instrument.max_defocus
        self.__stage_size_nm = stage_size_nm
        self.__data_scale = 1.0
        self.__aperture_ellipse: typing.Optional[typing.Tuple[float, float, float, float, float]] = None
        self.__aperture_mask = None
        theta = self._tv_pixel_angle * ronchigram_shape.height / 2  # half angle on camera
        defocus_m = instrument.defocus_m
        self.__aberrations_controller = AberrationsController(ronchigram_shape.height, ronchigram_shape.width, theta, max_defocus, defocus_m)
        self.noise = Noise.PoissonNoise()

    @property
    def _tv_pixel_angle(self) -> float:
        instrument = self.instrument
        return math.asin(instrument.stage_size_nm / (instrument.max_defocus * 1E9)) / self._camera_shape.height

    @property
    def _max_defocus(self) -> float:
        return self.instrument.max_defocus

    def _draw_aperture(self, frame_data: _NDArray, binning_shape: Geometry.IntSize, enlarge_by: float = 0.0) -> None:
        # TODO handle asymmetric binning
        binning = binning_shape[0]
        position = self.instrument.GetVal2D("CAperture")
        aperture_round = self.instrument.GetVal2D("ApertureRound")
        shape = frame_data.shape
        ellipse_center = 0.5 * shape[0] + position.y / self._tv_pixel_angle / binning, 0.5 * shape[1] + position.x / self._tv_pixel_angle / binning
        excentricity = math.sqrt(aperture_round[0]**2 + aperture_round[1]**2)
        # adapt excentricity so that control behaves linearly and is defined for all values
        # this is the modified inverse function of the calculation of the major half-axis a
        excentricity = math.sqrt(1-1/(1+abs(excentricity))**4)
        direction = numpy.arctan2(aperture_round[0], aperture_round[1])
        # Calculate a and b (the ellipse half-axes) from excentricity. Keep ellipse area constant
        convergence_angle = self.instrument.GetVal("ConvergenceAngle") * (1 + enlarge_by)
        convergence_angle_pixels = convergence_angle / self._tv_pixel_angle / binning
        a = math.sqrt(convergence_angle_pixels**2 / math.sqrt(1 - excentricity**2))
        b = convergence_angle_pixels**2 / a
        self.__aperture_ellipse = ellipse_center + (a, b, direction)
        aperture_mask = numpy.zeros_like(frame_data)
        draw_ellipse(aperture_mask, self.__aperture_ellipse)
        frame_data *= aperture_mask

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, scan_context: stem_controller.ScanContext, parked_probe_position: typing.Optional[Geometry.FloatPoint]) -> DataAndMetadata.DataAndMetadata:
        frame_settings = self._get_frame_settings(readout_area, binning_shape, exposure_s, scan_context, parked_probe_position)
        if frame_settings != self._last_frame_settings:
            self._needs_recalculation = True
            self._last_frame_settings = frame_settings

        if self._needs_recalculation or self.__cached_frame is None:
            # print("recalculating frame")
            height = readout_area.height
            width = readout_area.width
            offset_m = self.instrument.stage_position_m
            # full_fov_nm = abs(self.instrument.GetVal("C10Control")) * self._tv_pixel_angle * self._sensor_dimensions.height * 1e9
            full_fov_nm = self.__stage_size_nm
            fov_size_nm = Geometry.FloatSize(full_fov_nm * height / self._sensor_dimensions.height, full_fov_nm * width / self._sensor_dimensions.width)
            center_nm = Geometry.FloatPoint(
                full_fov_nm * (readout_area.center.y / self._sensor_dimensions.height - 0.5),
                full_fov_nm * (readout_area.center.x / self._sensor_dimensions.width - 0.5))
            size = Geometry.IntSize(height, width)
            data: numpy.typing.NDArray[numpy.float32] = numpy.zeros((height, width), numpy.float32)
            # features will be positive values; thickness can be simulated by subtracting the features from the
            # vacuum value. the higher the vacuum value, the thinner (i.e. less contribution from features).
            thickness_param = 100
            if not self.instrument.is_blanked:
                self.instrument.sample.plot_features(data, offset_m, fov_size_nm, Geometry.FloatPoint(), center_nm, size)
                data = thickness_param - data
            data = self._get_binned_data(data, binning_shape)

            if not self.instrument.is_blanked:
                scan_offset = Geometry.FloatPoint()
                scan_context_fov_nm = scan_context.fov_size_nm
                if scan_context.is_valid and frame_settings.current_probe_position is not None and scan_context_fov_nm is not None:
                    scan_offset = Geometry.FloatPoint(
                        y=frame_settings.current_probe_position[0] * scan_context_fov_nm[0] - scan_context_fov_nm[0] / 2,
                        x=frame_settings.current_probe_position[1] * scan_context_fov_nm[1] - scan_context_fov_nm[1] / 2)
                    scan_offset = scan_offset*1e-9

                theta = self._tv_pixel_angle * self._sensor_dimensions.height / 2  # half angle on camera
                aberrations: typing.Dict[str, typing.Union[float, int]] = dict()
                aberrations["height"] = data.shape[0]
                aberrations["width"] = data.shape[1]
                aberrations["theta"] = theta
                aberrations["c0a"] = self.instrument.GetVal2D("beam_shift_m").x + scan_offset[1]
                aberrations["c0b"] = self.instrument.GetVal2D("beam_shift_m").y + scan_offset[0]
                aberrations["c10"] = self.instrument.GetVal("C10Control")
                aberrations["c12a"] = self.instrument.GetVal2D("C12Control").x
                aberrations["c12b"] = self.instrument.GetVal2D("C12Control").y
                aberrations["c21a"] = self.instrument.GetVal2D("C21Control").x
                aberrations["c21b"] = self.instrument.GetVal2D("C21Control").y
                aberrations["c23a"] = self.instrument.GetVal2D("C23Control").x
                aberrations["c23b"] = self.instrument.GetVal2D("C23Control").y
                aberrations["c30"] = self.instrument.GetVal("C30Control")
                aberrations["c32a"] = self.instrument.GetVal2D("C32Control").x
                aberrations["c32b"] = self.instrument.GetVal2D("C32Control").y
                aberrations["c34a"] = self.instrument.GetVal2D("C34Control").x
                aberrations["c34b"] = self.instrument.GetVal2D("C34Control").y
                data = self.__aberrations_controller.apply(aberrations, data)
                if self.instrument.GetVal("S_VOA") > 0:
                    self._draw_aperture(data, binning_shape)
                elif self.instrument.GetVal("S_MOA") > 0:
                    self._draw_aperture(data, binning_shape, enlarge_by=0.1)

            intensity_calibration = Calibration.Calibration(units="counts")
            dimensional_calibrations = self.get_dimensional_calibrations(readout_area, binning_shape)

            self.__cached_frame = DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)
            self.__data_scale = self.get_total_counts(exposure_s) / (data.shape[0] * data.shape[1] * thickness_param)
            self._needs_recalculation = False

        self.noise.poisson_level = self.__data_scale
        assert self.__cached_frame
        return self.noise.apply(self.__cached_frame * self.__data_scale)

    def get_dimensional_calibrations(self, readout_area: typing.Optional[Geometry.IntRect], binning_shape: typing.Optional[Geometry.IntSize]) -> typing.Sequence[Calibration.Calibration]:
        height = readout_area.height if readout_area else self._sensor_dimensions[0]
        width = readout_area.width if readout_area else self._sensor_dimensions[1]
        scale_y = self._tv_pixel_angle
        scale_x = self._tv_pixel_angle
        offset_y = -scale_y * height * 0.5
        offset_x = -scale_x * width * 0.5
        dimensional_calibrations = [
            Calibration.Calibration(offset=offset_y, scale=scale_y, units="rad"),
            Calibration.Calibration(offset=offset_x, scale=scale_x, units="rad")
        ]
        return dimensional_calibrations
