# standard libraries
import math
import numpy
import os
import random
import scipy.stats
import threading
import typing

from nion.data import Calibration
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

    def plot_spectrum(self, data: numpy.ndarray, multiplier: float, energy_calibration: Calibration.Calibration) -> None:
        for edge_eV, onset_eV in self.edges:
            strength = multiplier * 0.1
            plot_powerlaw(data, strength, energy_calibration, edge_eV, onset_eV)
        for n in range(1, self.plurality + 1):
            plot_norm(data, multiplier / math.factorial(n), energy_calibration, self.plasmon_eV * n, math.sqrt(self.plasmon_eV))


def _relativeFile(filename):
    dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(dir, filename)


class Instrument(stem_controller.STEMController):

    def __init__(self):
        super().__init__()
        self.__camera_frame_event = threading.Event()
        self.__features = list()
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
        self.__slit_in = False
        self.__energy_offset_eV = 20
        self.__energy_per_channel_eV = 0.5
        self.__blanked = False
        self.property_changed_event = Event.Event()
        self.__ronchigram_shape = Geometry.IntSize(2048, 2048)
        self.__eels_shape = Geometry.IntSize(256, 1024)
        self.__last_scan_params = None
        self.live_probe_position = None

    def trigger_camera_frame(self) -> None:
        self.__camera_frame_event.set()

    def wait_for_camera_frame(self, timeout: float) -> bool:
        result = self.__camera_frame_event.wait(timeout)
        self.__camera_frame_event.clear()
        return result

    def get_scan_data(self, frame_parameters, channel) -> numpy.ndarray:
        height = frame_parameters.size[0]
        width = frame_parameters.size[1]
        offset_m = self.stage_position_m - self.beam_shift_m
        fov_nm = Geometry.FloatSize(frame_parameters.fov_nm, frame_parameters.fov_nm)
        center_nm = Geometry.FloatPoint.make(frame_parameters.center_nm)
        size = Geometry.IntSize(height, width)
        data = numpy.zeros((height, width), numpy.float32)
        for feature in self.__features:
            feature.plot(data, offset_m, fov_nm, center_nm, size)
        noise_factor = 0.3
        data = (data + numpy.random.randn(height, width) * noise_factor) * frame_parameters.pixel_time_us
        self.__last_scan_params = size, fov_nm, center_nm
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
            data = data_T[:(data_T.shape[-1] // binning_shape.height) * binning_shape.height].reshape(data_T.shape[0], -1, binning_shape.height).sum(axis=-1).T
            if binning_shape.width > 1:
                data = data[:(data.shape[-1] // binning_shape.width) * binning_shape.width].reshape(data.shape[0], -1, binning_shape.width).sum(axis=-1)
        return data

    def get_electrons_per_pixel(self, pixel_count: int, exposure_s: float) -> float:
        beam_current_pa = 40
        e_per_pa = 6.2E18 / 1E12
        beam_e = beam_current_pa * e_per_pa
        e_per_pixel_per_second = beam_e / pixel_count
        return e_per_pixel_per_second * exposure_s

    def get_camera_data(self, camera_type: str, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float) -> DataAndMetadata.DataAndMetadata:
        if camera_type == "ronchigram":
            height = readout_area.height
            width = readout_area.width
            offset_m = self.stage_position_m - self.beam_shift_m
            full_fov_nm = abs(self.__defocus_m) * math.sin(self.__convergence_angle_rad) * 1E9
            fov_nm = Geometry.FloatSize(full_fov_nm * height / self.__ronchigram_shape.height, full_fov_nm * width / self.__ronchigram_shape.width)
            center_nm = Geometry.FloatPoint(full_fov_nm * (readout_area.center.y / self.__ronchigram_shape.height- 0.5), full_fov_nm * (readout_area.center.x / self.__ronchigram_shape.width - 0.5))
            size = Geometry.IntSize(height, width)
            data = numpy.zeros((height, width), numpy.float32)
            for feature in self.__features:
                feature.plot(data, offset_m, fov_nm, center_nm, size)
            data = self.__get_binned_data(data, binning_shape)
            e_per_pixel = self.get_electrons_per_pixel(data.shape[0] * data.shape[1], exposure_s) * binning_shape[1] * binning_shape[0]
            rs = numpy.random.RandomState()  # use this to avoid blocking other calls to poisson
            data = data * e_per_pixel + rs.poisson(e_per_pixel, size=data.shape) - e_per_pixel
            intensity_calibration = Calibration.Calibration(units="e")
            dimensional_calibrations = self.get_camera_dimensional_calibrations(camera_type, readout_area, binning_shape)
            return DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)
        else:
            data = numpy.zeros(tuple(self.__eels_shape), numpy.float)
            slit_attenuation = 10 if self.__slit_in else 1
            e_per_pixel = self.get_electrons_per_pixel(data.shape[0] * data.shape[1], exposure_s) * binning_shape[1] * binning_shape[0] / slit_attenuation
            intensity_calibration = Calibration.Calibration(units="e")
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
                plot_norm(spectrum, e_per_pixel, dimensional_calibrations[1], 0, 0.5 / slit_attenuation)
                size, fov_nm, center_nm = self.__last_scan_params  # get these from last scan
                offset_m = self.stage_position_m - self.beam_shift_m  # get this from current values
                for index, feature in enumerate(self.__features):
                    if feature.intersects(offset_m, fov_nm, center_nm, Geometry.FloatPoint.make(probe_position)):
                        feature.plot_spectrum(spectrum, e_per_pixel / 10, dimensional_calibrations[1])
                data[:, ...] = spectrum
            else:
                e_per_pixel = 0
            data = self.__get_binned_data(data, binning_shape)
            poisson_level = e_per_pixel + 5  # camera noise
            rs = numpy.random.RandomState()  # use this to avoid blocking other calls to poisson
            data = data * e_per_pixel + (rs.poisson(poisson_level, size=data.shape) - poisson_level)
            return DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)

    def get_camera_dimensional_calibrations(self, camera_type: str, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize):
        if camera_type == "ronchigram":
            height = readout_area.height
            width = readout_area.width
            full_fov_nm = abs(self.__defocus_m) * math.sin(self.__convergence_angle_rad) * 1E9
            fov_nm = Geometry.FloatSize(full_fov_nm * height / self.__ronchigram_shape.height, full_fov_nm * width / self.__ronchigram_shape.width)
            scale_y = binning_shape[0] * fov_nm[0] / readout_area.size[0]
            scale_x = binning_shape[1] * fov_nm[1] / readout_area.size[1]
            offset_y = -scale_y * readout_area.size[0] * 0.5
            offset_x = -scale_x * readout_area.size[1] * 0.5
            dimensional_calibrations = [
                Calibration.Calibration(offset=offset_y, scale=scale_y, units="nm"),
                Calibration.Calibration(offset=offset_x, scale=scale_x, units="nm")
            ]
            return dimensional_calibrations
        if camera_type == "eels":
            dimensional_calibrations = [
                Calibration.Calibration(),
                Calibration.Calibration(offset=-self.__energy_offset_eV, scale=self.__energy_per_channel_eV, units="eV")
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

    # these are required functions to implement the standard stem controller interface.

    def TryGetVal(self, s: str) -> (bool, float):
        if s == "EELS_MagneticShift_Offset":
            return True, self.energy_offset_eV
        elif s == "C_Blanked":
            return True, 1.0 if self.is_blanked else 0.0
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
        return False

    def SetValWait(self, s: str, val: float, timeout_ms: int) -> bool:
        return self.SetVal(s, val)

    def SetValAndConfirm(self, s: str, val: float, tolfactor: float, timeout_ms: int) -> bool:
        return self.SetVal(s, val)

    def SetValDelta(self, s: str, delta: float) -> bool:
        return self.SetVal(s, self.GetVal(s) + delta)

    def InformControl(self, s: str, val: float) -> bool:
        return self.SetVal(s, val)
