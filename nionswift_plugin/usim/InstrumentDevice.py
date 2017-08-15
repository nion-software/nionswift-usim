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


def plot_powerlaw(data: numpy.ndarray, multiplier: float, energy_calibration: Calibration.Calibration) -> None:
    # calculate the range
    # 1 represents 0eV, 0 represents 4000eV
    # TODO: sub-pixel accuracy
    data_range = [0, data.shape[0]]
    energy_range_eV = [energy_calibration.convert_to_calibrated_value(data_range[0]), energy_calibration.convert_to_calibrated_value(data_range[1])]
    if energy_range_eV[0] < 4000 and energy_range_eV[1] > 0:
        energy_range_eV[0] = max(0, energy_range_eV[0])
        energy_range_eV[1] = min(4000, energy_range_eV[1])
        data_range[0] = int(energy_calibration.convert_from_calibrated_value(energy_range_eV[0]))
        data_range[1] = int(energy_calibration.convert_from_calibrated_value(energy_range_eV[1]))
        if energy_range_eV[1] > 4000:
            energy_range_eV[1] = 4000
        assert 0 <= energy_range_eV[0] <= energy_range_eV[1] <= 4000
        assert 0 <= data_range[0] <= data_range[1] <= data.shape[0]
        range = 1 - energy_range_eV[0] / 4000, 1 - energy_range_eV[1] / 4000
        if energy_range_eV[1] - energy_range_eV[0] > 0 and data_range[1] - data_range[0] > 0:
            data[data_range[0]:data_range[1]] += multiplier * scipy.stats.powerlaw(4, loc=0, scale=1).pdf(numpy.linspace(range[0], range[1], data_range[1] - data_range[0])) / 4


def plot_zlp(data: numpy.ndarray, multiplier: float, energy_calibration: Calibration.Calibration, slit_attentuation: float) -> None:
    width = data.shape[0]
    zlp_half_width_eV = 6 / 10 / slit_attentuation  # 12meV = 6meV x 2
    zlp_half_width = zlp_half_width_eV / energy_calibration.scale  # scale is eV/pixel
    half_width_eV = energy_calibration.scale * width // 2
    zlp_offset = (2 * -energy_calibration.offset) / energy_calibration.scale / width
    zlp_offset -= 2 * half_width_eV / energy_calibration.scale / width
    data += multiplier * scipy.stats.exponnorm(2, loc=-zlp_half_width/width + zlp_offset, scale=zlp_half_width/width).pdf(numpy.linspace(-1, 1, width))


class Feature:

    def __init__(self, position_m, size_m, edge_k_eV):
        self.position_m = position_m
        self.size_m = size_m
        self.edge_k_eV = edge_k_eV

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
        plot_powerlaw(data, multiplier, energy_calibration)
        offset_energy_calibration = Calibration.Calibration(offset=energy_calibration.offset - self.edge_k_eV, scale=energy_calibration.scale, units=energy_calibration.units)
        plot_powerlaw(data, multiplier * 0.1, offset_energy_calibration)


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
        energies = [855, 1217, 1839]
        for i in range(100):
            position_m = Geometry.FloatPoint(y=(2 * random.random() - 1.0) * sample_size_m.height, x=(2 * random.random() - 1.0) * sample_size_m.width)
            size_m = feature_percentage * Geometry.FloatSize(height=random.random() * sample_size_m.height, width=random.random() * sample_size_m.width)
            self.__features.append(Feature(position_m, size_m, energies[i%len(energies)]))
        random.setstate(random_state)
        self.__stage_position_m = Geometry.FloatPoint()
        self.__beam_shift_m = Geometry.FloatPoint()
        self.__convergence_angle_rad = 30 / 1000
        self.__defocus_m = 500 / 1E9
        self.__slit_in = False
        self.property_changed_event = Event.Event()
        self.__ronchigram_shape = Geometry.IntSize(1024, 1024)
        self.__eels_shape = Geometry.IntSize(256, 1024)
        self.__last_scan_params = None

    def trigger_camera_frame(self) -> None:
        self.__camera_frame_event.set()

    def wait_for_camera_frame(self, timeout: float) -> bool:
        result = self.__camera_frame_event.wait(timeout)
        self.__camera_frame_event.clear()
        return result

    def get_scan_data(self, frame_parameters) -> numpy.ndarray:
        height = frame_parameters.size[0]
        width = frame_parameters.size[1]
        offset_m = self.stage_position_m - self.beam_shift_m
        fov_nm = Geometry.FloatSize(frame_parameters.fov_nm, frame_parameters.fov_nm)
        center_nm = Geometry.FloatPoint.make(frame_parameters.center_nm)
        size = Geometry.IntSize(height, width)
        data = numpy.zeros((height, width), numpy.float32)
        for feature in self.__features:
            feature.plot(data, offset_m, fov_nm, center_nm, size)
        # print(f"S {offset_m} {fov_nm} {center_nm} {size}")
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
            data = data * e_per_pixel + numpy.random.poisson(e_per_pixel, size=data.shape).astype(numpy.float) - e_per_pixel
            intensity_calibration = Calibration.Calibration(units="e")
            dimensional_calibrations = self.get_camera_dimensional_calibrations(camera_type, readout_area, binning_shape)
            return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)
        else:
            data = numpy.zeros(tuple(self.__eels_shape), numpy.float)
            probe_position = self.probe_position
            slit_attenutation = 100 if self.__slit_in else 1
            e_per_pixel = self.get_electrons_per_pixel(data.shape[0] * data.shape[1], exposure_s) * binning_shape[1] * binning_shape[0] / slit_attenutation
            intensity_calibration = Calibration.Calibration(units="e")
            dimensional_calibrations = self.get_camera_dimensional_calibrations(camera_type, readout_area, binning_shape)
            if self.probe_state == "parked" and probe_position is not None:
                spectrum = numpy.zeros((data.shape[1], ), numpy.float)
                plot_zlp(spectrum, e_per_pixel, dimensional_calibrations[1], slit_attenutation)
                size, fov_nm, center_nm = self.__last_scan_params  # get these from last scan
                offset_m = self.stage_position_m - self.beam_shift_m  # get this from current values
                feature_count = len(self.__features)
                line_width = int(self.__eels_shape.width / (feature_count + 2))
                for index, feature in enumerate(self.__features):
                    if feature.intersects(offset_m, fov_nm, center_nm, Geometry.FloatPoint.make(probe_position)):
                        feature.plot_spectrum(spectrum, e_per_pixel / 10, dimensional_calibrations[1])
                data[:, ...] = spectrum
            data = self.__get_binned_data(data, binning_shape)
            data = data * e_per_pixel + numpy.random.poisson(e_per_pixel, size=data.shape).astype(numpy.float) - e_per_pixel
            return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)

    def get_camera_dimensional_calibrations(self, camera_type: str, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize):
        if camera_type == "ronchigram":
            height = readout_area.height
            width = readout_area.width
            full_fov_nm = abs(self.__defocus_m) * math.sin(self.__convergence_angle_rad) * 1E9
            fov_nm = Geometry.FloatSize(full_fov_nm * height / self.__ronchigram_shape.height, full_fov_nm * width / self.__ronchigram_shape.width)
            dimensional_calibrations = [
                Calibration.Calibration(scale=binning_shape[0]*fov_nm[0]/readout_area.size[0], units="nm"),
                Calibration.Calibration(scale=binning_shape[1]*fov_nm[1]/readout_area.size[1], units="nm")
            ]
            return dimensional_calibrations
        if camera_type == "eels":
            dimensional_calibrations = [
                Calibration.Calibration(),
                Calibration.Calibration(offset=-200, scale=3000/readout_area.size[1], units="eV")
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
    def is_slit_in(self) -> bool:
        return self.__slit_in

    @is_slit_in.setter
    def is_slit_in(self, value: bool) -> None:
        self.__slit_in = value
        self.property_changed_event.fire("is_slit_in")
