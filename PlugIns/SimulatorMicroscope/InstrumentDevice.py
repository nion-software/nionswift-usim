# standard libraries
import numpy
import random
import threading

from nion.utils import Geometry


class Feature:

    def __init__(self, position_m, size_m, angle_rad):
        self.position_m = position_m
        self.size_m = size_m
        self.angle_rad = angle_rad

    def plot(self, data: numpy.ndarray, frame_parameters):
        # TODO: how does center_nm interact with stage position?
        # TODO: take into account feature angle
        # TODO: take into account frame parameters angle
        # TODO: expand features to other shapes than rectangle
        scan_size_m = Geometry.FloatSize(height=frame_parameters.fov_nm, width=frame_parameters.fov_nm) / 1E9
        scan_rect_m = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint.make(frame_parameters.center_nm) / 1E9, scan_size_m)
        feature_rect_m = Geometry.FloatRect.from_center_and_size(self.position_m, self.size_m)
        if scan_rect_m.intersects_rect(feature_rect_m):
            feature_rect_top_px = int(frame_parameters.size[0] * (feature_rect_m.top - scan_rect_m.top) / scan_rect_m.height)
            feature_rect_left_px = int(frame_parameters.size[1] * (feature_rect_m.left - scan_rect_m.left) / scan_rect_m.width)
            feature_rect_height_px = int(frame_parameters.size[0] * feature_rect_m.height / scan_rect_m.height)
            feature_rect_width_px = int(frame_parameters.size[1] * feature_rect_m.width / scan_rect_m.width)
            if feature_rect_top_px < 0:
                feature_rect_height_px += feature_rect_top_px
                feature_rect_top_px = 0
            if feature_rect_left_px < 0:
                feature_rect_width_px += feature_rect_left_px
                feature_rect_left_px = 0
            if feature_rect_top_px + feature_rect_height_px > frame_parameters.size[0]:
                feature_rect_height_px = frame_parameters.size[0] - feature_rect_top_px
            if feature_rect_left_px + feature_rect_width_px > frame_parameters.size[1]:
                feature_rect_width_px = frame_parameters.size[1] - feature_rect_left_px
            feature_rect_origin_px = Geometry.IntPoint(y=feature_rect_top_px, x=feature_rect_left_px)
            feature_rect_size_px = Geometry.IntSize(height=feature_rect_height_px, width=feature_rect_width_px)
            feature_rect_px = Geometry.IntRect(feature_rect_origin_px, feature_rect_size_px)
            data[feature_rect_px.top:feature_rect_px.bottom, feature_rect_px.left:feature_rect_px.right] += 1.0


class Instrument:

    def __init__(self):
        self.__camera_frame_event = threading.Event()
        self.__features = list()
        sample_size_m = Geometry.FloatSize(height=20, width=20) / 1E9
        feature_percentage = 0.3
        for i in range(100):
            position_m = Geometry.FloatPoint(y=(2 * random.random() - 1.0) * sample_size_m.height, x=(2 * random.random() - 1.0) * sample_size_m.width)
            size_m = feature_percentage * Geometry.FloatSize(height=random.random() * sample_size_m.height, width=random.random() * sample_size_m.width)
            self.__features.append(Feature(position_m, size_m, 0.0))

    def trigger_camera_frame(self) -> None:
        self.__camera_frame_event.set()

    def wait_for_camera_frame(self, timeout: float) -> bool:
        result = self.__camera_frame_event.wait(timeout)
        self.__camera_frame_event.clear()
        return result

    def get_scan_data(self, frame_parameters) -> numpy.ndarray:
        height = frame_parameters.size[0]
        width = frame_parameters.size[1]
        data = numpy.zeros((height, width), numpy.float32)
        for feature in self.__features:
            feature.plot(data, frame_parameters)
        noise_factor = 0.3
        data = (data + numpy.random.randn(height, width) * noise_factor) * frame_parameters.pixel_time_us
        return data
