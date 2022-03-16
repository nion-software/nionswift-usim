# standard libraries
import abc
import gettext
import logging

import numpy
import numpy.typing
import random
import scipy.ndimage
import typing

from nion.data import Image
from nion.utils import Geometry

_NDArray = numpy.typing.NDArray[typing.Any]


_ = gettext.gettext


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


class Feature:
    def __init__(self, position_m: Geometry.FloatPoint, size_m: Geometry.FloatSize, edges: typing.Sequence[typing.Tuple[int, int]], plasmon_eV: float, plurality: int) -> None:
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

    def plot(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> int:
        raise NotImplementedError()


class FlakeFeature(Feature):

    def __init__(self, position_m: Geometry.FloatPoint, size_m: Geometry.FloatSize, edges: typing.Sequence[typing.Tuple[int, int]], plasmon_eV: float, plurality: int) -> None:
        super().__init__(position_m, size_m, edges, plasmon_eV, plurality)

    def plot(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> int:
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


class AmorphousBackground(Feature):

    def __init__(self, position_m: Geometry.FloatPoint, size_m: Geometry.FloatSize, edges: typing.Sequence[typing.Tuple[int, int]], plasmon_eV: float, plurality: int) -> None:
        super().__init__(position_m, size_m, edges, plasmon_eV, plurality)
        self.__amorphous = typing.cast(_NDArray, numpy.random.RandomState(1).randn(2048, 2048) * 2 + 1)

    def plot(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> int:
        half_size_nm = self.size_m * 1e9 * 0.5

        # calculate destination bounds in nm
        left_nm = -offset_m.x * 1E9 - fov_nm.width / 2
        top_nm = -offset_m.y * 1E9 - fov_nm.height / 2
        right_nm = left_nm + fov_nm.width
        bottom_nm = top_nm + fov_nm.height

        intersection_left_nm = max(left_nm, -half_size_nm.width)
        intersection_top_nm = max(top_nm, -half_size_nm.height)
        intersection_right_nm = min(right_nm, half_size_nm.width)
        intersection_bottom_nm = min(bottom_nm, half_size_nm.height)

        if intersection_left_nm < intersection_right_nm and intersection_top_nm < intersection_bottom_nm:
            src_left = int(self.__amorphous.shape[1] * max((intersection_left_nm + half_size_nm.width) / (half_size_nm.width * 2), 0))
            src_top = int(self.__amorphous.shape[0] * max((intersection_top_nm + half_size_nm.height) / (half_size_nm.height * 2), 0))
            src_right = int(self.__amorphous.shape[1] * min((intersection_right_nm + half_size_nm.width) / (half_size_nm.width * 2), 1))
            src_bottom = int(self.__amorphous.shape[0] * min((intersection_bottom_nm + half_size_nm.height) / (half_size_nm.height * 2), 1))
            dst_left = int(data.shape[1] * max((intersection_left_nm - left_nm) / (right_nm - left_nm), 0))
            dst_top = int(data.shape[0] * max((intersection_top_nm - top_nm) / (bottom_nm - top_nm), 0))
            dst_right = int(data.shape[1] * min((intersection_right_nm - left_nm) / (right_nm - left_nm), 1))
            dst_bottom = int(data.shape[0] * min((intersection_bottom_nm - top_nm) / (bottom_nm - top_nm), 1))

            src = self.__amorphous[src_top:src_bottom, src_left:src_right]
            range_ = numpy.ptp(src)
            if range_ > 0:
                src = 4 * (src - numpy.amin(src)) / range_
                data[dst_top:dst_bottom, dst_left:dst_right] += Image.scaled(src, (dst_bottom - dst_top, dst_right - dst_left))

        return 0


class GoldBallFeature(Feature):

    def __init__(self, position_m: Geometry.FloatPoint, size_m: Geometry.FloatSize, edges: typing.Sequence[typing.Tuple[int, int]], plasmon_eV: float, plurality: int, orientation: float) -> None:
        super().__init__(position_m, size_m, edges, plasmon_eV, plurality)
        self.orientation = orientation

    def plot(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> int:
        scan_rect_m = self.get_scan_rect_m(offset_m, fov_nm, center_nm)
        feature_rect_m = self.get_feature_rect_m()
        feature_rect_aspect_ratio = min(feature_rect_m.height, feature_rect_m.width) / max(feature_rect_m.height, feature_rect_m.width)
        feature_rect_m = Geometry.FloatRect.from_center_and_size(feature_rect_m.center, (max(feature_rect_m.size), max(feature_rect_m.size)))
        thickness = min(self.size_m) * 2.5e9
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
            feature_data = data[feature_rect_px.top:feature_rect_px.bottom, feature_rect_px.left:feature_rect_px.right]
            ellipse_array = numpy.zeros_like(feature_data)
            draw_ellipse(ellipse_array, (ellipse_array.shape[0] / 2, ellipse_array.shape[1] / 2, ellipse_array.shape[0] / 3,
                                         feature_rect_aspect_ratio * ellipse_array.shape[1] / 3, self.orientation), color=thickness)
            feature_data += scipy.ndimage.gaussian_filter(ellipse_array, 2.0)
            sum += (feature_rect_px.bottom - feature_rect_px.top) * (feature_rect_px.right - feature_rect_px.left)
        return sum


class Sample(abc.ABC):

    @property
    @abc.abstractmethod
    def title(self) -> str: ...

    @property
    @abc.abstractmethod
    def features(self) -> typing.List[Feature]: ...

    @abc.abstractmethod
    def plot_features(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize, extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint, used_size: Geometry.IntSize) -> None: ...


class RectangleFlakeSample(Sample):

    def __init__(self, stage_size_nm: float):
        self.__features: typing.List[Feature] = list()
        sample_size_m = Geometry.FloatSize(height=20 * stage_size_nm / 100, width=20 * stage_size_nm / 100) / 1E9
        feature_percentage = 0.3
        random_state = random.getstate()
        random.seed(1)
        energies = [[(68, 30), (855, 50), (872, 50)], [(29, 15), (1217, 50), (1248, 50)], [(1839, 5), (99, 50)]]  # Ni, Ge, Si
        plasmons = [20, 16.2, 16.8]
        for i in range(100):
            position_m = Geometry.FloatPoint(y=(2 * random.random() - 1.0) * sample_size_m.height, x=(2 * random.random() - 1.0) * sample_size_m.width)
            size_m = feature_percentage * Geometry.FloatSize(height=random.random() * sample_size_m.height, width=random.random() * sample_size_m.width)
            self.__features.append(FlakeFeature(position_m, size_m, energies[i%len(energies)], plasmons[i%len(plasmons)], 4))
        random.setstate(random_state)

    @property
    def title(self) -> str:
        return _("Flake")

    @property
    def features(self) -> typing.List[Feature]:
        return self.__features

    def plot_features(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize, extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint, used_size: Geometry.IntSize) -> None:
        for feature in self.__features:
            feature.plot(data, offset_m, fov_size_nm + extra_nm, center_nm, used_size)


class AmorphousSample(Sample):

    def __init__(self, stage_size_nm: float) -> None:
        self.__amorphous = typing.cast(_NDArray, numpy.random.RandomState(1).randn(2048, 2048)) * 2 + 1
        sample_size_m = Geometry.FloatSize(height=stage_size_nm, width=stage_size_nm) / 1E9
        self.__features: typing.List[Feature] = list()
        self.__features.append(AmorphousBackground(Geometry.FloatPoint(), sample_size_m, [(284, 30)], 25., 4))

    @property
    def title(self) -> str:
        return "Amorphous"

    @property
    def features(self) -> typing.List[Feature]:
        return self.__features

    def plot_features(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize, extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint, used_size: Geometry.IntSize) -> None:
        for feature in self.__features:
            feature.plot(data, offset_m, fov_size_nm + extra_nm, center_nm, used_size)


class CombinedTestSample(Sample):
    def __init__(self, stage_size_nm: float):
        self.__features: typing.List[Feature] = list()
        self.__last_plot_settings: typing.Any = None
        self.__last_plot: typing.Optional[_NDArray] = None
        sample_size_m = Geometry.FloatSize(height=stage_size_nm, width=stage_size_nm) / 1E9
        feature_max_size_m = 20e-9
        feature_min_size_m = 5e-9
        feature_aspect_ratio = 0.7
        random_state = random.getstate()
        random.seed(1)
        counter = 0
        # We map positions to a rather small grid to reduce the number of overlapping gold balls. This does not take into
        # account the gold ball sizes, but essentially gives each one a certain amount of space.
        position_map: _NDArray = numpy.zeros((120, 120), dtype=numpy.uint8)
        while (i := len(self.__features)) < 10000:
            if counter > 50000:
                logging.warning(f'Placing all features on the CTS sample took to long. Stopping after {i} features.')
                break
            counter += 1
            position_m = Geometry.FloatPoint(y=(2 * random.random() - 1.0) * sample_size_m.height, x=(2 * random.random() - 1.0) * sample_size_m.width)
            position_map_position = (int((position_m.y / sample_size_m.height + 1.0) * 0.5 * position_map.shape[0]),
                                     int((position_m.x / sample_size_m.width + 1.0) * 0.5 * position_map.shape[1]))
            if position_map[position_map_position] != 0:
                continue
            long_dimension_size = random.random() * (feature_max_size_m - feature_min_size_m) + feature_min_size_m
            # Aspect ratio is the maximum we allow, so add 1-maximum to the maximum and randomize the first part which
            # gives random nuzmbers between aspect_ratio and 1
            aspect_ratio = random.random() * (1 - feature_aspect_ratio) + feature_aspect_ratio
            short_dimension_size = aspect_ratio * long_dimension_size
            size_m = Geometry.FloatSize(height=long_dimension_size, width=short_dimension_size)
            # Random rotation between 0 and 180 degrees
            rotation = random.random() * numpy.pi
            new_feature = GoldBallFeature(position_m, size_m, [(54, 30), (83, 40), (2206, 300), (2291, 300)], 15., 4, rotation)
            self.__features.append(new_feature)
            position_map[position_map_position] = 1
        self.__features.append(AmorphousBackground(Geometry.FloatPoint(), sample_size_m, [(284, 30)], 25., 4))
        random.setstate(random_state)

    @property
    def title(self) -> str:
        return _("CTS")

    @property
    def features(self) -> typing.List[Feature]:
        return self.__features

    def plot_features(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize, extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint, used_size: Geometry.IntSize) -> None:
        plot_settings = (data.shape, offset_m.as_tuple(), fov_size_nm.as_tuple(), extra_nm.as_tuple(), center_nm.as_tuple(), used_size.as_tuple())
        if self.__last_plot is None or plot_settings != self.__last_plot_settings:
            for feature in self.__features:
                feature.plot(data, offset_m, fov_size_nm + extra_nm, center_nm, used_size)
            self.__last_plot_settings = plot_settings
            self.__last_plot = data.copy()
        else:
            data[:] = self.__last_plot
