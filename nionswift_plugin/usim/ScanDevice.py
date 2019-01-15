# standard libraries
import copy
import gettext
import math
import numpy
import time
import typing

# local libraries
from . import InstrumentDevice

# other plug-ins
from nion.instrumentation import scan_base
from nion.utils import Geometry
from nion.utils import Registry

_ = gettext.gettext


class Channel:

    def __init__(self, channel_id: int, name: str, enabled: bool):
        self.channel_id = channel_id
        self.name = name
        self.enabled = enabled
        self.data = None


class Frame:

    def __init__(self, frame_number: int, channels: typing.List[Channel], frame_parameters: scan_base.ScanFrameParameters):
        self.frame_number = frame_number
        self.channels = channels
        self.frame_parameters = frame_parameters
        self.complete = False
        self.bad = False
        self.data_count = 0
        self.start_time = time.time()
        self.scan_data = None


class Device:

    def __init__(self, instrument: InstrumentDevice.Instrument):
        self.scan_device_id = "usim_scan_device"
        self.scan_device_name = _("uSim Scan")
        self.stem_controller_id = "usim_stem_controller"
        self.__instrument = instrument
        self.__channels = self.__get_channels()
        self.__frame = None
        self.__frame_number = 0
        self.__is_scanning = False
        self.on_device_state_changed = None
        self.__profiles = self.__get_initial_profiles()
        self.__frame_parameters = copy.deepcopy(self.__profiles[0])
        self.flyback_pixels = 2
        self.__buffer = list()

    def close(self):
        pass

    def __get_channels(self) -> typing.List[Channel]:
        return [Channel(0, "HAADF", True), Channel(1, "MAADF", False), Channel(2, "X1", False), Channel(3, "X2", False)]

    def __get_initial_profiles(self) -> typing.List[scan_base.ScanFrameParameters]:
        profiles = list()
        # profiles.append(scan_base.ScanFrameParameters({"size": (512, 512), "pixel_time_us": 0.2}))
        # profiles.append(scan_base.ScanFrameParameters({"size": (1024, 1024), "pixel_time_us": 0.2}))
        # profiles.append(scan_base.ScanFrameParameters({"size": (2048, 2048), "pixel_time_us": 2.5}))
        profiles.append(scan_base.ScanFrameParameters({"size": (256, 256), "pixel_time_us": 1, "fov_nm": 10}))
        profiles.append(scan_base.ScanFrameParameters({"size": (512, 512), "pixel_time_us": 1, "fov_nm": 40}))
        profiles.append(scan_base.ScanFrameParameters({"size": (1024, 1024), "pixel_time_us": 1, "fov_nm": 100}))
        return profiles

    @property
    def current_frame_parameters(self) -> scan_base.ScanFrameParameters:
        return self.__frame_parameters

    @property
    def channel_count(self):
        return len(self.__channels)

    @property
    def channels_enabled(self) -> typing.Tuple[bool, ...]:
        return tuple(channel.enabled for channel in self.__channels)

    def set_channel_enabled(self, channel_index: int, enabled: bool) -> bool:
        assert 0 <= channel_index < self.channel_count
        self.__channels[channel_index].enabled = enabled
        if not any(channel.enabled for channel in self.__channels):
            self.cancel()
        return True

    def get_channel_name(self, channel_index: int) -> str:
        return self.__channels[channel_index].name

    def read_partial(self, frame_number, pixels_to_skip) -> (typing.Sequence[dict], bool, bool, tuple, int, int):
        """Read or continue reading a frame.

        The `frame_number` may be None, in which case a new frame should be read.

        The `frame_number` otherwise specifies which frame to continue reading.

        The `pixels_to_skip` specifies where to start reading the frame, if it is a continuation.

        Return values should be a list of dict's (one for each active channel) containing two keys: 'data' and
        'properties' (see below), followed by a boolean indicating whether the frame is complete, a boolean indicating
        whether the frame was bad, a tuple of the form (top, left), (height, width) indicating the valid sub-area
        of the data, the frame number, and the pixels to skip next time around if the frame is not complete.

        The 'data' keys in the list of dict's should contain a ndarray with the size of the full acquisition and each
        ndarray should be the same size. The 'properties' keys are dicts which must contain the frame parameters and
        a 'channel_id' indicating the index of the channel (may be an int or float).
        """

        if self.__frame is None:
            self.__start_next_frame()
        current_frame = self.__frame
        assert current_frame is not None
        frame_number = current_frame.frame_number

        frame_parameters = current_frame.frame_parameters
        size = Geometry.IntSize.make(frame_parameters.subscan_pixel_size if frame_parameters.subscan_pixel_size else frame_parameters.size)
        total_pixels = size.height * size.width
        time_slice = 0.005  # 5ms

        if current_frame.scan_data is None:
            scan_data = list()
            for channel in current_frame.channels:
                scan_data.append(self.__instrument.get_scan_data(current_frame.frame_parameters, channel))
            current_frame.scan_data = scan_data

        if current_frame.data_count == 0 and frame_parameters.external_clock_mode != 0:
            self.__instrument.live_probe_position = Geometry.FloatPoint()

        target_count = 0
        while self.__is_scanning and target_count <= current_frame.data_count:
            if frame_parameters.external_clock_mode != 0:
                h, w = current_frame.scan_data[0].shape
                y, x = current_frame.data_count // w, current_frame.data_count % w
                if current_frame.data_count % size.width == 0:
                    # throw away two flyback images at beginning of line
                    if not self.__is_scanning or not self.__instrument.wait_for_camera_frame(frame_parameters.external_clock_wait_time_ms / 1000):
                        current_frame.bad = True
                        current_frame.complete = True
                    if not self.__is_scanning or not self.__instrument.wait_for_camera_frame(frame_parameters.external_clock_wait_time_ms / 1000):
                        current_frame.bad = True
                        current_frame.complete = True
                if not self.__is_scanning or not self.__instrument.wait_for_camera_frame(frame_parameters.external_clock_wait_time_ms / 1000):
                    current_frame.bad = True
                    current_frame.complete = True
                self.__instrument.live_probe_position = Geometry.FloatPoint(y=y / h, x=x / w)
                sequence_progress = self.__instrument.sequence_progress
                # target count is the max of the sequence progress calculation vs the data count.
                # they will generally be the same; but might be out of sync slightly.
                # keeping both as a safety measure.
                target_count = max(sequence_progress - 2 * (sequence_progress // (size.width + 2) + 1) + 1, current_frame.data_count + 1)
            else:
                pixels_remaining = total_pixels - current_frame.data_count
                pixel_wait = min(pixels_remaining * frame_parameters.pixel_time_us / 1E6, time_slice)
                time.sleep(pixel_wait)
                target_count = min(int((time.time() - current_frame.start_time) / (frame_parameters.pixel_time_us / 1E6)), total_pixels)

        if self.__is_scanning and target_count > current_frame.data_count:
            for channel_index, channel in enumerate(current_frame.channels):
                scan_data_flat = current_frame.scan_data[channel_index].reshape((total_pixels,))
                channel_data_flat = channel.data.reshape((total_pixels,))
                channel_data_flat[current_frame.data_count:target_count] = scan_data_flat[current_frame.data_count:target_count]
            current_frame.data_count = target_count
            current_frame.complete = current_frame.data_count == total_pixels
        else:
            assert not self.__is_scanning
            current_frame.data_count = total_pixels
            current_frame.complete = True

        data_elements = list()

        for channel in current_frame.channels:
            data_element = dict()
            data_element["data"] = channel.data
            properties = current_frame.frame_parameters.as_dict()
            properties["center_x_nm"] = current_frame.frame_parameters.center_nm[1]
            properties["center_y_nm"] = current_frame.frame_parameters.center_nm[0]
            properties["rotation_deg"] = math.degrees(current_frame.frame_parameters.rotation_rad)
            properties["channel_id"] = channel.channel_id
            data_element["properties"] = properties
            data_elements.append(data_element)

        current_rows_read = current_frame.data_count // size.width

        if current_frame.complete:
            sub_area = ((0, 0), size)
            pixels_to_skip = 0
            self.__frame = None
        else:
            sub_area = ((pixels_to_skip // size.width, 0), (current_rows_read - pixels_to_skip // size.width, size.width))
            pixels_to_skip = size.width * current_rows_read

        complete = current_frame.complete
        bad_frame = False

        if complete:
            if len(self.__buffer) > 0 and len(self.__buffer[-1]) != len(data_elements):
                self.__buffer = list()
            self.__buffer.append(data_elements)
            while len(self.__buffer) > 100:
                del self.__buffer[0]

        return data_elements, complete, bad_frame, sub_area, frame_number, pixels_to_skip

    def get_profile_frame_parameters(self, profile_index: int) -> scan_base.ScanFrameParameters:
        return copy.deepcopy(self.__profiles[profile_index])

    def open_configuration_interface(self) -> None:
        """Open settings dialog, if any."""
        pass

    def save_frame_parameters(self) -> None:
        """Called when shutting down. Save frame parameters to persistent storage."""
        pass

    def set_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters) -> None:
        """Called just before and during acquisition.

        Device should use these parameters for new acquisition; and update to these parameters during acquisition.
        """
        self.__frame_parameters = copy.deepcopy(frame_parameters)

    def set_profile_frame_parameters(self, profile_index: int, frame_parameters: scan_base.ScanFrameParameters) -> None:
        """Set the acquisition parameters for the give profile_index (0, 1, 2)."""
        self.__profiles[profile_index] = copy.deepcopy(frame_parameters)

    def set_idle_position_by_percentage(self, x: float, y: float) -> None:
        """Set the idle position as a percentage of the last used frame parameters."""
        pass

    def start_frame(self, is_continuous: bool) -> int:
        """Start acquiring. Return the frame number."""
        if not self.__is_scanning:
            self.__buffer = list()
            self.__start_next_frame()
            self.__is_scanning = True
            self.__instrument.live_probe_position = Geometry.FloatPoint() if self.__frame_parameters.external_clock_mode != 0 else None
        return self.__frame_number

    def __start_next_frame(self):
        frame_parameters = copy.deepcopy(self.__frame_parameters)
        channels = [copy.deepcopy(channel) for channel in self.__channels if channel.enabled]
        size = Geometry.IntSize.make(frame_parameters.subscan_pixel_size if frame_parameters.subscan_pixel_size else frame_parameters.size)
        for channel in channels:
            channel.data = numpy.zeros(tuple(size), numpy.float32)
        self.__frame_number += 1
        self.__frame = Frame(self.__frame_number, channels, frame_parameters)

    def cancel(self) -> None:
        """Cancel acquisition (immediate)."""
        self.__is_scanning = False
        self.__instrument.live_probe_position = None
        self.__instrument.trigger_camera_frame()

    def stop(self) -> None:
        """Stop acquiring."""
        pass

    @property
    def is_scanning(self) -> bool:
        return self.__is_scanning

    def get_buffer_data(self, start: int, count: int) -> typing.Optional[typing.List[typing.List[typing.Dict]]]:
        if start < 0:
            return self.__buffer[start: start+count if count < -start else None]
        else:
            return self.__buffer[start: start+count]


def run(instrument: InstrumentDevice.Instrument) -> None:
    scan_device = Device(instrument)
    scan_device.priority = 20
    Registry.register_component(scan_device, {"scan_device"})
