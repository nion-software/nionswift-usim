# standard libraries
import copy
import ctypes
import gettext
import math
import numpy
import threading
import time
import typing

# local libraries
from . import InstrumentDevice

# other plug-ins
from Scan import ScanHardwareSource

_ = gettext.gettext


class Channel:

    def __init__(self, channel_id: int, name: str, enabled: bool):
        self.channel_id = channel_id
        self.name = name
        self.enabled = enabled
        self.data = None


class Frame:

    def __init__(self, frame_number: int, channels: typing.List[Channel], frame_parameters: ScanHardwareSource.ScanFrameParameters):
        self.frame_number = frame_number
        self.channels = channels
        self.frame_parameters = frame_parameters
        self.complete = False
        self.bad = False
        self.data_count = 0


class Device:

    def __init__(self, instrument: InstrumentDevice.Instrument):
        self.__instrument = instrument
        self.__blanker_enabled = False
        self.__channels = [Channel(0, "HAADF", True), Channel(1, "MAADF", False)]
        self.__frame_number = None
        self.__frames = list()
        self.__frames_lock = threading.RLock()
        self.__cancel = False
        self.__thread = threading.Thread(target=self.__acquisition_thread)
        self.__thread_event = threading.Event()
        self.__thread_pending_frame_parameters = None
        self.__thread_frame_number = 1
        self.__thread_has_data_event = threading.Event()
        self.__is_scanning = False
        self.__is_stopping = False
        self.on_device_state_changed = None
        self.__profiles = list()
        self.__profiles.append(ScanHardwareSource.ScanFrameParameters({"size": (512, 512), "pixel_time_us": 0.2}))
        self.__profiles.append(ScanHardwareSource.ScanFrameParameters({"size": (1024, 1024), "pixel_time_us": 0.2}))
        self.__profiles.append(ScanHardwareSource.ScanFrameParameters({"size": (2048, 2048), "pixel_time_us": 2.5}))
        self.__frame_parameters = copy.deepcopy(self.__profiles[0])
        self.__thread.start()

    def close(self):
        self.__cancel = True
        self.__thread_event.set()
        self.__thread.join()
        self.__thread = None

    @property
    def blanker_enabled(self) -> bool:
        """Return whether blanker is enabled."""
        return self.__blanker_enabled

    @blanker_enabled.setter
    def blanker_enabled(self, blanker_on: bool) -> None:
        """Set whether blanker is enabled."""
        self.__blanker_enabled = blanker_on

    def change_pmt(self, channel_index: int, increase: bool) -> None:
        """Change the PMT value for the give channel; increase or decrease only."""
        pass

    @property
    def current_frame_parameters(self) -> ScanHardwareSource.ScanFrameParameters:
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
        return True

    def get_channel_name(self, channel_index: int) -> str:
        return self.__channels[channel_index].name

    def __acquisition_thread(self):
        while True:
            if self.__cancel:  # case where cancel occurred in bottom part of this function
                break
            self.__is_stopping = False
            self.__is_scanning = False
            self.__thread_event.wait()
            self.__thread_event.clear()
            if self.__cancel:
                break
            while self.__is_scanning and not self.__cancel and not self.__is_stopping:
                frame_parameters = copy.deepcopy(self.__thread_pending_frame_parameters)
                channels = [copy.deepcopy(channel) for channel in self.__channels if channel.enabled]
                for channel in channels:
                    channel.data = numpy.zeros(frame_parameters.size, numpy.float32)
                frame = Frame(self.__thread_frame_number, channels, frame_parameters)
                self.__thread_frame_number += 1
                with self.__frames_lock:
                    self.__frames.append(frame)
                height = frame_parameters.size[0]
                width = frame_parameters.size[1]
                total_pixels = height * width
                start_time = time.time()
                time_slice = 0.005  # 5ms
                while self.__is_scanning and not self.__cancel and not frame.complete:
                    pixels_remaining = total_pixels - frame.data_count
                    pixel_wait = min(pixels_remaining * frame_parameters.pixel_time_us / 1E6, time_slice)
                    self.__thread_event.wait(pixel_wait)
                    self.__thread_event.clear()
                    if self.__cancel:
                        break
                    if frame_parameters.external_clock_mode != 0:
                        if frame.data_count % width == 0:
                            # throw away two flyback images
                            if not self.__instrument.wait_for_camera_frame(frame_parameters.external_clock_wait_time_ms):
                                frame.bad = True
                                frame.complete = True
                                break
                            if not self.__instrument.wait_for_camera_frame(frame_parameters.external_clock_wait_time_ms):
                                frame.bad = True
                                frame.complete = True
                                break
                        if not self.__instrument.wait_for_camera_frame(frame_parameters.external_clock_wait_time_ms):
                            frame.bad = True
                            frame.complete = True
                            break
                        target_count = frame.data_count + 1
                    else:
                        target_count = min(int((time.time() - start_time) / (frame_parameters.pixel_time_us / 1E6)), total_pixels)
                    if target_count > frame.data_count:
                        for channel in channels:
                            channel_data_flat = channel.data.reshape((total_pixels,))
                            channel_data_flat[frame.data_count:target_count] = numpy.random.randn(target_count - frame.data_count)
                        frame.data_count = target_count
                        frame.complete = frame.data_count == total_pixels
                        self.__thread_has_data_event.set()
                frame.data_count = total_pixels
                frame.complete = True
                # print(f"complete {time.time()} # {frame.frame_number} {frame_parameters.pixel_time_us / 1E6 * total_pixels}")

    def read_partial(self, frame_number, pixels_to_skip) -> (typing.Sequence[dict], bool, bool, tuple, int, int):
        """Read or continue reading a frame.

        The `frame_number` may be None, in which case a new frame should be read.

        The `frame_number` otherwise specifies which frame to continue reading.

        The `pixels_to_skip` specifies where to start reading the frame, if it is a continuation.

        Return values should be a list of dict's (one for each active channel) containing two keys: 'data' and
        'properties' (see below), followed by a boolean indicating whether the frame is complete, a boolean indicating
        whether the frame was bad, a tuple of the form (top, left), (height, width) indicating the valid sub-area
        of the data, the frame number, and the pixels to skip next time around if the frame is not complate.

        The 'data' keys in the list of dict's should contain a ndarray with the size of the full acquisition and each
        ndarray should be the same size. The 'properties' keys are dicts which must contain the frame parameters and
        a 'channel_id' indicating the index of the channel (may be an int or float).
        """

        # wait up to 50ms for a frame to end
        self.__thread_has_data_event.wait(0.05)
        self.__thread_has_data_event.clear()

        current_frame = None
        with self.__frames_lock:
            if frame_number is None:
                current_frame = self.__frames[-1]
                frame_number = current_frame.frame_number
            else:
                for frame in self.__frames:
                    if frame.frame_number == frame_number:
                        current_frame = frame
                        break
        assert current_frame is not None

        data_elements = list()

        for channel in current_frame.channels:
            data_element = dict()
            data_element["data"] = channel.data
            properties = current_frame.frame_parameters.as_dict()
            properties["pixels_x"] = current_frame.frame_parameters.size[1]
            properties["pixels_y"] = current_frame.frame_parameters.size[1]
            properties["center_x_nm"] = current_frame.frame_parameters.center_nm[1]
            properties["center_y_nm"] = current_frame.frame_parameters.center_nm[0]
            properties["rotation_deg"] = math.degrees(current_frame.frame_parameters.rotation_rad)
            properties["channel_id"] = channel.channel_id
            data_element["properties"] = properties
            data_elements.append(data_element)

        width = current_frame.frame_parameters.size[1]
        current_rows_read = current_frame.data_count // width

        if current_frame.complete:
            sub_area = ((0, 0), current_frame.frame_parameters.size)
            pixels_to_skip = 0
            with self.__frames_lock:
                for frame in self.__frames:
                    if frame.frame_number <= frame_number:
                        self.__frames.remove(frame)
        else:
            sub_area = ((pixels_to_skip // width, 0), (current_rows_read - pixels_to_skip // width, width))
            pixels_to_skip = width * current_rows_read

        complete = current_frame.complete
        bad_frame = False

        return data_elements, complete, bad_frame, sub_area, frame_number, pixels_to_skip

    def get_profile_frame_parameters(self, profile_index: int) -> ScanHardwareSource.ScanFrameParameters:
        return copy.deepcopy(self.__profiles[profile_index])

    def open_configuration_interface(self) -> None:
        """Open settings dialog, if any."""
        pass

    def save_frame_parameters(self) -> None:
        """Called when shutting down. Save frame parameters to persistent storage."""
        pass

    def set_frame_parameters(self, frame_parameters: ScanHardwareSource.ScanFrameParameters) -> None:
        """Called just before and during acquisition.

        Device should use these parameters for new acquisition; and update to these parameters during acquisition.
        """
        self.__thread_pending_frame_parameters = copy.deepcopy(frame_parameters)
        self.__frame_parameters = copy.deepcopy(frame_parameters)

    def set_profile_frame_parameters(self, profile_index: int, frame_parameters: ScanHardwareSource.ScanFrameParameters) -> None:
        """Set the acquisition parameters for the give profile_index (0, 1, 2)."""
        self.__profiles[profile_index] = copy.deepcopy(frame_parameters)

    def set_idle_position_by_percentage(self, x: float, y: float) -> None:
        """Set the idle position as a percentage of the last used frame parameters."""
        pass

    def start_frame(self, is_continuous: bool) -> int:
        """Start acquiring. Return the frame number."""
        thread_frame_number = self.__thread_frame_number
        if not self.__is_scanning:
            self.__thread_pending_frame_parameters = copy.deepcopy(self.__frame_parameters)
            self.__is_stopping = False
            self.__is_scanning = True
            self.__thread_event.set()
        return thread_frame_number

    def cancel(self) -> None:
        """Cancel acquisition (immediate)."""
        self.__is_scanning = False
        self.__thread_event.set()

    def stop(self) -> None:
        """Stop acquiring."""
        self.__is_stopping = True

    @property
    def is_scanning(self) -> bool:
        return self.__is_scanning


def run(instrument: InstrumentDevice.Instrument) -> None:

    from nion.swift.model import HardwareSource

    scan_adapter = ScanHardwareSource.ScanAdapter(Device(instrument), "usim_scan_device", _("uSim Scan"))
    scan_hardware_source = ScanHardwareSource.ScanHardwareSource(scan_adapter)
    HardwareSource.HardwareSourceManager().register_hardware_source(scan_hardware_source)
