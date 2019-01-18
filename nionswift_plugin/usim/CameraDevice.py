# standard libraries
import copy
import datetime
import gettext
import numpy
import threading
import time
import typing

# local libraries
from nion.data import Core
from nion.data import DataAndMetadata
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Registry

from . import InstrumentDevice

# other plug-ins
from nion.instrumentation import camera_base

_ = gettext.gettext


class Camera(camera_base.CameraDevice):
    """Implement a camera device."""

    def __init__(self, camera_id: str, camera_type: str, camera_name: str, instrument: InstrumentDevice.Instrument):
        self.camera_id = camera_id
        self.camera_type = camera_type
        self.camera_name = camera_name
        self.__instrument = instrument
        self.__sensor_dimensions = instrument.camera_sensor_dimensions(camera_type)
        self.__readout_area = instrument.camera_readout_area(camera_type)
        self.__symmetric_binning = True
        self.__integration_count = 1
        self.__xdata_buffer = None
        self.__frame_number = 0
        self.__thread = threading.Thread(target=self.__acquisition_thread)
        self.__thread_event = threading.Event()
        self.__has_data_event = threading.Event()
        self.__cancel = False
        self.__is_playing = False
        self.__is_acquiring = False
        self.__cancel_sequence = False
        self.__exposure = 1.0
        self.__binning = 1
        self.__processing = None
        self.__thread.start()

    def close(self):
        self.__cancel = True
        self.__thread_event.set()
        self.__thread.join()
        self.__thread = None

    def _test_rapid_changes(self, queue_task):
        delay = 0.05
        def set_mode(m):
            self.mode = m
        def t():
            import functools
            for i in range(1):
                queue_task(functools.partial(set_mode, "run"))
                # queue_task(functools.partial(set_binning, 1))
                # set_exposure(0.22, 3)
                time.sleep(delay)
                queue_task(functools.partial(set_mode, "snap"))
                # queue_task(functools.partial(set_binning, 2))
                # set_exposure(0.33, 3)
                time.sleep(delay)
                queue_task(functools.partial(set_mode, "tune"))
                # queue_task(functools.partial(set_binning, 4))
                # set_exposure(0.44, 3)
                time.sleep(delay)
        threading.Thread(target=t).start()

    @property
    def sensor_dimensions(self) -> (int, int):
        """Return the maximum sensor dimensions."""
        return self.__sensor_dimensions

    @property
    def readout_area(self) -> (int, int, int, int):
        """Return the readout area TLBR, returned in sensor coordinates (unbinned)."""
        return self.__readout_area

    @readout_area.setter
    def readout_area(self, readout_area_TLBR: (int, int, int, int)) -> None:
        """Set the readout area, specified in sensor coordinates (unbinned). Affects all modes."""
        self.__readout_area = readout_area_TLBR

    @property
    def flip(self):
        """Return whether data is flipped left-right (last dimension)."""
        return False

    @flip.setter
    def flip(self, do_flip):
        """Set whether data is flipped left-right (last dimension). Affects all modes."""
        pass

    @property
    def binning_values(self) -> typing.List[int]:
        """Return possible binning values."""
        return [1, 2, 4, 8]

    def get_expected_dimensions(self, binning: int) -> (int, int):
        """Return expected dimensions for the given binning value."""
        readout_area = self.__readout_area
        return (readout_area[2] - readout_area[0]) // binning, (readout_area[3] - readout_area[1]) // binning

    def set_frame_parameters(self, frame_parameters) -> None:
        self.__exposure = frame_parameters.exposure_ms / 1000
        self.__binning = frame_parameters.binning
        self.__processing = frame_parameters.processing
        self.__integration_count = frame_parameters.integration_count

    @property
    def calibration_controls(self) -> dict:
        """Define the STEM calibration controls for this camera.

        The controls should be unique for each camera if there are more than one.
        """
        return {
            "x_scale_control": self.camera_type + "_x_scale",
            "x_offset_control": self.camera_type + "_x_offset",
            "x_units_value": "eV" if self.camera_type == "eels" else "rad",
            "y_scale_control": self.camera_type + "_y_scale",
            "y_offset_control": self.camera_type + "_y_offset",
            "y_units_value": "" if self.camera_type == "eels" else "rad",
            "intensity_units_value": "counts",
            "counts_per_electron_value": self.__instrument.counts_per_electron
        }

    def start_live(self) -> None:
        """Start live acquisition. Required before using acquire_image."""
        if not self.__is_playing:
            self.__is_playing = True
            self.__has_data_event.clear()  # ensure any has_data_event is new data
            self.__thread_event.set()

    def stop_live(self) -> None:
        """Stop live acquisition."""
        self.__is_playing = False
        self.__thread_event.set()
        # has_data_event is cleared in the acquisition loop after stopping

    def acquire_image(self) -> dict:
        """Acquire the most recent data."""
        xdata_buffer = None
        integration_count = self.__integration_count or 1
        for frame_number in range(integration_count):
            if not self.__has_data_event.wait(self.__exposure * 200) and not self.__thread.is_alive():
                raise Exception("No simulator thread.")
            self.__has_data_event.clear()
            if xdata_buffer is None:
                xdata_buffer = copy.deepcopy(self.__xdata_buffer)
            else:
                xdata_buffer += self.__xdata_buffer
        self.__frame_number += 1
        # note: the data element will include spatial calibrations; but the camera adapter won't use them
        # right now (future fix); it uses a call to 'calibrations' instead.
        # whatever is in "hardware_source" will go into "properties" of data element
        assert len(xdata_buffer.dimensional_shape) == 2
        data_element = dict()
        data_element["version"] = 1
        data_element["data"] = xdata_buffer.data
        data_element["timestamp"] = xdata_buffer.timestamp
        data_element["properties"] = dict()
        data_element["properties"]["frame_number"] = self.__frame_number
        data_element["properties"]["integration_count"] = integration_count
        # data that has been binned vertically to a single row will be converted to 1D
        if xdata_buffer.dimensional_shape[0] == 1:
            data_element["data"] = numpy.squeeze(xdata_buffer.data)
            data_element["collection_dimension_count"] = 0
            data_element["datum_dimension_count"] = 1
        return data_element

    def acquire_sequence_prepare(self, n: int) -> None:
        self.__cancel_sequence = False

    def acquire_sequence(self, n: int) -> typing.Optional[typing.Dict]:
        # if the device does not implement acquire_sequence, fall back to looping acquisition.
        self.__is_acquiring = True
        self.__has_data_event.clear()  # ensure any has_data_event is new data
        self.__thread_event.set()
        self.__instrument.sequence_progress = 0
        try:
            properties = None
            data = None
            for index in range(n):
                if self.__cancel_sequence:
                    return None
                frame_data_element = self.acquire_image()
                frame_data = frame_data_element["data"]
                if data is None:
                    if self.__processing == "sum_project" and len(frame_data.shape) > 1:
                        data = numpy.empty((n,) + frame_data.shape[1:], frame_data.dtype)
                    else:
                        data = numpy.empty((n,) + frame_data.shape, frame_data.dtype)
                if self.__processing == "sum_project" and len(frame_data.shape) > 1:
                    data[index] = Core.function_sum(DataAndMetadata.new_data_and_metadata(frame_data), 0).data
                else:
                    data[index] = frame_data
                properties = copy.deepcopy(frame_data_element["properties"])
                if self.__processing == "sum_project":
                    properties["valid_rows"] = 1
                    spatial_properties = properties.get("spatial_calibrations")
                    if spatial_properties is not None:
                        properties["spatial_calibrations"] = spatial_properties[1:]
                self.__instrument.increment_sequence_progress()
        finally:
            self.__is_acquiring = False
        data_element = dict()
        data_element["data"] = data
        data_element["properties"] = properties
        return data_element

    def acquire_sequence_cancel(self) -> None:
        self.__cancel_sequence = True

    def __acquisition_thread(self):
        while True:
            if self.__cancel:  # case where exposure was canceled.
                break
            self.__thread_event.wait()
            self.__thread_event.clear()
            if self.__cancel:
                break
            while (self.__is_playing or self.__is_acquiring) and not self.__cancel:
                start = time.time()
                readout_area = self.readout_area
                binning_shape = Geometry.IntSize(self.__binning, self.__binning if self.__symmetric_binning else 1)
                xdata = self.__instrument.get_camera_data(self.camera_type, Geometry.IntRect.from_tlbr(*readout_area), binning_shape, self.__exposure)
                elapsed = time.time() - start
                wait_s = max(self.__exposure - elapsed, 0)
                if not self.__thread_event.wait(wait_s):
                    # thread event was not triggered during wait; signal that we have data
                    xdata._set_timestamp(datetime.datetime.utcnow())
                    self.__xdata_buffer = xdata
                    self.__has_data_event.set()
                    self.__instrument.trigger_camera_frame()
                else:
                    # thread event was triggered during wait; continue loop
                    self.__has_data_event.clear()
                    self.__thread_event.clear()


class CameraFrameParameters(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.exposure_ms = self.get("exposure_ms", 125)
        self.binning = self.get("binning", 1)
        self.processing = self.get("processing")
        self.integration_count = self.get("integration_count")

    def __copy__(self):
        return self.__class__(copy.copy(dict(self)))

    def __deepcopy__(self, memo):
        deepcopy = self.__class__(copy.deepcopy(dict(self)))
        memo[id(self)] = deepcopy
        return deepcopy

    def as_dict(self):
        return {
            "exposure_ms": self.exposure_ms,
            "binning": self.binning,
            "processing": self.processing,
            "integration_count": self.integration_count,
        }


class CameraSettings:

    def __init__(self, camera_id: str):
        # these events must be defined
        self.current_frame_parameters_changed_event = Event.Event()
        self.record_frame_parameters_changed_event = Event.Event()
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()

        # optional event and identifier for settings. defining settings_id signals that
        # the settings should be managed as a dict by the container of this class. the container
        # will call apply_settings to initialize settings and then expect settings_changed_event
        # to be fired when settings change.
        self.settings_changed_event = Event.Event()
        self.settings_id = camera_id

        self.__config_file = None

        self.__camera_id = camera_id

        # the list of possible modes should be defined here
        self.modes = ["Run", "Tune", "Snap"]

        # configure profiles
        self.__settings = [
            CameraFrameParameters({"exposure_ms": 100, "binning": 2}),
            CameraFrameParameters({"exposure_ms": 200, "binning": 2}),
            CameraFrameParameters({"exposure_ms": 500, "binning": 1}),
        ]

        self.__current_settings_index = 0

        self.__frame_parameters = copy.deepcopy(self.__settings[self.__current_settings_index])
        self.__record_parameters = copy.deepcopy(self.__settings[-1])

    def close(self):
        pass

    def initialize(self, **kwargs):
        pass

    def apply_settings(self, settings_dict: typing.Dict) -> None:
        """Initialize the settings with the settings_dict."""
        if isinstance(settings_dict, dict):
            settings_list = settings_dict.get("settings", list())
            if len(settings_list) == 3:
                self.__settings = [CameraFrameParameters(settings) for settings in settings_list]
            self.__current_settings_index = settings_dict.get("current_settings_index", 0)
            self.__frame_parameters = CameraFrameParameters(settings_dict.get("current_settings", self.__settings[0].as_dict()))
            self.__record_parameters = copy.deepcopy(self.__settings[-1])

    def __save_settings(self) -> typing.Dict:
        settings_dict = {
            "settings": [settings.as_dict() for settings in self.__settings],
            "current_settings_index": self.__current_settings_index,
            "current_settings": self.__frame_parameters.as_dict()
        }
        return settings_dict

    def get_frame_parameters_from_dict(self, d):
        return CameraFrameParameters(d)

    def set_current_frame_parameters(self, frame_parameters: CameraFrameParameters) -> None:
        """Set the current frame parameters.

        Fire the current frame parameters changed event and optionally the settings changed event.
        """
        self.__frame_parameters = copy.copy(frame_parameters)
        self.settings_changed_event.fire(self.__save_settings())
        self.current_frame_parameters_changed_event.fire(frame_parameters)

    def get_current_frame_parameters(self) -> CameraFrameParameters:
        """Get the current frame parameters."""
        return CameraFrameParameters(self.__frame_parameters)

    def set_record_frame_parameters(self, frame_parameters: CameraFrameParameters) -> None:
        """Set the record frame parameters.

        Fire the record frame parameters changed event and optionally the settings changed event.
        """
        self.__record_parameters = copy.copy(frame_parameters)
        self.record_frame_parameters_changed_event.fire(frame_parameters)

    def get_record_frame_parameters(self) -> CameraFrameParameters:
        """Get the record frame parameters."""
        return self.__record_parameters

    def set_frame_parameters(self, settings_index: int, frame_parameters: CameraFrameParameters) -> None:
        """Set the frame parameters with the settings index and fire the frame parameters changed event.

        If the settings index matches the current settings index, call set current frame parameters.

        If the settings index matches the record settings index, call set record frame parameters.
        """
        assert 0 <= settings_index < len(self.modes)
        frame_parameters = copy.copy(frame_parameters)
        self.__settings[settings_index] = frame_parameters
        # update the local frame parameters
        if settings_index == self.__current_settings_index:
            self.set_current_frame_parameters(frame_parameters)
        if settings_index == len(self.modes) - 1:
            self.set_record_frame_parameters(frame_parameters)
        self.settings_changed_event.fire(self.__save_settings())
        self.frame_parameters_changed_event.fire(settings_index, frame_parameters)

    def get_frame_parameters(self, settings_index) -> CameraFrameParameters:
        """Get the frame parameters for the settings index."""
        return copy.copy(self.__settings[settings_index])

    def set_selected_profile_index(self, settings_index: int) -> None:
        """Set the current settings index.

        Call set current frame parameters if it changed.

        Fire profile changed event if it changed.
        """
        assert 0 <= settings_index < len(self.modes)
        if self.__current_settings_index != settings_index:
            self.__current_settings_index = settings_index
            # set current frame parameters
            self.set_current_frame_parameters(self.__settings[self.__current_settings_index])
            self.settings_changed_event.fire(self.__save_settings())
            self.profile_changed_event.fire(settings_index)

    @property
    def selected_profile_index(self) -> int:
        """Return the current settings index."""
        return self.__current_settings_index

    def get_mode(self) -> str:
        """Return the current mode (named version of current settings index)."""
        return self.modes[self.__current_settings_index]

    def set_mode(self, mode: str) -> None:
        """Set the current mode (named version of current settings index)."""
        self.set_selected_profile_index(self.modes.index(mode))


class CameraModule:

    def __init__(self, stem_controller_id: str, camera_device: Camera, camera_settings: CameraSettings):
        self.stem_controller_id = stem_controller_id
        self.camera_device = camera_device
        self.camera_settings = camera_settings
        self.priority = 20


def run(instrument: InstrumentDevice.Instrument) -> None:
    component_types = {"camera_module"}  # the set of component types that this component represents

    camera_device = Camera("usim_ronchigram_camera", "ronchigram", _("uSim Ronchigram Camera"), instrument)
    camera_device.camera_panel_type = "ronchigram"

    camera_settings = CameraSettings("usim_ronchigram_camera")

    Registry.register_component(CameraModule("usim_stem_controller", camera_device, camera_settings), component_types)

    camera_device = Camera("usim_eels_camera", "eels", _("uSim EELS Camera"), instrument)
    camera_device.camera_panel_type = "eels"

    camera_settings = CameraSettings("usim_eels_camera")

    Registry.register_component(CameraModule("usim_stem_controller", camera_device, camera_settings), component_types)
