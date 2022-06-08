from __future__ import annotations

# standard libraries
import asyncio
import copy
import datetime
import gettext
import numpy
import numpy.typing
import pathlib
import threading
import time
import typing

# local libraries
from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata
from nion.swift.model import ImportExportManager
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Registry

# other plug-ins
from nion.instrumentation import camera_base
from . import RonchigramCameraSimulator
from . import EELSCameraSimulator

if typing.TYPE_CHECKING:
    from . import InstrumentDevice
    from . import CameraSimulator
    from . import ScanDevice

_NDArray = numpy.typing.NDArray[typing.Any]

_ = gettext.gettext


class Camera(camera_base.CameraDevice3):
    """Implement a camera device."""

    def __init__(self, camera_id: str, camera_type: str, camera_name: str, instrument: InstrumentDevice.Instrument):
        self.camera_id = camera_id
        self.camera_type = camera_type
        self.camera_name = camera_name
        self.__camera_task: typing.Optional[CameraTask] = None
        self.__instrument = instrument
        self.__scan_device: typing.Optional[ScanDevice.Device] = None
        self.__simulator: CameraSimulator.CameraSimulator
        if camera_type == "ronchigram":
            self.__simulator = RonchigramCameraSimulator.RonchigramCameraSimulator(instrument, Geometry.IntSize.make(instrument.camera_sensor_dimensions("ronchigram")), instrument.counts_per_electron, instrument.stage_size_nm)
        elif camera_type == "eels":
            self.__simulator = EELSCameraSimulator.EELSCameraSimulator(instrument, Geometry.IntSize.make(instrument.camera_sensor_dimensions("eels")), instrument.counts_per_electron)
        else:
            raise ValueError(f"Unsupported camera type '{camera_type}'.")
        self.__sensor_dimensions = instrument.camera_sensor_dimensions(camera_type)
        self.__readout_area = instrument.camera_readout_area(camera_type)
        self.__symmetric_binning = True
        self.__integration_count = 1
        self.__xdata_buffer: typing.Optional[DataAndMetadata.DataAndMetadata] = None
        self.__frame_number = 0
        self.__thread = threading.Thread(target=self.__acquisition_thread)
        self.__thread_event = threading.Event()
        self.__acquired_one_event = threading.Event()  # initial startup
        self.__has_data_event = threading.Event()
        self.__cancel = False
        self.__is_playing = False
        self.__is_acquiring = False
        self.__cancel_sequence_event = threading.Event()
        self.__exposure = 1.0
        self.__binning = 1
        self.__processing: typing.Optional[str] = None
        self.__mask_array: typing.Optional[_NDArray] = None
        self.__thread.start()

        # Also register the camera device and use a unique name for it so that we can directly access it
        if Registry.get_component(f"usim_{camera_type}_camera_device"):
            raise RuntimeError(f"Component 'usim_{camera_type}_camera_device' is already registered.")
        Registry.register_component(self, {f"usim_{camera_type}_camera_device"})

        # TODO Define external trigger interface and the mechanism for aqcuiring multiple sequences
        self._external_trigger = False # Just here for testing, we need to decide on how to specify external trigger mode in the base class

    def close(self) -> None:
        self.__cancel = True
        self.__thread_event.set()
        self.__thread.join()
        self.__thread = typing.cast(typing.Any, None)
        self.__simulator.close()
        Registry.unregister_component(self)

    @property
    def sensor_dimensions(self) -> typing.Tuple[int, int]:
        """Return the maximum sensor dimensions."""
        return self.__sensor_dimensions

    @property
    def readout_area(self) -> typing.Tuple[int, int, int, int]:
        """Return the readout area TLBR, returned in sensor coordinates (unbinned)."""
        return self.__readout_area

    @readout_area.setter
    def readout_area(self, readout_area_TLBR: typing.Tuple[int, int, int, int]) -> None:
        """Set the readout area, specified in sensor coordinates (unbinned). Affects all modes."""
        self.__readout_area = readout_area_TLBR

    @property
    def flip(self) -> bool:
        """Return whether data is flipped left-right (last dimension)."""
        return False

    @flip.setter
    def flip(self, do_flip: bool) -> None:
        """Set whether data is flipped left-right (last dimension). Affects all modes."""
        pass

    @property
    def binning_values(self) -> typing.List[int]:
        """Return possible binning values."""
        return [1, 2, 4, 8]

    @property
    def mask_array(self) -> typing.Optional[_NDArray]:
        return self.__mask_array

    @property
    def simulator(self) -> CameraSimulator.CameraSimulator:
        return self.__simulator

    def get_expected_dimensions(self, binning: int) -> typing.Tuple[int, int]:
        """Return expected dimensions for the given binning value."""
        readout_area = self.__readout_area
        return (readout_area[2] - readout_area[0]) // binning, (readout_area[3] - readout_area[1]) // binning

    def set_frame_parameters(self, frame_parameters: camera_base.CameraFrameParameters) -> None:
        self.__set_frame_parameters(frame_parameters)

    def __set_frame_parameters(self, frame_parameters: camera_base.CameraFrameParameters) -> None:
        self.__exposure = frame_parameters.exposure_ms / 1000
        self.__binning = frame_parameters.binning
        self.__processing = frame_parameters.processing
        self.__integration_count = frame_parameters.integration_count
        mask_array = [mask.get_mask_array(self.get_expected_dimensions(self.__binning)) for mask in frame_parameters.active_masks]
        self.__mask_array = numpy.array(mask_array) if mask_array else None

    @property
    def calibration_controls(self) -> typing.Mapping[str, typing.Union[str, int, float]]:
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

    def get_dimensional_calibrations(self, readout_area: typing.Optional[Geometry.IntRect], binning_shape: typing.Optional[Geometry.IntSize]) -> typing.Sequence[Calibration.Calibration]:
        return self.__simulator.get_dimensional_calibrations(readout_area, binning_shape)

    def start_live(self) -> None:
        """Start live acquisition. Required before using acquire_image."""
        if not self.__is_playing:
            self.__is_playing = True
            self.__has_data_event.clear()  # ensure any has_data_event is new data
            self.__thread_event.set()
            self.__acquired_one_event.wait(30)

    def stop_live(self) -> None:
        """Stop live acquisition."""
        self.__is_playing = False
        self.__thread_event.set()
        # has_data_event is cleared in the acquisition loop after stopping

    def acquire_image(self) -> ImportExportManager.DataElementType:
        return self.__acquire_image(direct=False)

    def __acquire_image(self, *, direct: bool, index: int = 0) -> ImportExportManager.DataElementType:
        """Acquire the most recent data.

        The direct parameter is a shortcut to optimize the speed during sequence acquisition. When
        set to True, it uses a direct form of acquisition that does not sync with the data thread.

        The index parameter is used if direct is True; it allows direct to synchronize with the
        start of the acquistion thread.
        """
        xdata_buffer = None
        integration_count = self.__integration_count or 1
        for frame_number in range(integration_count):
            if direct:
                self.__direct_acquire(self.__cancel_sequence_event, index == 0)
            else:
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
        assert xdata_buffer
        assert len(xdata_buffer.dimensional_shape) == 2
        data_element: typing.Dict[str, typing.Any] = dict()
        data_element["version"] = 1
        data_element["data"] = xdata_buffer.data
        data_element["timestamp"] = xdata_buffer.timestamp
        data_element["properties"] = dict()
        data_element["properties"]["frame_number"] = self.__frame_number
        data_element["properties"]["integration_count"] = integration_count
        # data that has been binned vertically to a single row will be converted to 1D
        if xdata_buffer.dimensional_shape[0] == 1:
            data_element["data"] = numpy.squeeze(xdata_buffer._data_ex)
            data_element["collection_dimension_count"] = 0
            data_element["datum_dimension_count"] = 1
        return data_element

    def _acquire_sequence(self, n: int) -> typing.Optional[ImportExportManager.DataElementType]:
        # if the device does not implement acquire_sequence, fall back to looping acquisition.
        self.__is_acquiring = True
        self.__has_data_event.clear()  # ensure any has_data_event is new data
        try:
            properties = None
            data = None
            if self._external_trigger:
                scan_device: typing.Optional[ScanDevice.Device] = Registry.get_component("scan_device")
                if scan_device and hasattr(scan_device, "blanker_signal_condition"):
                    with scan_device.blanker_signal_condition:
                        scan_device.blanker_signal_condition.wait(timeout=max(self.__exposure * 2, 5))
            for index in range(n):
                if self.__cancel_sequence_event.is_set():
                    return None
                frame_data_element = self.__acquire_image(direct=True, index=index)
                frame_data = frame_data_element["data"]
                if self.__processing == "sum_project" and len(frame_data.shape) > 1:
                    data_shape = (n,) + frame_data.shape[1:]
                    data_dtype = frame_data.dtype
                elif self.__processing == "sum_masked":
                    data_shape = (n, len(self.__mask_array) if self.__mask_array is not None else 1)
                    data_dtype = frame_data.dtype
                else:
                    data_shape = (n,) + frame_data.shape
                    data_dtype = frame_data.dtype
                if data is None:
                    data = numpy.zeros(data_shape, data_dtype)
                assert data.shape == data_shape
                assert data.dtype == data_dtype
                if self.__processing == "sum_project" and len(frame_data.shape) > 1:
                    summed_xdata = Core.function_sum(DataAndMetadata.new_data_and_metadata(frame_data), 0)
                    assert summed_xdata
                    summed_data = summed_xdata.data
                    assert summed_data is not None
                    data[index] = summed_data
                elif self.__processing == "sum_masked":
                    if self.__mask_array is not None:
                        summed_xdata = Core.function_sum(DataAndMetadata.new_data_and_metadata(frame_data * self.__mask_array), (1, 2))
                        assert summed_xdata
                        summed_data = summed_xdata.data
                        assert summed_data is not None
                        data[index] = summed_data
                    else:
                        data[index] = numpy.sum(frame_data)
                else:
                    data[index] = frame_data
                properties = copy.deepcopy(frame_data_element["properties"])
                if self.__processing == "sum_project":
                    properties["valid_rows"] = 1
                    spatial_properties = properties.get("spatial_calibrations")
                    if spatial_properties is not None:
                        properties["spatial_calibrations"] = spatial_properties[1:]
        finally:
            self.__is_acquiring = False
        data_element = dict()
        data_element["data"] = data
        data_element["properties"] = properties
        return data_element

    def acquire_sequence_cancel(self) -> None:
        self.__cancel_sequence_event.set()

    def acquire_sequence_begin(self, camera_frame_parameters: camera_base.CameraFrameParameters, count: int, **kwargs: typing.Any) -> camera_base.PartialData:
        self.__cancel_sequence_event.clear()
        self.__set_frame_parameters(camera_frame_parameters)
        self.__camera_task = CameraTask(self, camera_frame_parameters, (count,))
        self.__camera_task.start()
        return camera_base.PartialData(self.__camera_task._xdata_ex, False, False, None, 0)

    def acquire_sequence_continue(self, *, update_period: float = 1.0, **kwargs: typing.Any) -> camera_base.PartialData:
        assert self.__camera_task
        is_complete, is_canceled, valid_count = self.__camera_task.grab_partial(update_period=update_period)
        return camera_base.PartialData(self.__camera_task._xdata_ex, is_complete, is_canceled, None, valid_count)

    def acquire_sequence_end(self, **kwargs: typing.Any) -> None:
        self.__camera_task = None

    def __direct_acquire(self, cancel_event: threading.Event, do_sync: bool = False) -> bool:
        # if do_sync is True, the scan device will sync with its acquisition thread before
        # advancing the pixel. this avoids race conditions of when the pixel count gets reset.
        # this will be set only for the first frame in a sequence.
        start = time.time()
        readout_area = self.readout_area
        binning_shape = Geometry.IntSize(self.__binning, self.__binning if self.__symmetric_binning else 1)
        # scan device is only set during synchronized acquisition
        if self.__scan_device:
            self.__scan_device.advance_pixel(do_sync=do_sync)
        xdata = self.__simulator.get_frame_data(Geometry.IntRect.from_tlbr(*readout_area), binning_shape, self.__exposure, self.__instrument.scan_context, self.__instrument.probe_position)
        self.__acquired_one_event.set()
        elapsed = time.time() - start
        wait_s = max(self.__exposure - elapsed, 0)
        if not cancel_event.wait(wait_s):
            # thread event was not triggered during wait; signal that we have data
            xdata._set_timestamp(datetime.datetime.utcnow())
            self.__xdata_buffer = xdata
            return True
        return False

    def __acquisition_thread(self) -> None:
        while True:
            if self.__cancel:  # case where exposure was canceled.
                break
            self.__thread_event.wait()
            self.__thread_event.clear()
            if self.__cancel:
                break
            while (self.__is_playing or self.__is_acquiring) and not self.__cancel:
                if self.__direct_acquire(self.__thread_event):
                    self.__has_data_event.set()
                else:
                    # thread event was triggered during wait; continue loop
                    self.__has_data_event.clear()
                    self.__thread_event.clear()

    def acquire_synchronized_begin(self, camera_frame_parameters: camera_base.CameraFrameParameters, collection_shape: DataAndMetadata.ShapeType, **kwargs: typing.Any) -> camera_base.PartialData:
        self.__cancel_sequence_event.clear()
        self.__set_frame_parameters(camera_frame_parameters)
        self.__scan_device = Registry.get_component("scan_device")
        self.__camera_task = CameraTask(self, camera_frame_parameters, collection_shape)
        self.__camera_task.start()
        return camera_base.PartialData(self.__camera_task._xdata_ex, False, False, None, 0)

    def acquire_synchronized_continue(self, *, update_period: float = 1.0, **kwargs: typing.Any) -> camera_base.PartialData:
        assert self.__camera_task
        is_complete, is_canceled, valid_count = self.__camera_task.grab_partial(update_period=update_period)
        return camera_base.PartialData(self.__camera_task._xdata_ex, is_complete, is_canceled, None, valid_count)

    def acquire_synchronized_end(self, **kwargs: typing.Any) -> None:
        self.__camera_task = None
        self.__scan_device = None

    def acquire_synchronized_cancel(self) -> None:
        self.__cancel_sequence_event.set()

    @property
    def _is_acquire_synchronized_running(self) -> bool:
        return self.__camera_task is not None


class CameraTask:
    def __init__(self, camera_device: Camera, camera_frame_parameters: camera_base.CameraFrameParameters, collection_shape: typing.Tuple[int, ...]) -> None:
        self.__camera_device = camera_device
        self.__camera_frame_parameters = camera_frame_parameters
        self.__collection_shape = collection_shape
        self.__count = int(numpy.product(self.__collection_shape))
        self.__aborted = False
        self.__data: typing.Optional[_NDArray] = None
        self.__xdata: typing.Optional[DataAndMetadata.DataAndMetadata] = None
        self.__start = 0

    @property
    def xdata(self) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
        return self.__xdata

    @property
    def _xdata_ex(self) -> DataAndMetadata.DataAndMetadata:
        assert self.__xdata
        return self.__xdata

    def start(self) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
        # returns the full readout, including flyback pixels
        camera_readout_shape: typing.Tuple[int, ...] = self.__camera_device.get_expected_dimensions(self.__camera_frame_parameters.binning)
        if self.__camera_frame_parameters.processing == "sum_project":
            camera_readout_shape = camera_readout_shape[1:]
        elif self.__camera_frame_parameters.processing == "sum_masked":
            camera_readout_shape = (len(self.__camera_device.mask_array) if self.__camera_device.mask_array is not None else 1,)
        self.__data = numpy.zeros(self.__collection_shape + camera_readout_shape, numpy.float32)
        data_descriptor = DataAndMetadata.DataDescriptor(False, len(self.__collection_shape), len(camera_readout_shape))
        self.__xdata = DataAndMetadata.new_data_and_metadata(self.__data, data_descriptor=data_descriptor)
        return self.__xdata

    def grab_partial(self, *, update_period: float = 1.0) -> typing.Tuple[bool, bool, int]:
        # updates the full readout data, returns a tuple of is complete, is canceled, and
        # the number of valid rows.
        start = self.__start
        frame_parameters = self.__camera_frame_parameters
        exposure = frame_parameters.exposure_ms / 1000.0
        n = min(max(int(update_period / exposure), 1), self.__count - start)
        is_complete = start + n == self.__count
        # print(f"{start=} {n=} {self.__count=} {is_complete=}")
        data_element = self.__camera_device._acquire_sequence(n)
        if data_element and not self.__aborted:
            xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_element)
            dimensional_calibrations = tuple(Calibration.Calibration() for _ in range(len(self.__collection_shape))) + tuple(xdata.dimensional_calibrations[1:])
            assert self.__xdata
            self.__xdata._set_intensity_calibration(xdata.intensity_calibration)
            self.__xdata._set_dimensional_calibrations(dimensional_calibrations)
            if len(self.__collection_shape) > 1:
                row_size = self.__collection_shape[-1]
                start_row = start // row_size
                rows = n // row_size
                metadata = dict(xdata.metadata)
                metadata.setdefault("hardware_source", dict())["valid_rows"] = start_row + rows
                self.__xdata._set_metadata(metadata)
            else:
                row_size = 1
            # convert from a sequence to a collection.
            assert self.__data is not None
            self.__data.reshape((self.__data.shape[0] * row_size,) + self.__data.shape[len(self.__collection_shape):])[start:start + n, ...] = xdata._data_ex
            self.__start = start + n
            return is_complete, False, start + n
        self.__start = 0
        return True, True, 0


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
            camera_base.CameraFrameParameters({"exposure_ms": 100, "binning": 2}),
            camera_base.CameraFrameParameters({"exposure_ms": 200, "binning": 2}),
            camera_base.CameraFrameParameters({"exposure_ms": 500, "binning": 1}),
        ]

        self.__current_settings_index = 0
        self.__masks: typing.List[camera_base.Mask] = list()
        self.__frame_parameters = copy.deepcopy(self.__settings[self.__current_settings_index])
        self.__record_parameters = copy.deepcopy(self.__settings[-1])

    def close(self) -> None:
        pass

    def initialize(self, configuration_location: typing.Optional[pathlib.Path] = None, event_loop: typing.Optional[asyncio.AbstractEventLoop] = None, **kwargs: typing.Any) -> None:
        pass

    @property
    def masks(self) -> typing.List[camera_base.Mask]:
        return self.__masks

    def apply_settings(self, settings_dict: typing.Mapping[str, typing.Any]) -> None:
        """Initialize the settings with the settings_dict."""
        if isinstance(settings_dict, dict):
            settings_list = settings_dict.get("settings", list())
            if len(settings_list) == 3:
                self.__settings = [camera_base.CameraFrameParameters(settings) for settings in settings_list]
            self.__current_settings_index = settings_dict.get("current_settings_index", 0)
            self.__frame_parameters = camera_base.CameraFrameParameters(settings_dict.get("current_settings", self.__settings[0].as_dict()))
            self.__record_parameters = copy.deepcopy(self.__settings[-1])

    def __save_settings(self) -> typing.Mapping[str, typing.Any]:
        settings_dict = {
            "settings": [settings.as_dict() for settings in self.__settings],
            "current_settings_index": self.__current_settings_index,
            "current_settings": self.__frame_parameters.as_dict()
        }
        return settings_dict

    def get_frame_parameters_from_dict(self, d: typing.Mapping[str, typing.Any]) -> camera_base.CameraFrameParameters:
        return camera_base.CameraFrameParameters(d)

    def set_current_frame_parameters(self, frame_parameters: camera_base.CameraFrameParameters) -> None:
        """Set the current frame parameters.

        Fire the current frame parameters changed event and optionally the settings changed event.
        """
        self.__frame_parameters = copy.copy(frame_parameters)
        self.settings_changed_event.fire(self.__save_settings())
        self.current_frame_parameters_changed_event.fire(frame_parameters)

    def get_current_frame_parameters(self) -> camera_base.CameraFrameParameters:
        """Get the current frame parameters."""
        return camera_base.CameraFrameParameters(self.__frame_parameters.as_dict())

    def set_record_frame_parameters(self, frame_parameters: camera_base.CameraFrameParameters) -> None:
        """Set the record frame parameters.

        Fire the record frame parameters changed event and optionally the settings changed event.
        """
        self.__record_parameters = copy.copy(frame_parameters)
        self.record_frame_parameters_changed_event.fire(frame_parameters)

    def get_record_frame_parameters(self) -> camera_base.CameraFrameParameters:
        """Get the record frame parameters."""
        return self.__record_parameters

    def set_frame_parameters(self, settings_index: int, frame_parameters: camera_base.CameraFrameParameters) -> None:
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

    def get_frame_parameters(self, settings_index: int) -> camera_base.CameraFrameParameters:
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
    setattr(camera_device, "camera_panel_type", "ronchigram")
    camera_settings = CameraSettings("usim_ronchigram_camera")
    Registry.register_component(CameraModule("usim_stem_controller", camera_device, camera_settings), component_types)

    camera_device = Camera("usim_eels_camera", "eels", _("uSim EELS Camera"), instrument)
    setattr(camera_device, "camera_panel_type", "eels")
    camera_settings = CameraSettings("usim_eels_camera")
    Registry.register_component(CameraModule("usim_stem_controller", camera_device, camera_settings), component_types)
