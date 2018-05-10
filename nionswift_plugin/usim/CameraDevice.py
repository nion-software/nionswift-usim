# standard libraries
import copy
import datetime
import gettext
import numpy
import threading
import time
import typing

# local libraries
from nion.utils import Geometry
from nion.utils import Registry
from nion.swift.model import ImportExportManager

from . import InstrumentDevice

# other plug-ins
from nion.instrumentation import camera_base

_ = gettext.gettext


class Camera(camera_base.Camera):
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
        self.__modes = ("run", "tune", "snap")
        self.__exposures_s = [0.1, 0.2, 0.5]
        self.__binnings = [2, 2, 1]
        self.__processing = None
        self.__mode = self.__modes[0]
        self.on_low_level_parameter_changed = None
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
        def set_exposure(m):
            self.exposure_ms = m
        def set_binning(m):
            self.binning = m
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

    @property
    def mode(self):
        """Return the current mode of the camera, as a case-insensitive string identifier."""
        return self.__mode

    @mode.setter
    def mode(self, mode) -> None:
        """Set the current mode of the camera, using a case-insensitive string identifier."""
        if True or mode.lower() != self.__mode.lower():
            self.__mode = mode.lower()
            if callable(self.on_low_level_parameter_changed):
                self.on_low_level_parameter_changed("mode")

    @property
    def mode_as_index(self) -> int:
        """Return the index of the current mode of the camera."""
        return self.__modes.index(self.__mode.lower())

    def get_exposure_ms(self, mode_id) -> float:
        """Return the exposure (in milliseconds) for the mode."""
        mode_index = self.__modes.index(mode_id.lower())
        return self.__exposures_s[mode_index] * 1000

    def set_exposure_ms(self, exposure_ms: float, mode_id) -> None:
        """Set the exposure (in milliseconds) for the mode."""
        mode_index = self.__modes.index(mode_id.lower())
        exposure_s = exposure_ms / 1000
        if int(exposure_s * 1E6) != int(self.__exposures_s[mode_index] * 1E6):
            self.__exposures_s[mode_index] = exposure_s
            if callable(self.on_low_level_parameter_changed):
                self.on_low_level_parameter_changed("exposureTimems")

    def get_binning(self, mode_id) -> int:
        """Return the binning for the mode."""
        mode_index = self.__modes.index(mode_id.lower())
        return self.__binnings[mode_index]

    def set_binning(self, binning: int, mode_id) -> None:
        """Set the binning for the mode."""
        mode_index = self.__modes.index(mode_id.lower())
        if binning != self.__binnings[mode_index]:
            self.__binnings[mode_index] = binning
            if callable(self.on_low_level_parameter_changed):
                self.on_low_level_parameter_changed("binning")
            # if parameter_name == "mode":
            #     self.on_mode_changed(self.mode)
            # elif parameter_name in ("exposureTimems", "binning"):
            #     for mode in enumerate(self.modes):
            #         self.on_mode_parameter_changed(mode, "exposure_ms", self.get_exposure_ms(mode))
            #         self.on_mode_parameter_changed(mode, "binning", self.get_binning(mode))

    def set_integration_count(self, integration_count: int, mode_id) -> None:
        """Set the integration code for the mode."""
        self.__integration_count = integration_count

    @property
    def exposure_ms(self) -> float:
        """Return the exposure (in milliseconds) for the current mode."""
        return self.get_exposure_ms(self.__mode)

    @exposure_ms.setter
    def exposure_ms(self, value: float) -> None:
        """Set the exposure (in milliseconds) for the current mode."""
        self.set_exposure_ms(value, self.__mode)

    @property
    def binning(self) -> int:
        """Return the binning for the current mode."""
        return self.get_binning(self.__mode)

    @binning.setter
    def binning(self, value: int) -> None:
        """Set the binning for the current mode."""
        self.set_binning(value, self.__mode)

    @property
    def processing(self) -> typing.Optional[str]:
        """Return processing actions for the current mode."""
        return self.__processing

    @processing.setter
    def processing(self, value: str) -> None:
        """Set processing actions for the current mode."""
        self.__processing = value

    @property
    def calibration(self) -> typing.List[dict]:
        """Return list of calibrations, one for each dimension."""
        readout_area = self.readout_area
        binning_shape = Geometry.IntSize(self.binning, self.binning if self.__symmetric_binning else 1)
        dimensional_calibrations = self.__instrument.get_camera_dimensional_calibrations(self.camera_type, Geometry.IntRect.from_tlbr(*readout_area), binning_shape)
        return [dimensional_calibration.rpc_dict for dimensional_calibration in dimensional_calibrations]

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
        mode_index = self.__modes.index(self.__mode)
        exposure_s = self.__exposures_s[mode_index]
        for frame_number in range(integration_count):
            if not self.__has_data_event.wait(exposure_s * 200):
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
        data_element["properties"]["counts_per_electron"] = self.__instrument.counts_per_electron
        data_element["intensity_calibration"] = dict()
        data_element["intensity_calibration"]["scale"] = xdata_buffer.intensity_calibration.scale
        data_element["intensity_calibration"]["units"] = xdata_buffer.intensity_calibration.units
        # data that has been binned vertically to a single row will be converted to 1D
        if xdata_buffer.dimensional_shape[0] == 1:
            data_element["data"] = numpy.squeeze(xdata_buffer.data)
            data_element["collection_dimension_count"] = 0
            data_element["datum_dimension_count"] = 1
        return data_element

    # def acquire_sequence_prepare(self, n: int) -> None:
    #     pass

    # def acquire_sequence(self, n: int) -> dict:
    #     pass

    def show_config_window(self) -> None:
        """Show a configuration dialog, if needed. Dialog can be modal or modeless."""
        pass

    def start_monitor(self) -> None:
        """Show a monitor dialog, if needed. Dialog can be modal or modeless."""
        pass

    def __acquisition_thread(self):
        while True:
            if self.__cancel:  # case where exposure was canceled.
                break
            self.__thread_event.wait()
            self.__thread_event.clear()
            if self.__cancel:
                break
            while self.__is_playing and not self.__cancel:
                start = time.time()
                readout_area = self.readout_area
                binning_shape = Geometry.IntSize(self.binning, self.binning if self.__symmetric_binning else 1)
                xdata = self.__instrument.get_camera_data(self.camera_type, Geometry.IntRect.from_tlbr(*readout_area), binning_shape, self.exposure_ms / 1000)
                elapsed = time.time() - start
                mode_index = self.__modes.index(self.__mode)
                exposure_s = self.__exposures_s[mode_index]
                wait_s = max(exposure_s - elapsed, 0)
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


def run(instrument: InstrumentDevice.Instrument) -> None:
    component_types = {"camera_device"}  # the set of component types that this component represents

    camera_device = Camera("usim_ronchigram_camera", "ronchigram", _("uSim Ronchigram Camera"), instrument)
    camera_device.stem_controller_id = "usim_stem_controller"
    camera_device.camera_panel_type = "ronchigram"

    Registry.register_component(camera_device, component_types)

    camera_device = Camera("usim_eels_camera", "eels", _("uSim EELS Camera"), instrument)
    camera_device.stem_controller_id = "usim_stem_controller"
    camera_device.camera_panel_type = "eels"

    Registry.register_component(camera_device, component_types)
