from __future__ import annotations

# standard libraries
import gettext
import numpy.typing
import threading
import typing

# libraries
from nion.utils import Geometry

# other plug-ins
from nion.instrumentation import scan_base
from nion.device_kit import InstrumentDevice
from nion.device_kit import ScanDevice
from nion.usim_device import InstrumentDevice as InstrumentDevice_


_NDArray = numpy.typing.NDArray[typing.Any]
_DataElementType = typing.Dict[str, typing.Any]

_ = gettext.gettext


class ScanBoxSimulator(ScanDevice.ScanSimulatorLike):
    """Scan box simulator

    This class simulates the behavior of the scan box used in Nion microscopes. It supports two types of signals that
    can be used to syncronize another device with the scan:

    1. Blanker signal : This is an outgoing singal that is high during the time the beam is moving from the end of one line
                        to the beginning of the next line (flyback time). This signal is typically used to blank the beam
                        during flyback to reduce the electron dose on the sample. It can also be used to do "line-by-line"
                        synchronization with a camera. In this mode, the camera is triggered by the blanker signal and is
                        then free-running for one line in a synchronized acquisition. The implementation here is intended
                        to be used for the latter purpose: `blanker_signal_condition` will notify waiting threads before
                        the scan begins a new line.

    2. External clock : This is an incoming signal that signals the scan box to move the beam to the next probe location.
                        This signal is used for "pixel-by-pixel" synchronization with a camera. The camera will be
                        free-running for the whole synchronized acquisition and it needs to emit a sync signal for each
                        frame. To simulate this behavior, cameras should call `advance_pixel` once per frame during a
                        synchronized acquisition.
    """

    def __init__(self, scan_data_generator: ScanDevice.ScanDataGeneratorLike) -> None:
        self.__scan_data_generator = scan_data_generator
        self.__blanker_signal_condition = threading.Condition()
        self.__advance_pixel_lock = threading.RLock()
        self.__current_pixel_flat = 0
        self.__scan_shape_pixels = Geometry.IntSize()
        self.__pixel_size_nm = Geometry.FloatSize()
        self.flyback_pixels = 2
        self.__n_flyback_pixels = 0
        self.__current_line = 0
        self.external_clock = False

    def reset_frame(self) -> None:
        with self.__advance_pixel_lock:
            self.__current_pixel_flat = 0
            self.__current_line = 0
            self.__n_flyback_pixels = 0

    @property
    def scan_shape_pixels(self) -> Geometry.IntSize:
        return self.__scan_shape_pixels

    @scan_shape_pixels.setter
    def scan_shape_pixels(self, shape: typing.Union[Geometry.IntSize, Geometry.SizeIntTuple]) -> None:
        self.__scan_shape_pixels = Geometry.IntSize.make(shape)

    @property
    def pixel_size_nm(self) -> Geometry.FloatSize:
        return self.__pixel_size_nm

    @pixel_size_nm.setter
    def pixel_size_nm(self, size: typing.Union[Geometry.FloatSize, Geometry.SizeFloatTuple]) -> None:
        self.__pixel_size_nm = Geometry.FloatSize.make(size)

    @property
    def probe_position_pixels(self) -> Geometry.IntPoint:
        if self.__scan_shape_pixels.width != 0:
            current_pixel_flat = self.__current_pixel_flat
            return Geometry.IntPoint(y=current_pixel_flat // self.__scan_shape_pixels.width, x=current_pixel_flat % self.__scan_shape_pixels.width)
        return Geometry.IntPoint()

    @property
    def current_pixel_flat(self) -> int:
        return self.__current_pixel_flat

    @property
    def blanker_signal_condition(self) -> threading.Condition:
        """Blanker signal condition.

        This can be used like the blanker signal on real hardware: The signal is emitted when the beam moves from the
        last pixel in a line to the first pixel in the next line.
        To use it, you need to wait for the condition to be set. Note that you need to acquire the underlying
        lock of the condition before calling its "wait" method, so best practice is to use a "with" statement:

        .. code-block:: python

            with scan_box_simulator.blanker_signal_condition:
                scan_box_simulator.blanker_signal_condition.wait()

        """
        return self.__blanker_signal_condition

    def _advance_pixel(self, n: int) -> None:
        with self.__advance_pixel_lock:
            next_line = (self.__current_pixel_flat + n) // self.__scan_shape_pixels.width
            if next_line > self.__current_line:
                self.__n_flyback_pixels = 0
                self.__current_line = next_line
                with self.__blanker_signal_condition:
                    self.__blanker_signal_condition.notify_all()
            if self.__n_flyback_pixels < self.flyback_pixels:
                new_flyback_pixels = min(self.flyback_pixels - self.__n_flyback_pixels, n)
                n -= new_flyback_pixels
                self.__n_flyback_pixels += new_flyback_pixels
            self.__current_pixel_flat += n

    def advance_pixel(self) -> None:
        """Advance pixel.

        This equals the external clock input. From a camera simply call this function to simulate a sync pulse.
        """
        if self.external_clock:
            self._advance_pixel(1)

    def generate_scan_data(self, instrument: InstrumentDevice.Instrument, scan_frame_parameters: ScanDevice.ScanFrameParameters) -> numpy.typing.NDArray[numpy.float32]:
        return self.__scan_data_generator.generate_scan_data(instrument, scan_frame_parameters)


class ScanModule(scan_base.ScanModule):
    def __init__(self, instrument: InstrumentDevice_.Instrument) -> None:
        self.stem_controller_id = instrument.instrument_id
        self.device = ScanDevice.Device("usim_scan_device", _("uSim Scan"), instrument, ScanBoxSimulator(instrument.scan_data_generator))
        setattr(self.device, "priority", 20)
        scan_modes = (
            scan_base.ScanSettingsMode(_("Fast"), "fast", ScanDevice.ScanFrameParameters(pixel_size=(256, 256), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 0.1)),
            scan_base.ScanSettingsMode(_("Slow"), "slow", ScanDevice.ScanFrameParameters(pixel_size=(512, 512), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 0.4)),
            scan_base.ScanSettingsMode(_("Record"), "record", ScanDevice.ScanFrameParameters(pixel_size=(1024, 1024), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 1.0))
        )
        self.settings = scan_base.ScanSettings(scan_modes, lambda d: ScanDevice.ScanFrameParameters(d), 0, 2)
