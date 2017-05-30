# standard libraries
import gettext
import numpy
import os
import threading
import time
import typing

# local libraries
from . import InstrumentDevice

# other plug-ins
from Camera import CameraHardwareSource

_ = gettext.gettext


class Camera(CameraHardwareSource.Camera):

    def __init__(self, instrument: InstrumentDevice.Instrument, image: numpy.ndarray):
        self.__instrument = instrument
        self.__sensor_dimensions = image.shape
        self.__readout_area = 0, 0, *image.shape  # TLBR
        self.__exposure_s = 0.1  # 100ms
        self.__integration_count = 1
        self.__data = image
        self.__data_buffer = None
        self.__frame_number = 0
        self.__thread = threading.Thread(target=self.__acquisition_thread)
        self.__thread_event = threading.Event()
        self.__has_data_event = threading.Event()
        self.__cancel = False
        self.__is_playing = False
        self.__thread.start()

    def close(self):
        self.__cancel = True
        self.__thread_event.set()
        self.__thread.join()
        self.__thread = None

    @property
    def sensor_dimensions(self) -> (int, int):
        return self.__sensor_dimensions

    @property
    def readout_area(self) -> (int, int, int, int):
        return self.__readout_area

    @readout_area.setter
    def readout_area(self, readout_area_TLBR: (int, int, int, int)) -> None:
        self.__readout_area = readout_area_TLBR

    @property
    def flip(self):
        return False

    @flip.setter
    def flip(self, do_flip):
        pass

    def start_live(self) -> None:
        if not self.__is_playing:
            self.__is_playing = True
            self.__thread_event.set()

    def stop_live(self) -> None:
        self.__is_playing = False
        self.__thread_event.set()

    def acquire_image(self) -> dict:
        data_buffer = None
        properties = dict()
        integration_count = self.__integration_count or 1
        for frame_number in range(integration_count):
            if not self.__has_data_event.wait(self.__exposure_s * 20):
                raise Exception("No simulator thread.")
            self.__has_data_event.clear()
            if data_buffer is None:
                data_buffer = numpy.copy(self.__data_buffer)
            else:
                data_buffer += self.__data_buffer
        self.__frame_number += 1
        properties["frame_number"] = self.__frame_number
        properties["integration_count"] = integration_count
        return {"data": data_buffer, "properties": properties}

    @property
    def calibration(self) -> typing.List[dict]:
        return [{}, {}]

    @property
    def mode(self):
        return "run"

    @mode.setter
    def mode(self, mode) -> None:
        pass

    @property
    def mode_as_index(self) -> int:
        return 0

    def get_exposure_ms(self, mode_id) -> float:
        return self.__exposure_s * 1000

    def set_exposure_ms(self, exposure_ms: float, mode_id) -> None:
        self.__exposure_s = exposure_ms / 1000

    def get_binning(self, mode_id) -> int:
        return 1

    def set_binning(self, binning: int, mode_id) -> None:
        pass

    def set_integration_count(self, integration_count: int, mode_id) -> None:
        self.__integration_count = integration_count

    @property
    def exposure_ms(self) -> float:
        return self.__exposure_s * 1000

    @exposure_ms.setter
    def exposure_ms(self, value: float) -> None:
        self.__exposure_s = value / 1000

    @property
    def binning(self) -> int:
        return 1

    @binning.setter
    def binning(self, value: int) -> None:
        pass

    @property
    def processing(self) -> str:
        return None

    @processing.setter
    def processing(self, value: str) -> None:
        pass

    @property
    def binning_values(self) -> typing.List[int]:
        return [1]

    def get_expected_dimensions(self, binning: int) -> (int, int):
        return self.__sensor_dimensions

    # def acquire_sequence_prepare(self) -> None:
    #     pass

    # def acquire_sequence(self, n: int) -> dict:
    #     pass

    def show_config_window(self) -> None:
        pass

    def start_monitor(self) -> None:
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
                self.__data_buffer = self.__data + numpy.random.randn(*self.__data.shape)
                elapsed = time.time() - start
                if not self.__thread_event.wait(max(self.__exposure_s - elapsed, 0)):
                    self.__has_data_event.set()
                    self.__instrument.trigger_camera_frame()
                else:
                    self.__thread_event.clear()


def _relativeFile(filename):
    dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(dir, filename)


def run(instrument: InstrumentDevice.Instrument) -> None:

    from nion.data import Image
    from nion.swift import Application
    from nion.swift.model import HardwareSource

    image = Image.read_grayscale_image_from_file(Application.app.ui, _relativeFile(os.path.join("resources", "GoldBalls.png")), dtype=numpy.float)

    camera_adapter = CameraHardwareSource.CameraAdapter("usim_ronchigram_camera", "ronchigram", _("uSim Ronchigram Camera"), Camera(instrument, image))
    camera_hardware_source = CameraHardwareSource.CameraHardwareSource(camera_adapter)
    HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)
