# standard libraries
import threading

class Instrument:

    def __init__(self):
        self.__camera_frame_event = threading.Event()

    def trigger_camera_frame(self) -> None:
        self.__camera_frame_event.set()

    def wait_for_camera_frame(self, timeout: float) -> bool:
        result = self.__camera_frame_event.wait(timeout)
        self.__camera_frame_event.clear()
        return result
