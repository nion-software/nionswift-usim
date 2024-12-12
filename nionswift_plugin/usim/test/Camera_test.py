import numpy
import threading
import time
import unittest

from nion.instrumentation.test import CameraControl_test
from nion.instrumentation.test import AcquisitionTestContext
from nion.usim_device import DeviceConfiguration
from nion.utils import Geometry


class TestCamera(CameraControl_test.TestCameraControlClass):

    def _test_context(self, *, is_eels: bool = False) -> AcquisitionTestContext.AcquisitionTestContext:
        return AcquisitionTestContext.AcquisitionTestContext(DeviceConfiguration.AcquisitionContextConfiguration(), is_eels=is_eels)

    def test_camera_integrate_frames_updates_frame_count_by_integration_count(self) -> None:
        with self._test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.integration_count = 4
            hardware_source.set_current_frame_parameters(frame_parameters)
            hardware_source.start_playing()
            try:
                frame0_integration_count = hardware_source.get_next_xdatas_to_finish()[0].metadata["hardware_source"]["integration_count"]
                frame1_integration_count = hardware_source.get_next_xdatas_to_finish()[0].metadata["hardware_source"]["integration_count"]
                self.assertEqual(frame0_integration_count, 4)
                self.assertEqual(frame1_integration_count, 4)
            finally:
                hardware_source.abort_playing(sync_timeout=3.0)

    def test_camera_eels_works_when_it_produces_1d_data(self) -> None:
        with self._test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.binning = 256  # binning to 1d
            hardware_source.set_current_frame_parameters(frame_parameters)
            # two acquisitions will force the data item to be re-used, which triggered an error once
            self._acquire_one(document_controller, hardware_source)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(len(document_model.data_items[0].xdata.data_shape), 1)

    def test_camera_eels_connects_to_probe_position(self) -> None:
        # ensure that the probe position is connected to the EELS camera data
        with self._test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            stem_controller = test_context.instrument
            stem_controller.probe_position = Geometry.FloatPoint(x=0.5, y=0.5)
            eels_camera_hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, eels_camera_hardware_source)
            self.assertLess(10e6, numpy.average(document_controller.document_model.data_items[-1].xdata.data[30:45]))
            stem_controller.probe_position = Geometry.FloatPoint(x=0.25, y=0.25)
            self._acquire_one(document_controller, eels_camera_hardware_source)
            self.assertGreater(1e3, numpy.average(document_controller.document_model.data_items[-1].xdata.data[30:45]))

    def test_camera_waits_for_external_trigger(self) -> None:
        for external_trigger in [False, True]:
            with self.subTest(external_trigger=external_trigger):
                with self._test_context(is_eels=True) as test_context:
                    document_controller = test_context.document_controller
                    document_model = test_context.document_model
                    hardware_source = test_context.camera_hardware_source
                    scan = test_context.scan_hardware_source
                    frame_parameters = hardware_source.get_frame_parameters(0)
                    frame_parameters.binning = 1
                    frame_parameters.processing = "sum_project"
                    frame_parameters.exposure_ms = 100
                    hardware_source.set_current_frame_parameters(frame_parameters)
                    hardware_source.camera._external_trigger = external_trigger
                    sequence_data_elements = None
                    sequence_time = 0.
                    camera_event = threading.Event()
                    def acquire() -> None:
                        nonlocal sequence_data_elements, sequence_time
                        starttime = time.time()
                        sequence_data_elements = hardware_source.acquire_sequence(10)
                        sequence_time = time.time() - starttime
                        camera_event.set()
                    scan_frame_parameters = scan.get_frame_parameters(0)
                    scan_frame_parameters.size = (1, 10)
                    scan_frame_parameters.pixel_time_us = 100000
                    scan.set_current_frame_parameters(scan_frame_parameters)
                    threading.Thread(target=acquire).start()
                    time.sleep(3)
                    xdata_list = scan.record_immediate(scan_frame_parameters)
                    self.assertTrue(camera_event.wait(10))
                    if external_trigger:
                        self.assertGreater(sequence_time, 3.0)
                    else:
                        self.assertLess(sequence_time, 3.0)


if __name__ == '__main__':
    unittest.main()
