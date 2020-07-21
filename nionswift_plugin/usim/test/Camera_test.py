import contextlib
import unittest

from nion.swift.model import DocumentModel
from nion.swift.model import HardwareSource
from nion.swift import DocumentController
from nion.instrumentation import camera_base
from nion.instrumentation.test import CameraControl_test
from nion.utils import Registry
from nionswift_plugin.nion_instrumentation_ui import CameraControlPanel
from nionswift_plugin.usim import CameraDevice
from nionswift_plugin.usim import InstrumentDevice


class TestCamera(CameraControl_test.TestCameraControlClass):

    def test_camera_integrate_frames_updates_frame_count_by_integration_count(self):
        with self._test_context() as test_context:
            hardware_source = test_context.hardware_source
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

    def test_camera_eels_works_when_it_produces_1d_data(self):
        with self._test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.binning = 256  # binning to 1d
            hardware_source.set_current_frame_parameters(frame_parameters)
            # two acquisitions will force the data item to be re-used, which triggered an error once
            self._acquire_one(document_controller, hardware_source)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(len(document_model.data_items[0].xdata.data_shape), 1)


if __name__ == '__main__':
    unittest.main()
