import contextlib
import unittest

from nion.swift.model import DocumentModel
from nion.swift.model import HardwareSource
from nion.swift import DocumentController
from nion.instrumentation import camera_base
from nion.instrumentation.test import CameraControl_test
from nionswift_plugin.nion_instrumentation_ui import CameraControlPanel
from nionswift_plugin.usim import CameraDevice
from nionswift_plugin.usim import InstrumentDevice


class TestCamera(CameraControl_test.TestCameraControlClass):

    def _setup_hardware_source(self, initialize: bool=True, is_eels: bool=False) -> (DocumentController.DocumentController, DocumentModel.DocumentModel, HardwareSource.HardwareSource, CameraControlPanel.CameraControlStateController):

        document_model = DocumentModel.DocumentModel()
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")

        # this is simulator specific. replace this code but be sure to set up self.exposure and blanked and positioned
        # initial settings.
        self.exposure = 0.04

        instrument = InstrumentDevice.Instrument()

        camera_adapter = camera_base.CameraAdapter("usim_ronchigram_camera", "ronchigram", "uSim Ronchigram Camera", CameraDevice.Camera("ronchigram", instrument))
        camera_hardware_source = camera_base.CameraHardwareSource(camera_adapter)

        if is_eels:
            camera_hardware_source.features["is_eels_camera"] = True
            camera_hardware_source.add_channel_processor(0, HardwareSource.SumProcessor(((0.25, 0.0), (0.5, 1.0))))
        camera_hardware_source.set_frame_parameters(0, camera_base.CameraFrameParameters({"exposure_ms": self.exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(1, camera_base.CameraFrameParameters({"exposure_ms": self.exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(2, camera_base.CameraFrameParameters({"exposure_ms": self.exposure * 1000 * 2, "binning": 1}))
        camera_hardware_source.set_selected_profile_index(0)

        HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)

        state_controller = CameraControlPanel.CameraControlStateController(camera_hardware_source, document_controller.queue_task, document_controller.document_model)
        if initialize:
            state_controller.initialize_state()

        return document_controller, document_model, camera_hardware_source, state_controller

    def test_camera_integrate_frames_updates_frame_count_by_integration_count(self):
        document_controller, document_model, hardware_source, state_controller = self._setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
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


if __name__ == '__main__':
    unittest.main()
