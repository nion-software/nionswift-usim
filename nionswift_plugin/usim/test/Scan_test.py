import contextlib
import unittest
from nion.swift import Facade
from nion.instrumentation.test import ScanControl_test


class TestSimulatorScan(ScanControl_test.TestScanControlClass):

    def test_facade_record_data_with_immediate_close(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(scan_hardware_source.hardware_source_id, "~1.0")
            assert hardware_source_facade
            scan_frame_parameters = hardware_source_facade.get_frame_parameters_for_profile_by_index(2)
            scan_frame_parameters["external_clock_wait_time_ms"] = 20000 # int(camera_frame_parameters["exposure_ms"] * 1.5)
            scan_frame_parameters["external_clock_mode"] = 1
            scan_frame_parameters["ac_line_sync"] = False
            scan_frame_parameters["ac_frame_sync"] = False
            # this tests an issue for a race condition where thread for record task isn't started before the task
            # is canceled, resulting in the close waiting for the thread and the thread waiting for the acquire.
            # this reduces the problem, but it's still possible that during external sync, the acquisition starts
            # before being canceled and must timeout.
            with contextlib.closing(hardware_source_facade.create_record_task(scan_frame_parameters)) as task:
                pass


if __name__ == '__main__':
    unittest.main()
