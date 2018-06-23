import contextlib
import unittest
from nion.swift import Facade
from nion.swift.model import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation.test import ScanControl_test
from nion.utils import Registry
from nionswift_plugin.usim import InstrumentDevice
from nionswift_plugin.usim import ScanDevice


class TestSimulatorScan(ScanControl_test.TestScanControlClass):

    def _setup_hardware_source(self, instrument) -> HardwareSource.HardwareSource:
        stem_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id("usim_stem_controller")
        scan_hardware_source = scan_base.ScanHardwareSource(stem_controller, ScanDevice.Device(instrument), "usim_scan_device", "uSim Scan")
        return scan_hardware_source

    def _close_hardware_source(self) -> None:
        pass

    def _setup_instrument(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        Registry.register_component(instrument, {"stem_controller"})
        return instrument

    def _close_instrument(self, instrument) -> None:
        HardwareSource.HardwareSourceManager().unregister_instrument("usim_stem_controller")

    def test_facade_record_data_with_immediate_close(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
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
