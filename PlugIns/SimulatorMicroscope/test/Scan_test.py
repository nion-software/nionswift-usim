import unittest
from nion.swift.model import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation.test import ScanControl_test
from SimulatorMicroscope import InstrumentDevice
from SimulatorMicroscope import ScanDevice


class TestSimulatorScan(ScanControl_test.TestScanControlClass):

    def _setup_hardware_source(self, instrument) -> HardwareSource.HardwareSource:
        scan_adapter = scan_base.ScanAdapter(ScanDevice.Device(instrument), "usim_scan_device", "uSim Scan")
        scan_hardware_source = scan_base.ScanHardwareSource(scan_adapter, "usim_stem_controller")
        return scan_hardware_source

    def _close_hardware_source(self) -> None:
        pass

    def _setup_instrument(self):
        instrument = InstrumentDevice.Instrument()
        HardwareSource.HardwareSourceManager().register_instrument("usim_stem_controller", instrument)
        return instrument

    def _close_instrument(self, instrument) -> None:
        HardwareSource.HardwareSourceManager().unregister_instrument("usim_stem_controller")



if __name__ == '__main__':
    unittest.main()
