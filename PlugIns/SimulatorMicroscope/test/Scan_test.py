import unittest
from nion.swift.model import HardwareSource
from Scan import ScanHardwareSource
from Scan.test import ScanControl_test
from SimulatorMicroscope import InstrumentDevice
from SimulatorMicroscope import ScanDevice


class TestSimulatorScan(ScanControl_test.TestScanControlClass):

    def _setup_hardware_source(self, instrument) -> HardwareSource.HardwareSource:
        scan_adapter = ScanHardwareSource.ScanAdapter(ScanDevice.Device(instrument), "usim_scan_device", "uSim Scan")
        scan_hardware_source = ScanHardwareSource.ScanHardwareSource(scan_adapter, "usim_stem_controller")
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
