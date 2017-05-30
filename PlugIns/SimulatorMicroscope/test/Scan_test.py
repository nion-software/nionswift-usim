import contextlib
import threading
import time
import unittest
import numpy
from nion.swift.model import HardwareSource
from Scan import ScanHardwareSource
from Scan.test import ScanControl_test
from SimulatorMicroscope import InstrumentDevice
from SimulatorMicroscope import ScanDevice


class TestSimulatorScan(ScanControl_test.TestScanControlClass):

    def _setup_hardware_source(self) -> HardwareSource.HardwareSource:

        instrument = InstrumentDevice.Instrument()
        scan_adapter = ScanHardwareSource.ScanAdapter(ScanDevice.Device(instrument), "usim_scan_device", "uSim Scan")
        scan_hardware_source = ScanHardwareSource.ScanHardwareSource(scan_adapter)
        return scan_hardware_source


if __name__ == '__main__':
    unittest.main()
