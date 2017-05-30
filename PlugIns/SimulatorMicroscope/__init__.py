from . import InstrumentDevice
from . import CameraDevice
from . import ScanDevice

def run():
    instrument = InstrumentDevice.Instrument()
    CameraDevice.run(instrument)
    ScanDevice.run(instrument)
