from . import InstrumentDevice
from . import CameraDevice
from . import ScanDevice
from . import InstrumentPanel

def run():
    instrument = InstrumentDevice.Instrument()
    CameraDevice.run(instrument)
    ScanDevice.run(instrument)
    InstrumentPanel.run(instrument)
