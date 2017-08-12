from nion.swift.model import HardwareSource

from . import InstrumentDevice
from . import CameraDevice
from . import ScanDevice
from . import InstrumentPanel

def run():
    instrument = InstrumentDevice.Instrument()

    HardwareSource.HardwareSourceManager().register_instrument("usim_stem_controller", instrument)

    CameraDevice.run(instrument)
    ScanDevice.run(instrument)
    InstrumentPanel.run(instrument)
