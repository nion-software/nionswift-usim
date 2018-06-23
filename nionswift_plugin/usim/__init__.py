from nion.utils import Registry

from . import InstrumentDevice
from . import CameraDevice
from . import ScanDevice
from . import InstrumentPanel

def run():
    instrument = InstrumentDevice.Instrument("usim_stem_controller")
    Registry.register_component(instrument, {"stem_controller"})

    CameraDevice.run(instrument)
    ScanDevice.run(instrument)
    InstrumentPanel.run(instrument)
