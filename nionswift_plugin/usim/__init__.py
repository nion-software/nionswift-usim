from nion.utils import Registry

from . import InstrumentDevice
from . import CameraDevice
from . import ScanDevice
from . import InstrumentPanel

def run() -> None:
    instrument = InstrumentDevice.Instrument("usim_stem_controller")
    Registry.register_component(instrument, {"instrument_controller", "stem_controller"})

    CameraDevice.run(instrument)
    ScanDevice.run(instrument)
    InstrumentPanel.run(instrument)
