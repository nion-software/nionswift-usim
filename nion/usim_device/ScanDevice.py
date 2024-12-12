from __future__ import annotations

# standard libraries
import gettext
import numpy.typing
import typing

# other plug-ins
from nion.instrumentation import scan_base
from nion.device_kit import InstrumentDevice
from nion.device_kit import ScanDevice


_NDArray = numpy.typing.NDArray[typing.Any]
_DataElementType = typing.Dict[str, typing.Any]

_ = gettext.gettext


class ScanModule(scan_base.ScanModule):
    def __init__(self, instrument: InstrumentDevice.Instrument) -> None:
        self.stem_controller_id = instrument.instrument_id
        self.device = ScanDevice.Device("usim_scan_device", _("uSim Scan"), instrument)
        setattr(self.device, "priority", 20)
        scan_modes = (
            scan_base.ScanSettingsMode(_("Fast"), "fast", ScanDevice.ScanFrameParameters(pixel_size=(256, 256), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 0.1)),
            scan_base.ScanSettingsMode(_("Slow"), "slow", ScanDevice.ScanFrameParameters(pixel_size=(512, 512), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 0.4)),
            scan_base.ScanSettingsMode(_("Record"), "record", ScanDevice.ScanFrameParameters(pixel_size=(1024, 1024), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 1.0))
        )
        self.settings = scan_base.ScanSettings(scan_modes, lambda d: ScanDevice.ScanFrameParameters(d), 0, 2)
