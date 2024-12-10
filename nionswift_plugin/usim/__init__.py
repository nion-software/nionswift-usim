import gettext

from nion.utils import Geometry
from nion.utils import Registry

from . import CameraDevice
from . import EELSCameraSimulator
from . import InstrumentDevice
from . import InstrumentPanel
from . import RonchigramCameraSimulator
from . import ScanDevice


_ = gettext.gettext


class AcquisitionContextConfiguration:
    def __init__(self, *, sample_index: int = 0) -> None:
        self.instrument_id = "usim_stem_controller"
        instrument = InstrumentDevice.Instrument(self.instrument_id, InstrumentDevice.ScanDataGenerator(sample_index=sample_index))
        self.instrument = instrument
        self.ronchigram_camera_device_id = "usim_ronchigram_camera"
        self.eels_camera_device_id = "usim_eels_camera"
        self.scan_module = ScanDevice.ScanModule(instrument)

        ronchigram_simulator = RonchigramCameraSimulator.RonchigramCameraSimulator(instrument, Geometry.IntSize.make(instrument.camera_sensor_dimensions("ronchigram")), instrument.counts_per_electron, instrument.stage_size_nm)
        self.ronchigram_camera_device = CameraDevice.Camera("usim_ronchigram_camera", "ronchigram", _("uSim Ronchigram Camera"), ronchigram_simulator, instrument)
        self.ronchigram_camera_settings = CameraDevice.CameraSettings(self.ronchigram_camera_device_id)

        eels_camera_simulator = EELSCameraSimulator.EELSCameraSimulator(instrument, Geometry.IntSize.make(instrument.camera_sensor_dimensions("eels")), instrument.counts_per_electron)
        self.eels_camera_device = CameraDevice.Camera("usim_eels_camera", "eels", _("uSim EELS Camera"), eels_camera_simulator, instrument)
        self.eels_camera_settings = CameraDevice.CameraSettings(self.eels_camera_device_id)


def run() -> None:
    acquisition_context_configuration = AcquisitionContextConfiguration()

    Registry.register_component(acquisition_context_configuration.instrument, {"instrument_controller", "stem_controller"})

    component_types = {"camera_module"}  # the set of component types that this component represents

    setattr(acquisition_context_configuration.ronchigram_camera_device, "camera_panel_type", "ronchigram")
    Registry.register_component(CameraDevice.CameraModule("usim_stem_controller", acquisition_context_configuration.ronchigram_camera_device, acquisition_context_configuration.ronchigram_camera_settings), component_types)

    setattr(acquisition_context_configuration.eels_camera_device, "camera_panel_type", "eels")
    Registry.register_component(CameraDevice.CameraModule("usim_stem_controller", acquisition_context_configuration.eels_camera_device, acquisition_context_configuration.eels_camera_settings), component_types)

    Registry.register_component(acquisition_context_configuration.scan_module, {"scan_module"})

    InstrumentPanel.run(acquisition_context_configuration.instrument)


def stop() -> None:
    Registry.unregister_component(Registry.get_component("scan_module"), {"scan_module"})
