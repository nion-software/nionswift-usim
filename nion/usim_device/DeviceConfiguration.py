import gettext

from nion.device_kit import CameraDevice
from nion.instrumentation import camera_base
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.usim_device import EELSCameraSimulator
from nion.usim_device import InstrumentDevice
from nion.usim_device import RonchigramCameraSimulator
from nion.usim_device import ScanDevice
from nion.utils import Geometry


_ = gettext.gettext


class AcquisitionContextConfiguration:
    def __init__(self, *, sample_index: int = 0) -> None:
        self.instrument_id = "usim_stem_controller"
        value_manager = InstrumentDevice.ValueManager()
        axis_manager = InstrumentDevice.AxisManager()
        scan_data_generator = InstrumentDevice.ScanDataGenerator(sample_index=sample_index)
        instrument = InstrumentDevice.Instrument(self.instrument_id, value_manager, axis_manager, scan_data_generator)
        self._instrument = instrument
        self.instrument: stem_controller.STEMController = instrument
        self.ronchigram_camera_device_id = "usim_ronchigram_camera"
        self.eels_camera_device_id = "usim_eels_camera"
        self._scan_module = ScanDevice.ScanModule(instrument)
        self.scan_module: scan_base.ScanModule = self._scan_module

        ronchigram_simulator = RonchigramCameraSimulator.RonchigramCameraSimulator(instrument, Geometry.IntSize.make(instrument.camera_sensor_dimensions("ronchigram")), instrument.counts_per_electron, instrument.stage_size_nm)
        self._ronchigram_camera_device = CameraDevice.Camera("usim_ronchigram_camera", "ronchigram", _("uSim Ronchigram Camera"), ronchigram_simulator, instrument)
        self._ronchigram_camera_settings = CameraDevice.CameraSettings(self.ronchigram_camera_device_id)

        eels_camera_simulator = EELSCameraSimulator.EELSCameraSimulator(instrument, Geometry.IntSize.make(instrument.camera_sensor_dimensions("eels")), instrument.counts_per_electron)
        self._eels_camera_device = CameraDevice.Camera("usim_eels_camera", "eels", _("uSim EELS Camera"), eels_camera_simulator, instrument)
        self._eels_camera_settings = CameraDevice.CameraSettings(self.eels_camera_device_id)

        self.ronchigram_camera_device: camera_base.CameraDevice3 = self._ronchigram_camera_device
        self.ronchigram_camera_settings: camera_base.CameraSettings = self._ronchigram_camera_settings
        self.eels_camera_device: camera_base.CameraDevice3 = self._eels_camera_device
        self.eels_camera_settings: camera_base.CameraSettings = self._eels_camera_settings

        value_manager.ronchigram_camera = self._ronchigram_camera_device
        value_manager.eels_camera = self._eels_camera_device
