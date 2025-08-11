from nion.device_kit import CameraDevice
from nion.usim_device import DeviceConfiguration
from nion.utils import Registry
from nionswift_plugin.usim import InstrumentPanel


def run() -> None:
    acquisition_context_configuration = DeviceConfiguration.AcquisitionContextConfiguration(set_configuration_location=False)
    Registry.register_component(acquisition_context_configuration.instrument, {"instrument_controller", "stem_controller"})
    component_types = {"camera_module"}  # the set of component types that this component represents
    setattr(acquisition_context_configuration.ronchigram_camera_device, "camera_panel_type", "ronchigram")
    Registry.register_component(CameraDevice.CameraModule("usim_stem_controller", acquisition_context_configuration._ronchigram_camera_device, acquisition_context_configuration._ronchigram_camera_settings), component_types)
    setattr(acquisition_context_configuration.eels_camera_device, "camera_panel_type", "eels")
    Registry.register_component(CameraDevice.CameraModule("usim_stem_controller", acquisition_context_configuration._eels_camera_device, acquisition_context_configuration._eels_camera_settings), component_types)
    Registry.register_component(acquisition_context_configuration.scan_module, {"scan_module"})
    InstrumentPanel.run(acquisition_context_configuration._instrument)


def stop() -> None:
    for component in Registry.get_components_by_type("camera_module"):
        Registry.unregister_component(component, {"camera_module"})
    Registry.unregister_component(Registry.get_component("scan_module"), {"scan_module"})
    Registry.unregister_component(Registry.get_component("stem_controller"), {"instrument_controller", "stem_controller"})
