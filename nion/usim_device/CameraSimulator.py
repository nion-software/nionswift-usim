from __future__ import annotations

import numpy.typing
import typing
import copy
from dataclasses import dataclass

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.device_kit import InstrumentDevice
from nion.instrumentation import stem_controller
from nion.utils import Geometry
from nion.usim_device import InstrumentDevice as InstrumentDevice_


_NDArray = numpy.typing.NDArray[typing.Any]


@dataclass
class FrameSettings:
    readout_area: Geometry.IntRect
    binning_shape: Geometry.IntSize
    exposure_s: float
    scan_context: stem_controller.ScanContext
    current_probe_position: typing.Optional[Geometry.FloatPoint]
    sample_name: str


def get_value_manager(instrument: InstrumentDevice.Instrument) -> InstrumentDevice_.ValueManager:
    return typing.cast(InstrumentDevice_.ValueManager, instrument.value_manager)


class CameraSimulator:

    depends_on: typing.List[str] = list() # subclasses should define the controls and attributes they depend on here

    def __init__(self, instrument: InstrumentDevice_.Instrument, camera_type: str, sensor_dimensions: Geometry.IntSize, counts_per_electron: int) -> None:
        self.__instrument = instrument
        self._camera_type = camera_type
        self._sensor_dimensions = sensor_dimensions
        self._counts_per_electron = counts_per_electron
        self._needs_recalculation = True
        self._last_frame_settings: typing.Optional[FrameSettings] = None

        def property_changed(name: str) -> None:
            if name in self.depends_on:
                self._needs_recalculation = True

        self.__property_changed_event_listener = get_value_manager(instrument).property_changed_event.listen(property_changed)

        # we also need to inform the cameras about changes to the (parked) probe position
        def probe_state_changed(probe_state: str, probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
            property_changed("probe_state")
            property_changed("probe_position")

        self.__probe_state_changed_event_listener = instrument.probe_state_changed_event.listen(probe_state_changed)

    def close(self) -> None:
        self.__property_changed_event_listener.close()
        self.__property_changed_event_listener = typing.cast(typing.Any, None)
        self.__probe_state_changed_event_listener.close()
        self.__probe_state_changed_event_listener = typing.cast(typing.Any, None)

    @property
    def _camera_shape(self) -> Geometry.IntSize:
        return self._sensor_dimensions

    @property
    def instrument(self) -> InstrumentDevice_.Instrument:
        return self.__instrument

    def get_dimensional_calibrations(self, readout_area: typing.Optional[Geometry.IntRect], binning_shape: typing.Optional[Geometry.IntSize]) -> typing.Sequence[Calibration.Calibration]:
        """
        Subclasses should override this method
        """
        return [Calibration.Calibration(), Calibration.Calibration()]

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, scan_context: stem_controller.ScanContext, probe_position: typing.Optional[Geometry.FloatPoint]) -> DataAndMetadata.DataAndMetadata:
        """
        Subclasses must override this method
        """
        raise NotImplementedError()

    def get_total_counts(self, exposure_s: float) -> float:
        beam_current_pa = self.instrument.GetVal("BeamCurrent") * 1E12
        e_per_pa = 6.241509074E18 / 1E12
        return beam_current_pa * e_per_pa * exposure_s * self._counts_per_electron

    def _get_binned_data(self, data: _NDArray, binning_shape: Geometry.IntSize) -> _NDArray:
        if binning_shape.height > 1:
            # do binning by taking the binnable area, reshaping last dimension into bins, and taking sum of those bins.
            data_T = data.T
            data = data_T[:(data_T.shape[0] // binning_shape.height) * binning_shape.height].reshape(data_T.shape[0], -1, binning_shape.height).sum(axis=-1).T
            if binning_shape.width > 1:
                data = data[:(data.shape[-1] // binning_shape.width) * binning_shape.width].reshape(data.shape[0], -1, binning_shape.width).sum(axis=-1)
        return data

    def _get_frame_settings(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, scan_context: stem_controller.ScanContext, parked_probe_position: typing.Optional[Geometry.FloatPoint] = None) -> FrameSettings:
        scan_hardware_source = self.instrument.scan_controller
        scan_device = scan_hardware_source.scan_device if scan_hardware_source else None
        probe_position: typing.Optional[Geometry.FloatPoint] = Geometry.FloatPoint(0.5, 0.5)
        if scan_device:
            if self.instrument.probe_state == "scanning" and hasattr(scan_device, "current_probe_position"):
                probe_position = scan_device.current_probe_position
            elif self.instrument.probe_state == "parked" and parked_probe_position is not None:
                probe_position = parked_probe_position
        scan_data_generator = typing.cast(InstrumentDevice_.ScanDataGenerator, self.instrument.scan_data_generator)
        return FrameSettings(readout_area, binning_shape, exposure_s, copy.deepcopy(scan_context), probe_position, scan_data_generator.sample.title)
