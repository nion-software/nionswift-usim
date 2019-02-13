# standard libraries
import numpy

from nion.data import DataAndMetadata

from nion.utils import Geometry


class CameraSimulator:

    depends_on = list() # subclasses should define the controls and attributes they depend on here

    def __init__(self, instrument: "Instrument", camera_type: str, sensor_dimensions: Geometry.IntSize, counts_per_electron: int):
        self.__instrument = instrument
        self._camera_type = camera_type
        self._sensor_dimensions = sensor_dimensions
        self._counts_per_electron = counts_per_electron
        self._needs_recalculation = True
        self._last_frame_settings = [Geometry.IntRect((0, 0), (0, 0)), Geometry.IntSize(), 0.0, None]

        def property_changed(name):
            if name in self.depends_on:
                self._needs_recalculation = True

        self.__property_changed_event_listener = instrument.property_changed_event.listen(property_changed)

        # we also need to inform the cameras about changes to the (parked) probe position
        def probe_state_changed(probe_state, probe_position):
            property_changed("probe_state")
            property_changed("probe_position")

        self.__probe_state_changed_event_listener = instrument.probe_state_changed_event.listen(probe_state_changed)

    def close(self):
        self.__property_changed_event_listener.close()
        self.__property_changed_event_listener = None
        self.__probe_state_changed_event_listener.close()
        self.__probe_state_changed_event_listener = None

    @property
    def _camera_shape(self) -> Geometry.IntSize:
        return self._sensor_dimensions

    @property
    def instrument(self) -> "Instrument":
        return self.__instrument

    def __getattr__(self, attr):
        if attr in self.depends_on:
            return getattr(self.__instrument, attr)
        raise AttributeError(attr)

    def get_dimensional_calibrations(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize):
        """
        Subclasses should override this method
        """
        return [{}, {}]

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, last_scan_params=None) -> DataAndMetadata.DataAndMetadata:
        """
        Subclasses must override this method
        """
        raise NotImplementedError

    def get_total_counts(self, exposure_s: float) -> float:
        beam_current_pa = self.beam_current * 1E12
        e_per_pa = 6.242E18 / 1E12
        return beam_current_pa * e_per_pa * exposure_s * self._counts_per_electron

    def _get_binned_data(self, data: numpy.ndarray, binning_shape: Geometry.IntSize) -> numpy.ndarray:
        if binning_shape.height > 1:
            # do binning by taking the binnable area, reshaping last dimension into bins, and taking sum of those bins.
            data_T = data.T
            data = data_T[:(data_T.shape[0] // binning_shape.height) * binning_shape.height].reshape(data_T.shape[0], -1, binning_shape.height).sum(axis=-1).T
            if binning_shape.width > 1:
                data = data[:(data.shape[-1] // binning_shape.width) * binning_shape.width].reshape(data.shape[0], -1, binning_shape.width).sum(axis=-1)
        return data
