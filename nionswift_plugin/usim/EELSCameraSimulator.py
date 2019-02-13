# standard libraries
import math
import numpy
import scipy.ndimage.interpolation
import scipy.stats

from nion.data import Calibration
from nion.data import DataAndMetadata

from nion.utils import Geometry

from . import CameraSimulator
from . import Noise


def plot_powerlaw(data: numpy.ndarray, multiplier: float, energy_calibration: Calibration.Calibration, offset_eV: float, onset_eV: float) -> None:
    # calculate the range
    # 1 represents 0eV, 0 represents 4000eV
    # TODO: sub-pixel accuracy
    energy_range_eV = [energy_calibration.convert_to_calibrated_value(0), energy_calibration.convert_to_calibrated_value(data.shape[0])]
    envelope = scipy.stats.norm(loc=offset_eV, scale=onset_eV).cdf(numpy.linspace(energy_range_eV[0], energy_range_eV[1], data.shape[0]))
    max_ev = 4000
    powerlaw_dist = scipy.stats.powerlaw(8, loc=0, scale=max_ev)  # this is an increasing function; must be reversed below; 8 is arbitrary but looks good
    powerlaw = powerlaw_dist.pdf(numpy.linspace(max_ev - energy_range_eV[0], max_ev - energy_range_eV[1], data.shape[0]))
    data += envelope * multiplier * powerlaw


def plot_norm(data: numpy.ndarray, multiplier: float, energy_calibration: Calibration.Calibration, energy_eV: float, energy_width_eV: float) -> None:
    # calculate the range
    # 1 represents 0eV, 0 represents 4000eV
    # TODO: sub-pixel accuracy
    data_range = [0, data.shape[0]]
    energy_range_eV = [energy_calibration.convert_to_calibrated_value(data_range[0]), energy_calibration.convert_to_calibrated_value(data_range[1])]
    norm = scipy.stats.norm(loc=energy_eV, scale=energy_width_eV)
    data += multiplier * norm.pdf(numpy.linspace(energy_range_eV[0], energy_range_eV[1], data_range[1])) / norm.pdf(energy_eV)


def plot_spectrum(feature, data: numpy.ndarray, multiplier: float, energy_calibration: Calibration.Calibration) -> None:
    for edge_eV, onset_eV in feature.edges:
        # print(f"edge_eV {edge_eV} onset_eV {onset_eV}")
        strength = multiplier * 0.1
        plot_powerlaw(data, strength, energy_calibration, edge_eV, onset_eV)
    for n in range(1, feature.plurality + 1):
        plot_norm(data, multiplier / math.factorial(n), energy_calibration, feature.plasmon_eV * n, math.sqrt(feature.plasmon_eV))


class EELSCameraSimulator(CameraSimulator.CameraSimulator):
    depends_on = ["is_slit_in", "probe_state", "probe_position", "live_probe_position", "is_blanked", "ZLPoffset",
                  "stage_position_m", "beam_shift_m", "features", "energy_offset_eV", "energy_per_channel_eV",
                  "beam_current"]

    def __init__(self, instrument: "Instrument", sensor_dimensions: Geometry.IntSize, counts_per_electron: int):
        super().__init__(instrument, "eels", sensor_dimensions, counts_per_electron)
        self.__cached_frame = None
        self.__data_scale = 1.0
        self.noise = Noise.PoissonNoise()

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, last_scan_params=None):
        """
        Features at the probe position will add plasmons and edges in addition to a ZLP.

        There are two inputs to this model: the beam current and the T/l (thickness / mean free path).

        The sum of the spectrum data should add up to the beam current (using counts per electron and conversion from
        electrons to amps).

        The natural log of the ratio of the sum of the spectrum to the sum of the ZLP should equal thickness / mean free
        path.

        The strategy is to have low level routines for adding the shapes of the ZLP (gaussian normal) and plasmons and
        edges (power law multiplied by integrated gaussian normal) and then scaling these shapes such that they satisfy
        the conditions above.

        A complication of this is that the specified energy range may not include the ZLP. So two spectrums are built:
        the one for caller and the one for reference. The reference one is used for calculating the scaling of the ZLP
        and edges, which are then applied to the spectrum for the caller.

        If we define the following values:
            z = sum/integration of unscaled ZLP gaussian
            f = sum/integration of unscaled plasmons/edges
            P = target count such that P / counts_per_electron matches beam current
            T = thickness (nm)
            L = lambda (mean_free_path_nm)
            T/l = thickness / lambda (mean free path)
        then we can solve for two unknowns:
            A = scale of ZLP
            B = scale of plasmons/edges
        using the two equations:
            Az + Bf = P (beam current)
            ln(P / Az) = T/l => P / Az = exp(T/l) (thickness = natural log of ratio of total counts to ZLP counts)
        solving:
            A = P / exp(T/l) / z
            B = (P - Az) / f
        """

        # check if one of the arguments has changed since last call
        new_frame_settings = [readout_area, binning_shape, exposure_s, last_scan_params]
        if new_frame_settings != self._last_frame_settings:
            self._needs_recalculation = True
        self._last_frame_settings = new_frame_settings

        if self._needs_recalculation or self.__cached_frame is None:
            data = numpy.zeros(tuple(self._sensor_dimensions), numpy.float)
            slit_attenuation = 10 if self.is_slit_in else 1
            intensity_calibration = Calibration.Calibration(units="counts")
            dimensional_calibrations = self.get_dimensional_calibrations(readout_area, binning_shape)
            probe_position = Geometry.FloatPoint(0.5, 0.5)
            if self.is_blanked:
                probe_position = None
            elif self.probe_state == "scanning":
                probe_position = self.live_probe_position
            elif self.probe_state == "parked" and self.probe_position is not None:
                probe_position = self.probe_position

            # typical thickness over mean free path (T/l) will be 0.5
            mean_free_path_nm = 100  # nm. (lambda values from back of Edgerton)
            thickness_per_layer_nm = 30  # nm

            # this is the number of pixel counts expected if the ZLP is visible in vacuum for the given exposure
            # and beam current (in get_total_counts).
            target_pixel_count = self.get_total_counts(exposure_s) / data.shape[0]

            # grab the specific calibration for the energy direction and offset by ZLPoffset
            used_calibration = dimensional_calibrations[1]
            used_calibration.offset = self.instrument.get_control("ZLPoffset").local_value

            if last_scan_params is not None and probe_position is not None:

                # make a buffer for the spectrum
                spectrum = numpy.zeros((data.shape[1], ), numpy.float)

                # configure a calibration for the reference spectrum. then plot the ZLP on the reference data. sum it to
                # get the zlp_pixel_count and the zlp_scale. this is the value to multiple zlp data by to scale it so
                # that it will produce the target pixel count. since we will be storing the spectra in a 2d array,
                # divide by the height of that array so that when it is summed, the value comes out correctly.
                zlp0_calibration = Calibration.Calibration(scale=used_calibration.scale, offset=-20)
                spectrum_ref = numpy.zeros((int(zlp0_calibration.convert_from_calibrated_value(-20 + 1000) - zlp0_calibration.convert_from_calibrated_value(-20)), ), numpy.float)
                plot_norm(spectrum_ref, 1.0, Calibration.Calibration(scale=used_calibration.scale, offset=-20), 0, 0.5 / slit_attenuation)
                zlp_ref_pixel_count = float(numpy.sum(spectrum_ref))

                # build the spectrum and reference spectrum by adding the features. the data is unscaled.
                spectrum_ref = numpy.zeros((int(zlp0_calibration.convert_from_calibrated_value(-20 + 1000) - zlp0_calibration.convert_from_calibrated_value(-20)), ), numpy.float)
                size, fov_size_nm, center_nm = last_scan_params  # get these from last scan
                offset_m = self.stage_position_m - self.beam_shift_m  # get this from current values
                feature_layer_count = 0
                for index, feature in enumerate(self.instrument.sample.features):
                    if feature.intersects(offset_m, fov_size_nm, center_nm, Geometry.FloatPoint.make(probe_position)):
                        plot_spectrum(feature, spectrum, 1.0, used_calibration)
                        plot_spectrum(feature, spectrum_ref, 1.0, zlp0_calibration)
                        feature_layer_count += 1
                feature_pixel_count = max(numpy.sum(spectrum_ref), 0.01)

                # make the calculations for A, B (zlp_scale and feature_scale).
                thickness_factor = feature_layer_count * thickness_per_layer_nm / mean_free_path_nm
                zlp_scale = target_pixel_count / math.exp(thickness_factor) / zlp_ref_pixel_count
                feature_scale = (target_pixel_count - (target_pixel_count / math.exp(thickness_factor))) / feature_pixel_count
                # print(f"thickness_factor {thickness_factor}")

                # apply the scaling. spectrum holds the features at this point, but not the ZLP. just multiple by
                # feature_scale to make the feature part of the spectrum final. then plot the ZLP scaled by zlp_scale.
                spectrum *= feature_scale
                # print(f"sum {numpy.sum(spectrum) * data.shape[0]}")
                # print(f"zlp_ref_pixel_count {zlp_ref_pixel_count} feature_pixel_count {feature_pixel_count}")
                # print(f"zlp_scale {zlp_scale} feature_scale {feature_scale}")
                plot_norm(spectrum, zlp_scale, used_calibration, 0, 0.5 / slit_attenuation)
                # print(f"sum {numpy.sum(spectrum) * data.shape[0]}")
                # print(f"target_pixel_count {target_pixel_count}")

                # finally, store the spectrum into each row of the data
                data[:, ...] = spectrum

                # spectrum_pixel_count = float(numpy.sum(spectrum)) * data.shape[0]
                # print(f"z0 {zlp_ref_pixel_count * data.shape[0]} / {used_calibration.offset}")
                # print(f"beam current {self.instrument.beam_current * 1e12}pA")
                # print(f"current {spectrum_pixel_count / exposure_s / self.instrument.counts_per_electron / 6.242e18 * 1e12:#.2f}pA")
                # print(f"target {target_pixel_count}  actual {spectrum_pixel_count}")
                # print(f"s {spectrum_pixel_count} z {zlp_ref_pixel_count * zlp_scale * data.shape[0]}")
                # print(f"{math.log(spectrum_pixel_count / (zlp_ref_pixel_count * zlp_scale * data.shape[0]))} {thickness_factor}")

            data = self._get_binned_data(data, binning_shape)

            self.__cached_frame = DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)
            self.__data_scale = self.get_total_counts(exposure_s) / target_pixel_count / slit_attenuation / self._sensor_dimensions[0]
            self._needs_recalculation = False

        self.noise.poisson_level = self.__data_scale
        return self.noise.apply(self.__cached_frame)

    def get_dimensional_calibrations(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize):
        energy_offset_eV = self.energy_offset_eV
        # energy_offset_eV += random.uniform(-1, 1) * self.__energy_per_channel_eV * 5
        dimensional_calibrations = [
            Calibration.Calibration(),
            Calibration.Calibration(offset=energy_offset_eV, scale=self.energy_per_channel_eV, units="eV")
        ]
        return dimensional_calibrations
