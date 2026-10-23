from __future__ import annotations

# standard libraries
import dataclasses
import math
import typing

import numpy
import numpy.typing

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import stem_controller
from nion.usim_device import CameraSimulator
from nion.usim_device import InstrumentDevice as InstrumentDevice_
from nion.utils import Geometry


_NDArray = numpy.typing.NDArray[typing.Any]

# the elementary charge, used to convert the beam current into electrons per second.
_ELECTRON_CHARGE_C = 1.602176634E-19

# the number of characteristic X-ray photons detected per beam electron per feature layer, summed over the lines of an
# element in proportion to their relative intensities. this is a simulator constant chosen to give count rates of a
# few tens of thousands per second at the default beam current; it is not physically calibrated.
_LINE_YIELD_PER_LAYER = 2E-5

# the Kramers' law constant: the number of continuum (bremsstrahlung) photons detected per beam electron per feature
# layer per eV of photon energy is this times the atomic number times (E0 - E) / E. chosen to give a peak to background
# ratio of roughly one hundred for the Si K-alpha line at 100 kV.
_KRAMERS_CONSTANT_PER_EV = 2E-12

# the detector window absorbs low energy photons. its transmission is modelled as exp(-(E_w / E)^3), where E_w is this
# energy; the cube follows the energy dependence of photoelectric absorption.
_WINDOW_ENERGY_EV = 250.0

# the Fano-limited part of the line width: FWHM(E)^2 = noise^2 + this * E, from 2.355^2 * F * epsilon with the Fano
# factor F = 0.115 and the electron-hole pair energy epsilon = 3.64 eV of silicon.
_FANO_FWHM_SQUARED_PER_EV = 2.32

# the energy of Mn K-alpha, at which a silicon drift detector's resolution is conventionally specified.
_MN_K_ALPHA_EV = 5895.0

# the most photons simulated individually per frame. above this, the expected recorded spectrum is calculated instead.
_MAX_PHOTONS_PER_FRAME = 200000

# the number of parts each channel is divided into when calculating the energy distribution of piled up pulses.
_SUB_CHANNEL_COUNT = 16

# the expected recorded spectrum includes pulses of more and more photons until their recorded rate falls below this
# fraction of the photon rate.
_NEGLIGIBLE_PULSE_RATE_FRACTION = 1E-9

# stray X-rays from the column and holder, present whenever the beam is not blanked, expressed as a number of feature
# layers of each element. Cu and Fe are the usual system peaks.
_SYSTEM_LAYERS: typing.Mapping[str, float] = {"Cu": 0.01, "Fe": 0.003}


@dataclasses.dataclass(frozen=True)
class XRayLine:
    """A characteristic X-ray emission line.

    energy_eV: the energy of the emitted photon.
    relative_intensity: the intensity of the line relative to the strongest line of its element.
    critical_energy_eV: the ionization energy of the shell the line comes from. a beam whose energy is below this
    does not excite the line.
    """
    energy_eV: float
    relative_intensity: float
    critical_energy_eV: float


@dataclasses.dataclass(frozen=True)
class ElementXRays:
    """The X-ray emission of an element.

    atomic_number: the atomic number, which scales the continuum emission.
    lines: the characteristic lines of the element.
    """
    atomic_number: int
    lines: typing.Sequence[XRayLine]


# characteristic line energies and critical ionization energies (eV) of the elements in the uSim samples and the
# system peaks, from standard X-ray data tables.
ELEMENT_XRAYS: typing.Mapping[str, ElementXRays] = {
    "C": ElementXRays(6, (XRayLine(277.0, 1.0, 284.0),)),  # K-alpha
    "Si": ElementXRays(14, (XRayLine(1740.0, 1.0, 1839.0), XRayLine(1836.0, 0.02, 1839.0))),  # K-alpha, K-beta
    "Fe": ElementXRays(26, (XRayLine(6404.0, 1.0, 7112.0), XRayLine(7058.0, 0.13, 7112.0))),  # K-alpha, K-beta
    "Ni": ElementXRays(28, (XRayLine(7478.0, 1.0, 8333.0), XRayLine(8265.0, 0.13, 8333.0))),  # K-alpha, K-beta
    "Cu": ElementXRays(29, (XRayLine(8048.0, 1.0, 8979.0), XRayLine(8905.0, 0.13, 8979.0))),  # K-alpha, K-beta
    "Ge": ElementXRays(32, (XRayLine(9886.0, 1.0, 11103.0), XRayLine(10982.0, 0.15, 11103.0))),  # K-alpha, K-beta
    "Au": ElementXRays(79, (XRayLine(2123.0, 0.6, 2206.0), XRayLine(9713.0, 1.0, 11919.0), XRayLine(11442.0, 0.18, 13734.0))),  # M-alpha, L-alpha, L-beta
}


class EDXSDetectorSimulator(CameraSimulator.CameraSimulator):
    """Simulate a silicon drift detector (SDD) for energy-dispersive X-ray spectroscopy (EDXS).

    The detector records a one row spectrum of the X-rays from the features under the probe and faint stray X-rays,
    including the effects of dead time and pulse pile-up.

    The parameters are the instrument, the sensor dimensions (one row, one column per channel), the width of a channel,
    the resolution (FWHM) at Mn K-alpha, the dead time, and the pulse pair resolution.

    Raises ValueError if the sensor does not have exactly one row and at least one channel, if the channel width is not
    positive, if the resolution is below the Fano limit at Mn K-alpha, or if the dead time or the pulse pair resolution
    is negative.
    """

    depends_on = ["probe_state", "probe_position", "features", "stage_position_m", "beam_shift_m", "EHT"]

    def __init__(self, instrument: InstrumentDevice_.Instrument, sensor_dimensions: Geometry.IntSize, *,
                 energy_per_channel_eV: float = 10.0, energy_resolution_eV: float = 130.0, dead_time_s: float = 5E-7,
                 pulse_pair_resolution_s: float = 1E-7) -> None:
        if sensor_dimensions.height != 1 or sensor_dimensions.width < 1:
            raise ValueError(f"An EDXS detector needs one row and at least one channel, not {sensor_dimensions}.")
        if energy_per_channel_eV <= 0.0:
            raise ValueError(f"An EDXS detector needs a positive channel width, not {energy_per_channel_eV} eV.")
        noise_fwhm_squared = energy_resolution_eV ** 2 - _FANO_FWHM_SQUARED_PER_EV * _MN_K_ALPHA_EV
        if noise_fwhm_squared < 0.0:
            raise ValueError(f"An EDXS detector resolution of {energy_resolution_eV} eV is below the Fano limit at Mn K-alpha.")
        if dead_time_s < 0.0:
            raise ValueError(f"An EDXS detector needs a dead time of zero or more, not {dead_time_s} s.")
        if pulse_pair_resolution_s < 0.0:
            raise ValueError(f"An EDXS detector needs a pulse pair resolution of zero or more, not {pulse_pair_resolution_s} s.")
        # each photon counts once, so get_total_counts gives the number of beam electrons.
        super().__init__(instrument, "edxs", sensor_dimensions, 1)
        self.__energy_per_channel_eV = energy_per_channel_eV
        self.__noise_fwhm_squared = noise_fwhm_squared
        self.__dead_time_s = dead_time_s
        self.__pulse_pair_resolution_s = pulse_pair_resolution_s

        # the random generator of the recorded photons.
        self.__random_generator = numpy.random.default_rng()

        # the spectrum of photons reaching the detector, recalculated when the instrument changes. the photon yield is
        # the number of photons per beam electron, and the photon distribution gives the probability of each channel.
        self.__photon_yield = 0.0
        self.__photon_distribution: _NDArray | None = None

        # the expected recorded counts per second at a photon rate too high to simulate each photon of an exposure,
        # kept until the photon rate or the photon distribution changes.
        self.__high_rate_photon_rate = 0.0
        self.__high_rate_recorded_spectrum_per_s: _NDArray | None = None

    def get_counts_per_electron(self) -> float | None:
        """Return None, since an X-ray detector cannot measure the beam current."""
        return None

    def get_dimensional_calibrations(self, readout_area: Geometry.IntRect | None, binning_shape: Geometry.IntSize | None) -> typing.Sequence[Calibration.Calibration]:
        """Return the calibrations of a frame with the readout area and binning, or of the whole sensor unbinned."""
        first_channel = readout_area.left if readout_area else 0
        channels_per_bin = binning_shape.width if binning_shape else 1
        return [
            Calibration.Calibration(),
            Calibration.Calibration(offset=first_channel * self.__energy_per_channel_eV, scale=channels_per_bin * self.__energy_per_channel_eV, units="eV"),
        ]

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, scan_context: stem_controller.ScanContext, parked_probe_position: Geometry.FloatPoint | None) -> DataAndMetadata.DataAndMetadata:
        """Return a frame recorded during the exposure, as counts per channel.

        The camera calls this from one thread at a time; see CameraSimulator.get_frame_data.
        """
        frame_settings = self._get_frame_settings(readout_area, binning_shape, exposure_s, scan_context, parked_probe_position)
        if frame_settings != self._last_frame_settings:
            self._needs_recalculation = True
            self._last_frame_settings = frame_settings

        photon_distribution = self.__photon_distribution
        if self._needs_recalculation or photon_distribution is None:
            # clear the flag before reading the instrument so that a change during the recalculation causes another recalculation
            self._needs_recalculation = False
            previous_photon_distribution = photon_distribution
            photon_spectrum = self._get_photon_spectrum(frame_settings)
            self.__photon_yield = float(numpy.sum(photon_spectrum))
            photon_distribution = photon_spectrum / self.__photon_yield if self.__photon_yield > 0.0 else numpy.zeros_like(photon_spectrum)
            self.__photon_distribution = photon_distribution
            # a spectrum image moves the probe at each pixel, usually within one feature, so the expected recorded
            # spectrum is kept while the photon distribution is unchanged.
            if previous_photon_distribution is None or not numpy.array_equal(photon_distribution, previous_photon_distribution):
                self.__high_rate_recorded_spectrum_per_s = None

        value_manager = CameraSimulator.get_value_manager(self.instrument)
        photon_count_mean = 0.0 if value_manager.is_blanked else self.__photon_yield * self.get_total_counts(exposure_s)
        spectrum = self.__record_spectrum(photon_count_mean, exposure_s, photon_distribution)

        # crop the sensor to the readout area, then bin.
        data = spectrum[numpy.newaxis, readout_area.left:readout_area.right][readout_area.top:readout_area.bottom]
        data = self._get_binned_data(data, binning_shape)

        intensity_calibration = Calibration.Calibration(units="counts")
        dimensional_calibrations = self.get_dimensional_calibrations(readout_area, binning_shape)
        return DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)

    def _get_photon_spectrum(self, frame_settings: CameraSimulator.FrameSettings) -> _NDArray:
        """Return the number of photons per beam electron reaching each channel of the sensor."""
        value_manager = CameraSimulator.get_value_manager(self.instrument)
        beam_energy_eV = self.instrument.GetVal("EHT")

        # count the layers of each element under the probe, starting from the stray X-rays.
        element_layers = dict(_SYSTEM_LAYERS)
        if frame_settings.current_probe_position is not None:
            scan_data_generator = typing.cast(InstrumentDevice_.ScanDataGenerator, self.instrument.scan_data_generator)
            offset_m = value_manager.actual_offset_m
            scan_context = frame_settings.scan_context
            scan_context_fov_size_nm = scan_context.fov_size_nm or Geometry.FloatSize()
            scan_context_center_nm = scan_context.center_nm or Geometry.FloatPoint()
            for feature in scan_data_generator.sample.features:
                if feature.intersects(offset_m, scan_context_fov_size_nm, scan_context_center_nm, frame_settings.current_probe_position):
                    for element in feature.elements:
                        element_layers[element] = element_layers.get(element, 0.0) + 1.0

        # the energy at the center of each channel. a channel is a bin, so channel i covers i to i + 1 channel widths.
        energies_eV = (numpy.arange(self._sensor_dimensions.width) + 0.5) * self.__energy_per_channel_eV
        window_transmission = numpy.exp(-(_WINDOW_ENERGY_EV / energies_eV) ** 3)

        photon_spectrum = numpy.zeros(self._sensor_dimensions.width, float)
        for element, layer_count in element_layers.items():
            element_xrays = ELEMENT_XRAYS.get(element)
            # an element without X-ray data contributes nothing; the samples only declare elements which have it.
            if element_xrays is None:
                continue
            for line in element_xrays.lines:
                if beam_energy_eV > line.critical_energy_eV:
                    sigma_eV = math.sqrt(self.__noise_fwhm_squared + _FANO_FWHM_SQUARED_PER_EV * line.energy_eV) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
                    line_shape = numpy.exp(-0.5 * ((energies_eV - line.energy_eV) / sigma_eV) ** 2) * self.__energy_per_channel_eV / (sigma_eV * math.sqrt(2.0 * math.pi))
                    line_transmission = math.exp(-(_WINDOW_ENERGY_EV / line.energy_eV) ** 3)
                    photon_spectrum += layer_count * _LINE_YIELD_PER_LAYER * line.relative_intensity * line_transmission * line_shape
            continuum = _KRAMERS_CONSTANT_PER_EV * element_xrays.atomic_number * numpy.maximum(beam_energy_eV - energies_eV, 0.0) / energies_eV
            photon_spectrum += layer_count * continuum * window_transmission * self.__energy_per_channel_eV
        return photon_spectrum

    def __record_spectrum(self, photon_count_mean: float, exposure_s: float, photon_distribution: _NDArray) -> _NDArray:
        """Return the counts per channel recorded during the exposure, for the mean number of photons reaching the detector."""
        if photon_count_mean <= 0.0 or exposure_s <= 0.0:
            return numpy.zeros(len(photon_distribution), float)

        if photon_count_mean <= _MAX_PHOTONS_PER_FRAME:
            photon_count = int(self.__random_generator.poisson(photon_count_mean))
            return self.__simulate_pulses(photon_count, exposure_s, photon_distribution)

        # with more photons, draw the frame's counts from the expected recorded spectrum of the same pulse model.
        photon_rate = photon_count_mean / exposure_s
        recorded_spectrum_per_s = self.__high_rate_recorded_spectrum_per_s
        if recorded_spectrum_per_s is None or photon_rate != self.__high_rate_photon_rate:
            recorded_spectrum_per_s = self.__get_recorded_spectrum_per_s(photon_rate, photon_distribution)
            self.__high_rate_photon_rate = photon_rate
            self.__high_rate_recorded_spectrum_per_s = recorded_spectrum_per_s
        return typing.cast(_NDArray, self.__random_generator.poisson(recorded_spectrum_per_s * exposure_s).astype(float))

    def __get_recorded_spectrum_per_s(self, photon_rate: float, photon_distribution: _NDArray) -> _NDArray:
        """Return the expected counts per second recorded in each channel at the photon rate, for the pulse model of __simulate_pulses."""
        # the photons arrive as a Poisson process, so the gaps between them are independent and exponential. a photon
        # starts a recorded pulse of k photons when the gap before it and the gap after the pulse's last photon are at
        # least the isolation time, and the k - 1 gaps within the pulse are shorter than the pulse pair resolution.
        isolation_s = max(self.__dead_time_s, self.__pulse_pair_resolution_s)
        pulse_rate = photon_rate * math.exp(-2.0 * photon_rate * isolation_s)
        piling_probability = 1.0 - math.exp(-photon_rate * self.__pulse_pair_resolution_s)

        # the energy of a photon is uniform within its channel, so it is spread evenly over the sub-channels. the energy
        # distribution of a pulse of k photons is the k-fold convolution of the photon distribution. a pulse beyond the
        # last channel is rejected and adding photons only raises the energy, so each convolution is truncated there.
        channel_count = len(photon_distribution)
        sub_channel_count = channel_count * _SUB_CHANNEL_COUNT
        transform_length = 2 * sub_channel_count
        photon_density = numpy.repeat(photon_distribution / _SUB_CHANNEL_COUNT, _SUB_CHANNEL_COUNT)
        photon_transform = numpy.fft.rfft(photon_density, transform_length)
        pulse_density = photon_density
        recorded_density = numpy.zeros(sub_channel_count, float)
        photon_count = 1
        while pulse_rate * float(numpy.sum(pulse_density)) > _NEGLIGIBLE_PULSE_RATE_FRACTION * photon_rate:
            # a sub-channel of each photon stands for its center, so a sum of k sub-channels lies (k - 1) / 2 sub-channels
            # below the sum of the photon energies.
            shift = (photon_count - 1) // 2
            recorded_density[shift:] += pulse_rate * pulse_density[:sub_channel_count - shift]
            pulse_rate *= piling_probability
            pulse_density = numpy.fft.irfft(numpy.fft.rfft(pulse_density, transform_length) * photon_transform, transform_length)[:sub_channel_count]
            photon_count += 1
        return typing.cast(_NDArray, numpy.maximum(recorded_density, 0.0).reshape(channel_count, _SUB_CHANNEL_COUNT).sum(axis=1))

    def __simulate_pulses(self, photon_count: int, duration_s: float, photon_distribution: _NDArray) -> _NDArray:
        """Return the counts per channel recorded from the photons.

        The photons arrive at random times during the duration, with energies drawn from the photon distribution.
        """
        channel_count = len(photon_distribution)
        if photon_count == 0:
            return numpy.zeros(channel_count, float)
        channels = self.__random_generator.choice(channel_count, size=photon_count, p=photon_distribution)
        photon_energies_eV = (channels + self.__random_generator.uniform(0.0, 1.0, size=photon_count)) * self.__energy_per_channel_eV
        arrival_times_s = numpy.sort(self.__random_generator.uniform(0.0, duration_s, size=photon_count))

        # a photon closer than the pulse pair resolution to the previous photon belongs to the same pulse, which records
        # the sum of their energies. this produces the sum peaks.
        starts_pulse = numpy.empty(photon_count, bool)
        starts_pulse[0] = True
        starts_pulse[1:] = numpy.diff(arrival_times_s) >= self.__pulse_pair_resolution_s
        pulse_indexes = numpy.cumsum(starts_pulse) - 1
        pulse_energies_eV = numpy.bincount(pulse_indexes, weights=photon_energies_eV)
        pulse_start_times_s = arrival_times_s[starts_pulse]
        pulse_end_times_s = arrival_times_s[numpy.append(numpy.flatnonzero(starts_pulse)[1:] - 1, photon_count - 1)]

        # both pulses of a pair closer than the dead time, from the end of the first to the start of the second, are
        # rejected, so the recorded count rate peaks and then falls as the photon rate grows. the detector is idle before
        # the duration starts and after it ends.
        is_separated = pulse_start_times_s[1:] - pulse_end_times_s[:-1] >= self.__dead_time_s
        is_recorded = numpy.append(True, is_separated) & numpy.append(is_separated, True)
        recorded_channels = numpy.floor(pulse_energies_eV[is_recorded] / self.__energy_per_channel_eV).astype(int)
        # a pulse beyond the last channel is rejected.
        return numpy.bincount(recorded_channels[recorded_channels < channel_count], minlength=channel_count).astype(float)
