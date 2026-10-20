import contextlib
import numpy
import numpy.typing
import typing
import unittest

from nion.data import DataAndMetadata
from nion.usim_device import EDXSDetectorSimulator
from nion.usim_device import InstrumentDevice
from nion.utils import Geometry


_NDArray = numpy.typing.NDArray[typing.Any]

SENSOR_SIZE = Geometry.IntSize(1, 2048)

READOUT_AREA = Geometry.IntRect(origin=Geometry.IntPoint(), size=SENSOR_SIZE)

ENERGY_PER_CHANNEL_EV = 10.0


def make_instrument() -> InstrumentDevice.Instrument:
    instrument = InstrumentDevice.Instrument("usim_stem_controller", InstrumentDevice.ValueManager(), InstrumentDevice.AxisManager(), InstrumentDevice.ScanDataGenerator())
    instrument._update_scan_context(Geometry.IntSize(256, 256), Geometry.FloatPoint(), 10, 0.0)
    return instrument


def get_value_manager(instrument: InstrumentDevice.Instrument) -> InstrumentDevice.ValueManager:
    return typing.cast(InstrumentDevice.ValueManager, instrument.value_manager)


# move the stage so that the probe, at the center of the scan, is on a feature made only of the element.
def place_probe_on_element(instrument: InstrumentDevice.Instrument, element: str) -> None:
    features = typing.cast(InstrumentDevice.ScanDataGenerator, instrument.scan_data_generator).sample.features
    for feature in features:
        if feature.elements == (element,) and not any(other is not feature and other.get_feature_rect_m().contains_point(feature.position_m) for other in features):
            get_value_manager(instrument).stage_position_m = -feature.position_m
            return
    raise ValueError(f"No feature of {element} alone.")


def acquire_spectrum(simulator: EDXSDetectorSimulator.EDXSDetectorSimulator, instrument: InstrumentDevice.Instrument, exposure_s: float = 0.1) -> _NDArray:
    xdata = simulator.get_frame_data(READOUT_AREA, Geometry.IntSize(1, 1), exposure_s, instrument.scan_context, instrument.probe_position)
    return typing.cast(_NDArray, xdata.data[0])


# the counts within 60 eV of the energy. a channel is a bin, so the energy is in channel floor(E / channel width).
def window_counts(spectrum: _NDArray, energy_eV: float) -> float:
    channel = int(energy_eV // ENERGY_PER_CHANNEL_EV)
    return float(numpy.sum(spectrum[channel - 6:channel + 7]))


# the counts of a window 300 eV either side of the energy, averaged.
def background_counts(spectrum: _NDArray, energy_eV: float) -> float:
    return (window_counts(spectrum, energy_eV - 300.0) + window_counts(spectrum, energy_eV + 300.0)) / 2


# the calibrated energy of the centroid of the strongest line, from the channels within 150 eV of its peak.
def strongest_line_energy(xdata: DataAndMetadata.DataAndMetadata) -> float:
    spectrum = numpy.reshape(xdata.data, (-1,))
    energy_calibration = xdata.dimensional_calibrations[-1]
    half_width = round(150.0 / energy_calibration.scale)
    peak_channel = int(numpy.argmax(spectrum))
    channels = numpy.arange(max(peak_channel - half_width, 0), min(peak_channel + half_width + 1, len(spectrum)))
    centroid_channel = float(numpy.sum(channels * spectrum[channels]) / numpy.sum(spectrum[channels]))
    # a channel is a bin, so its center is half a channel past its calibrated position.
    return energy_calibration.convert_to_calibrated_value(centroid_channel + 0.5)


# the standard deviation in eV of the counts within 250 eV of the energy, above the background level of the channels
# 250 to 300 eV either side, which is the width of a line there.
def line_width(spectrum: _NDArray, energy_eV: float) -> float:
    channel = int(energy_eV // ENERGY_PER_CHANNEL_EV)
    channels = numpy.arange(channel - 25, channel + 26)
    background = (numpy.mean(spectrum[channel - 30:channel - 25]) + numpy.mean(spectrum[channel + 26:channel + 31])) / 2
    counts = numpy.maximum(spectrum[channels] - background, 0.0)
    mean_channel = numpy.sum(channels * counts) / numpy.sum(counts)
    return float(numpy.sqrt(numpy.sum((channels - mean_channel) ** 2 * counts) / numpy.sum(counts))) * ENERGY_PER_CHANNEL_EV


class TestEDXSDetector(unittest.TestCase):

    def test_edxs_spectrum_shows_the_lines_of_the_feature_under_the_probe(self) -> None:
        with contextlib.closing(make_instrument()) as instrument:
            with contextlib.closing(EDXSDetectorSimulator.EDXSDetectorSimulator(instrument, SENSOR_SIZE)) as simulator:
                place_probe_on_element(instrument, "Si")
                spectrum = acquire_spectrum(simulator, instrument)
                self.assertGreater(window_counts(spectrum, 1740.0), 20 * background_counts(spectrum, 1740.0) + 100)
                place_probe_on_element(instrument, "Ni")
                spectrum = acquire_spectrum(simulator, instrument)
                self.assertGreater(window_counts(spectrum, 7478.0), 20 * background_counts(spectrum, 7478.0) + 100)
                self.assertLess(window_counts(spectrum, 1740.0), 2 * background_counts(spectrum, 1740.0) + 20)

    def test_edxs_simulates_the_x_rays_of_every_element_the_samples_declare(self) -> None:
        scan_data_generator = InstrumentDevice.ScanDataGenerator()
        for sample_index in range(len(scan_data_generator.sample_titles)):
            scan_data_generator.sample_index = sample_index
            for feature in scan_data_generator.sample.features:
                for element in feature.elements:
                    self.assertIn(element, EDXSDetectorSimulator.ELEMENT_XRAYS)

    def test_edxs_spectrum_in_vacuum_has_only_the_faint_system_peaks(self) -> None:
        with contextlib.closing(make_instrument()) as instrument:
            with contextlib.closing(EDXSDetectorSimulator.EDXSDetectorSimulator(instrument, SENSOR_SIZE)) as simulator:
                place_probe_on_element(instrument, "Si")
                sample_spectrum = acquire_spectrum(simulator, instrument)
                get_value_manager(instrument).stage_position_m = Geometry.FloatPoint(y=10E-6, x=10E-6)
                vacuum_spectrum = acquire_spectrum(simulator, instrument, 1.0)
                self.assertLess(numpy.sum(vacuum_spectrum), 0.5 * numpy.sum(sample_spectrum))
                self.assertGreater(window_counts(vacuum_spectrum, 8048.0), 10 * background_counts(vacuum_spectrum, 8048.0) + 20)
                self.assertLess(window_counts(vacuum_spectrum, 1740.0), 20)

    def test_edxs_spectrum_is_empty_when_the_beam_is_blanked(self) -> None:
        with contextlib.closing(make_instrument()) as instrument:
            with contextlib.closing(EDXSDetectorSimulator.EDXSDetectorSimulator(instrument, SENSOR_SIZE)) as simulator:
                place_probe_on_element(instrument, "Si")
                get_value_manager(instrument).is_blanked = True
                self.assertEqual(0, numpy.sum(acquire_spectrum(simulator, instrument)))

    def test_edxs_counts_follow_the_beam_current(self) -> None:
        with contextlib.closing(make_instrument()) as instrument:
            with contextlib.closing(EDXSDetectorSimulator.EDXSDetectorSimulator(instrument, SENSOR_SIZE)) as simulator:
                place_probe_on_element(instrument, "Si")
                counts = numpy.sum(acquire_spectrum(simulator, instrument, 0.5))
                instrument.SetVal("BeamCurrent", instrument.GetVal("BeamCurrent") / 4)
                self.assertAlmostEqual(0.25, numpy.sum(acquire_spectrum(simulator, instrument, 0.5)) / counts, delta=0.05)

    def test_edxs_lines_are_not_excited_by_a_beam_below_their_critical_energy(self) -> None:
        with contextlib.closing(make_instrument()) as instrument:
            with contextlib.closing(EDXSDetectorSimulator.EDXSDetectorSimulator(instrument, SENSOR_SIZE)) as simulator:
                place_probe_on_element(instrument, "Si")
                # the beam energy is above the energy of Si K-alpha, 1740 eV, but below its critical energy, 1839 eV.
                instrument.SetVal("EHT", 1800)
                self.assertLess(window_counts(acquire_spectrum(simulator, instrument), 1740.0), 5)

    def test_edxs_continuum_falls_with_the_beam_energy(self) -> None:
        with contextlib.closing(make_instrument()) as instrument:
            with contextlib.closing(EDXSDetectorSimulator.EDXSDetectorSimulator(instrument, SENSOR_SIZE)) as simulator:
                place_probe_on_element(instrument, "Si")
                spectrum_100_kV = acquire_spectrum(simulator, instrument, 0.5)
                instrument.SetVal("EHT", 30000)
                spectrum_30_kV = acquire_spectrum(simulator, instrument, 0.5)
                # the continuum between 4 and 6 keV, where there are no lines, falls as (E0 - E) / E.
                self.assertLess(numpy.sum(spectrum_30_kV[400:600]), 0.5 * numpy.sum(spectrum_100_kV[400:600]))

    def test_edxs_line_width_grows_with_the_line_energy(self) -> None:
        with contextlib.closing(make_instrument()) as instrument:
            instrument.SetVal("BeamCurrent", 1E-9)
            with contextlib.closing(EDXSDetectorSimulator.EDXSDetectorSimulator(instrument, SENSOR_SIZE)) as simulator:
                place_probe_on_element(instrument, "Si")
                si_k_alpha_width = line_width(acquire_spectrum(simulator, instrument, 0.5), 1740.0)
                place_probe_on_element(instrument, "Ni")
                ni_k_alpha_width = line_width(acquire_spectrum(simulator, instrument, 0.5), 7478.0)
                self.assertGreater(ni_k_alpha_width, 1.4 * si_k_alpha_width)

    def test_edxs_high_count_rate_produces_a_sum_peak(self) -> None:
        with contextlib.closing(make_instrument()) as instrument:
            place_probe_on_element(instrument, "Si")
            with contextlib.closing(EDXSDetectorSimulator.EDXSDetectorSimulator(instrument, SENSOR_SIZE)) as simulator:
                # at a low beam current, a longer exposure records about as many Si K-alpha counts, but no sum peak.
                instrument.SetVal("BeamCurrent", 0.2E-9)
                spectrum = acquire_spectrum(simulator, instrument, 1.0)
                self.assertLess(window_counts(spectrum, 2 * 1740.0), 3 * background_counts(spectrum, 2 * 1740.0))
                instrument.SetVal("BeamCurrent", 5E-9)
                spectrum = acquire_spectrum(simulator, instrument)
                self.assertGreater(window_counts(spectrum, 2 * 1740.0), 3 * background_counts(spectrum, 2 * 1740.0))

    def test_edxs_recorded_spectrum_does_not_depend_on_the_exposure(self) -> None:
        # at 20 nA, a 1 s exposure has more photons than the simulator follows individually, and a 20 ms one does not.
        with contextlib.closing(make_instrument()) as instrument:
            instrument.SetVal("BeamCurrent", 20E-9)
            place_probe_on_element(instrument, "Si")
            with contextlib.closing(EDXSDetectorSimulator.EDXSDetectorSimulator(instrument, SENSOR_SIZE)) as simulator:
                short_spectrum = numpy.sum([acquire_spectrum(simulator, instrument, 0.02) for _ in range(50)], axis=0)
                long_spectrum = acquire_spectrum(simulator, instrument, 1.0)
                self.assertAlmostEqual(1.0, numpy.sum(long_spectrum) / numpy.sum(short_spectrum), delta=0.03)
                short_sum_peak_fraction = window_counts(short_spectrum, 2 * 1740.0) / numpy.sum(short_spectrum)
                long_sum_peak_fraction = window_counts(long_spectrum, 2 * 1740.0) / numpy.sum(long_spectrum)
                self.assertAlmostEqual(1.0, long_sum_peak_fraction / short_sum_peak_fraction, delta=0.1)

    def test_edxs_count_rate_falls_when_the_detector_is_paralyzed(self) -> None:
        with contextlib.closing(make_instrument()) as instrument:
            place_probe_on_element(instrument, "Si")
            with contextlib.closing(EDXSDetectorSimulator.EDXSDetectorSimulator(instrument, SENSOR_SIZE)) as simulator:
                instrument.SetVal("BeamCurrent", 5E-9)
                peak_rate_counts = numpy.sum(acquire_spectrum(simulator, instrument))
                instrument.SetVal("BeamCurrent", 50E-9)
                self.assertLess(numpy.sum(acquire_spectrum(simulator, instrument)), 0.1 * peak_rate_counts)


if __name__ == '__main__':
    unittest.main()
