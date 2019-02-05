import contextlib
import unittest

from nion.utils import Geometry

from nionswift_plugin.usim import InstrumentDevice


class TestCamera(unittest.TestCase):

    def test_defocus_is_observable(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")

        defocus_m = instrument.defocus_m
        property_changed_count = 0

        def property_changed(key: str) -> None:
            nonlocal defocus_m, property_changed_count
            if key == "defocus_m":
                defocus_m = instrument.defocus_m
                property_changed_count += 1
            if key == "C10":
                property_changed_count += 1

        with contextlib.closing(instrument.property_changed_event.listen(property_changed)):
            # setting defocus should set c10 which should trigger its dependent defocus_m
            instrument.defocus_m += 20 / 1E9
            self.assertAlmostEqual(520 / 1E9, defocus_m)
            # setting c10 should trigger its dependent defocus_m
            instrument.SetVal("C10", 600 / 1E9)
            self.assertAlmostEqual(600 / 1E9, defocus_m)
            # a total of 4 changes should happen; no more, no less
            self.assertEqual(4, property_changed_count)

    def test_ronchigram_handles_dependencies_properly(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        camera = InstrumentDevice.RonchigramCameraSimulator(instrument, Geometry.IntSize(128, 128), 10, 0.030)
        camera._needs_recalculation = False
        instrument.defocus_m += 10 / 1E9
        self.assertTrue(camera._needs_recalculation)
        camera._needs_recalculation = False
        instrument.C30 += 1E9
        self.assertTrue(camera._needs_recalculation)
        camera._needs_recalculation = False
        instrument.ZLPoffset += 1
        self.assertFalse(camera._needs_recalculation)
        camera._needs_recalculation = False
        instrument.beam_current += 1
        self.assertTrue(camera._needs_recalculation)
