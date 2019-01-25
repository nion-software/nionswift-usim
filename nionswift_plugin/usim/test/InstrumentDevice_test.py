import contextlib
import unittest

from nionswift_plugin.usim import InstrumentDevice


class TestCamera(unittest.TestCase):

    def test_defocus_is_observable(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")

        defocus_ref = [instrument.defocus_m]
        property_changed_count_ref = [0]

        def property_changed(key: str) -> None:
            if key == "defocus_m":
                defocus_ref[0] = instrument.defocus_m
            property_changed_count_ref[0] += 1

        with contextlib.closing(instrument.property_changed_event.listen(property_changed)):
            # setting defocus should set c10 which should trigger its dependent defocus_m
            instrument.defocus_m += 20 / 1E9
            self.assertAlmostEqual(520 / 1E9, defocus_ref[0])
            # setting c10 should trigger its dependent defocus_m
            instrument.SetVal("C10", 600 / 1E9)
            self.assertAlmostEqual(600 / 1E9, defocus_ref[0])
            # a total of 4 changes should happen; no more, no less
            self.assertEqual(4, property_changed_count_ref[0])
