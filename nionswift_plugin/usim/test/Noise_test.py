import unittest

import numpy
import numpy.typing

from nion.data import DataAndMetadata
from nion.usim_device import Noise


class TestPoissonNoise(unittest.TestCase):

    def __apply_noise(self, level: float) -> numpy.typing.NDArray[numpy.float32]:
        # apply noise to a constant frame, so the noise alone is what remains after subtracting the constant.
        input_data = numpy.full((300, 700), 10.0, dtype=numpy.float32)
        noise = Noise.PoissonNoise()
        noise.poisson_level = level
        output_data = noise.apply(DataAndMetadata.new_data_and_metadata(input_data)).data
        assert output_data is not None
        self.assertEqual(numpy.float32, output_data.dtype)
        return output_data - input_data

    def test_noise_has_zero_mean_and_level_variance_above_threshold(self) -> None:
        # above the threshold the noise uses the normal approximation. the tolerances are many standard errors wide
        # for 210,000 samples, so the test fails only if the distribution is wrong.
        noise = self.__apply_noise(50.0)
        self.assertAlmostEqual(0.0, float(numpy.mean(noise)), delta=0.2)
        self.assertAlmostEqual(50.0, float(numpy.var(noise)), delta=1.5)

    def test_noise_has_zero_mean_and_level_variance_at_or_below_threshold(self) -> None:
        # at or below the threshold the noise samples the Poisson distribution, shifted to zero mean.
        noise = self.__apply_noise(0.5)
        self.assertAlmostEqual(0.0, float(numpy.mean(noise)), delta=0.02)
        self.assertAlmostEqual(0.5, float(numpy.var(noise)), delta=0.02)

    def test_disabled_noise_leaves_data_unchanged(self) -> None:
        input_data = numpy.full((3, 7), 10.0, dtype=numpy.float32)
        noise = Noise.PoissonNoise()
        noise.poisson_level = 50.0
        noise.enabled = False
        output_data = noise.apply(DataAndMetadata.new_data_and_metadata(input_data)).data
        self.assertTrue(numpy.array_equal(input_data, output_data))


if __name__ == '__main__':
    unittest.main()
