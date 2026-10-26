# standard libraries
import numpy
import numpy.typing
import typing

from nion.data import DataAndMetadata

_NDArray = numpy.typing.NDArray[typing.Any]


class PoissonNoise:
    """Applies Poisson noise to simulated detector data.

    poisson_level is the noise level, which is applied uniformly: the variance of the perturbation at each pixel
    is poisson_level, so its standard deviation is the square root of that level. Note that poisson_level is not
    the count at the pixel; the caller supplies a level scaled to the data it passes.

    The level enables the noise when it is non-zero, and it selects between the Poisson distribution and its
    Gaussian approximation.

    scale multiplies the noise's standard deviation. It is 1 by default; a lower scale gives quieter frames, for
    example for tests which acquire at short exposures.
    """

    def __init__(self) -> None:
        self.enabled = True
        self.poisson_level: typing.Optional[float] = None
        self.scale = 1.0
        # each simulated detector holds its own generator. numpy's global generator would make detectors on different
        # threads wait on its lock, and creating a generator for each frame costs time on every frame.
        self.__random_generator = numpy.random.default_rng()

    def apply(self, input: DataAndMetadata.DataAndMetadata, lambda_thresh: float = 1.0) -> DataAndMetadata.DataAndMetadata:
        """Return the data with noise applied, or the data unchanged if the noise is disabled."""
        if self.enabled and self.poisson_level:
            input_data = input.data
            assert input_data is not None
            # the generator supports only these two floating point types, so match the data to the nearer one.
            noise_dtype = numpy.float32 if input_data.dtype == numpy.dtype(numpy.float32) else numpy.float64
            noise: _NDArray
            if self.poisson_level > lambda_thresh:
                # at a high level the Poisson distribution is closely approximated by a normal distribution whose
                # standard deviation is the square root of the level. generating the requested dtype directly
                # avoids a wider intermediate buffer and the cast which would follow it.
                noise = self.__random_generator.standard_normal(input_data.shape, dtype=noise_dtype)
                noise *= noise_dtype(numpy.sqrt(self.poisson_level))
            else:
                # at a low level the normal approximation is poor, since the distribution is skewed and cannot go
                # negative, so sample the Poisson distribution itself and subtract the level it was drawn around.
                noise = (self.__random_generator.poisson(self.poisson_level, size=input_data.shape) - self.poisson_level).astype(noise_dtype)
            if self.scale != 1.0:
                noise *= noise_dtype(self.scale)
            return input + noise
        return input
