# standard libraries
import numpy
import numpy.typing
import typing

from nion.data import DataAndMetadata

_NDArray = numpy.typing.NDArray[typing.Any]


class PoissonNoise:

    def __init__(self) -> None:
        self.enabled = True
        self.poisson_level: typing.Optional[float] = None

    def apply(self, input: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
        if self.enabled and self.poisson_level:
            rs = numpy.random.RandomState()  # use this to avoid blocking other calls to poisson
            input_data = input.data
            assert input_data is not None
            poisson_data = typing.cast(_NDArray, rs.poisson(self.poisson_level, size=input_data.shape).astype(input_data.dtype))
            return input + (poisson_data - self.poisson_level)
        return input
