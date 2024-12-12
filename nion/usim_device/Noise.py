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

    def apply(self, input: DataAndMetadata.DataAndMetadata, lambda_thresh: float = 1.0) -> DataAndMetadata.DataAndMetadata:
        if self.enabled and self.poisson_level:
            rs = numpy.random.RandomState()  # use this to avoid blocking other calls to poisson
            input_data = input.data
            input_data_shape = input.data_shape
            assert input_data is not None
            if self.poisson_level > lambda_thresh:
                # Since it is 'high' lambda, we can approximate it to a normal distribution
                poisson_data = typing.cast(_NDArray, rs.normal(loc=self.poisson_level, scale=numpy.sqrt(self.poisson_level), size=input_data_shape).astype(input_data.dtype))
            else:
                poisson_data = typing.cast(_NDArray, rs.poisson(self.poisson_level, size=input_data_shape).astype(input_data.dtype))
            return input + (poisson_data - self.poisson_level)
        return input

