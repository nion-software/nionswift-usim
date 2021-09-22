# standard libraries
import numpy

from nion.data import DataAndMetadata


class PoissonNoise:

    def __init__(self):
        self.enabled = True
        self.poisson_level = None

    def apply(self, input: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
        if self.enabled and self.poisson_level:
            rs = numpy.random.RandomState()  # use this to avoid blocking other calls to poisson
            input_data = input.data
            assert input_data is not None
            return input + (rs.poisson(self.poisson_level, size=input_data.shape).astype(input_data.dtype) - self.poisson_level)
        return input
