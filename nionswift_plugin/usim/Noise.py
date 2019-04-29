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
            return input + (rs.poisson(self.poisson_level, size=input.data.shape).astype(input.data.dtype) - self.poisson_level)
        return input
