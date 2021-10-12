from common import utils
import numpy as np
import enum

class PlotType(enum.Enum):
    FIELDS = 'fields'
    LOSSES = 'losses'
    LATENT = 'latent'

class MeasurementsCollection(utils.Collection):
    def collect(self, lm, which_tset):
        for curr in self:
            curr.add(lm, which_tset)

def mae(data):
    return np.mean(np.abs(data),axis=0)

def var(data):
    return np.var(data,axis=0)

def mean(data):
    return np.mean(data, axis=0)

class Measurement(object):
    def __init__(self,name, aggregation_fn=None):
        self.name = name
        self.data = dict()
        if aggregation_fn is None:
            self.aggregation_fn = mean
        else:
            self.aggregation_fn = aggregation_fn
        self.clear()

    def clear(self):
        for tset in utils.Tsets:
            self.data[tset] = []

    def set(self, which_tset, data):
        self.data[which_tset] = data

    def add(self, lm, which_tset):
        chunk = getattr(lm, self.name)
        self.data[which_tset].append(which_tset, chunk)

    def vstack(self,which_tset):
        return np.vstack(self.data[which_tset])

    def aggregation(self, which_tset):
        stacked = self.vstack(which_tset)
        return self.aggregation_fn(stacked)

class DatapointMeasurement(Measurement):
    """
    Measurement indexed by datapoint
    """
    pass

class BatchMeasurement(Measurement):
    """
    Measurement indexed by batch idx
    """
    pass

class EpochMeasurement(Measurement):
    """
    Measurement indexed by epoch nr.
    """
    pass
