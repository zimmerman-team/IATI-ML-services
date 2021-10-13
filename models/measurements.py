from common import utils
import numpy as np
import enum
import gc

class PlotType(enum.Enum):
    FIELDS = 'fields'
    LOSSES = 'losses'
    LATENT = 'latent'

class MeasurementsCollection(utils.Collection):
    def collect(self, lm, which_tset, of_type):
        # measurements that have been utilized in the aggregations
        utilized = set()
        for curr in self.all_of_type(of_type):
            src = self.src(curr.name)
            if len(src) == 0:
                # look for source in the Lightning Model
                curr.add_from_lm(lm, which_tset)
            else:
                for name, aggregation_fn in src.items():
                    # aggregate the specified source measurements
                    tmp = self[name].aggregate(which_tset, aggregation_fn)
                    utilized.add(self[name])
                    curr.add(tmp, which_tset)

        # empty all the measurements that have been utilized in this moment
        # example: at each epoch empty the measurements that were collected
        #          in every batch
        # possible FIXME: an event-based system?
        for curr in utilized:
            curr.clear()
        gc.collect()

    def all_of_type(self,of_type):
        ret = []
        for curr in self:
            if type(curr) is of_type:
                ret.append(curr)
        return ret

    def src(self, name, of_type=None):
        ret = dict()
        for curr in self:
            if name in curr.dst_names:
                if of_type is None or type(curr) is of_type:
                    ret[curr.name] = curr.dst[name]
        return ret

def mae(data):
    return np.mean(np.abs(data),axis=0)

def var(data):
    return np.var(data,axis=0)

def mean(data):
    return np.mean(data, axis=0)

def random_sampling(data):
    ar = np.arange(data.shape[0])
    rc = np.random.choice(ar, size=100)
    ret = data[rc,:]
    return ret

class Measurement(object):
    def __init__(
            self,
            name,
            aggregation_fn=None,
            dst=None):
        self.name = name
        self.data = dict()
        if aggregation_fn is None:
            self.aggregation_fn = mean
        else:
            self.aggregation_fn = aggregation_fn
        self._dst = dst
        self.clear()

    def clear(self):
        for tset in utils.Tsets:
            self.data[tset.value] = []

    def set(self, which_tset, data):
        self.data[which_tset] = data

    def add(self, chunk, which_tset):
        self.data[which_tset].append(chunk)

    def add_from_lm(self, lm, which_tset):
        chunk = getattr(lm, self.name)
        self.add(chunk, which_tset)

    def vstack(self,which_tset):
        data = self.data[which_tset]
        if len(data) == 0:
            return np.array([[]])
        else:
            return np.vstack(data)

    @property
    def dst_names(self):
        return self.dst.keys() if self.dst else ()

    @property
    def dst(self):
        return self._dst

    def aggregate(self, which_tset, aggregation_fn):
        stacked = self.vstack(which_tset)
        return aggregation_fn(stacked)

    def aggregation(self, which_tset):
        return self.aggregate(which_tset, self.aggregation_fn)

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

class LastEpochMeasurement(Measurement):
    """
    Measurement of the last epoch
    """
    pass
