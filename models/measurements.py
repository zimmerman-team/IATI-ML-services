from common import utils
import numpy as np
import enum
import gc
import logging
import mlflow

def log(*args):
    strs = map(str,args)
    msg = "measurements:"+" ".join(strs)
    logging.debug(msg)

class PlotType(enum.Enum):
    FIELDS = 'fields'
    LOSSES = 'losses'
    LATENT = 'latent'

class MeasurementsCollection(utils.Collection):
    def collect(self, lm, which_tset, of_type):
        # measurements that have been utilized in the aggregations
        utilized = set()
        if type(of_type) not in (tuple,list):
            of_type = (of_type,)
        for curr_of_type in of_type:
            log(f"collect: now considering type {curr_of_type} among {of_type}")
            for curr in self.all_of_type(curr_of_type):
                src = self.src(curr.name)
                log(f"processing {curr} - its src is {src}")
                if len(src) == 0:
                    log(f"look for data origin for {curr.name} in the Lightning Model")
                    curr.add_from_lm(lm, which_tset)
                else:
                    for name, aggregation_fn in src.items():
                        # aggregate the specified source measurements
                        log(f"aggregating {name}->{curr.name} with {aggregation_fn}..")
                        tmp = self[name].aggregate(which_tset, aggregation_fn)
                        curr.add(tmp, which_tset)
                        utilized.add(self[name])

        # empty all the measurements that have been utilized in this moment
        # example: at each epoch empty the measurements that were collected
        #          in every batch
        # possible FIXME: an event-based system?
        log(f"now clearing {utilized}")
        for curr in utilized:
            curr.clear(which_tset)
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

    def print_debug_info(self):
        print("Measurements counts:")
        for curr in self:
            print(f"\t{curr.name}: ",end='')
            curr.print_counts()
            if curr.dst:
                for currdst in curr.dst:
                    print(f"\t\t-> {currdst}: ",end='')
                    self[currdst].print_counts()

    @property
    def plottable(self):
        ret = []
        for curr in self:
            if curr.plot_type is not None:
                ret.append(curr)
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
            dst=None,
            plot_type=None,
            mlflow_log=False
    ):
        self.name = name
        self.data = dict()
        self.plot_type = plot_type
        self.mlflow_log = mlflow_log
        if aggregation_fn is None:
            self.aggregation_fn = mean
        else:
            self.aggregation_fn = aggregation_fn
        self._dst = dst
        for which_tset in utils.Tsets:
            self.clear(which_tset.value)

    def clear(self,which_tset):
        log(f"clearing {self.name} {which_tset}..")
        self.data[which_tset] = []

    def set(self, which_tset, data):
        self.data[which_tset] = data

    def add(self, chunk, which_tset):
        if type(chunk) is np.ndarray:
            chunk_size = chunk.shape
        elif type(chunk) is list:
            chunk_size = len(chunk)
        else:
            chunk_size = f"??{type(chunk)}"
        if self.mlflow_log:
            mean = np.mean(np.array(chunk))
            mlflow.log_metric(self.name,mean)
        log(f"{self} {which_tset} added with chunk {chunk_size}")
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

    def __str__(self):
        # FIXME: maybe add the dimension sizes over here?
        return f"<Measurement {self.name}>"

    def aggregate(self, which_tset, aggregation_fn):
        stacked = self.vstack(which_tset)
        log(f"{self} aggregating with {aggregation_fn} on {which_tset} (shape {stacked.shape})..")
        ret = aggregation_fn(stacked)
        log(f"done aggregating; the result has shape {ret.shape}.")
        return ret

    def aggregation(self, which_tset):
        return self.aggregate(which_tset, self.aggregation_fn)

    def count(self, which_tset):
        return len(self.data[which_tset])

    def _recursive_counts_str(self, stuff):
        if not utils.is_vectoriform(stuff):
            return "_"
        elif utils.is_empty(stuff):
            return "[]"
        else:
            c = len(stuff)
            rec = self._recursive_counts_str(stuff[0])
            tmp = f"{c}/{rec}"
            return tmp

    def print_counts(self):
        for which_tset in utils.Tsets:
            d = self.data[which_tset.value]
            tmp = self._recursive_counts_str(d)
            print(which_tset.value + ":" + tmp+" ", end='')
        print(".")

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
