from bson.binary import Binary
import pickle
import zlib
import pymongo
import numpy as np
import tempfile
import mlflow
import urllib
import os
import glob
import torch
import enum
import yaml
import collections
import datetime
from sklearn.base import BaseEstimator, TransformerMixin

class Collection(dict):

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            l = args[0]
            assert type(l) is list, \
                "first argument of the collection has to be a list of elements"
            for curr in l:
                if curr.name in self.names:
                    raise Exception(f"element {curr.name} already in this Collection")
                self[curr.name] = curr
        super().__init__(**kwargs)

    def __iter__(self):
        return iter(self.values())

    @property
    def names(self):
        return self.keys()

    def __getattr__(self,name):
        assert name in self.names,f"{name} not in collection's names"
        return self[name]

class Tsets(enum.Enum):
    TRAIN = 'train' # training set
    VAL = 'val' # validation set

class OneHotCrossEntropyLoss():
    def __init__(self, weight = None):
        #print("OneHostCrossEntropyLoss weight:",weight)
        if weight is not None:
            self.cross_entropy_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(weight))
        else:
            self.cross_entropy_fn = torch.nn.CrossEntropyLoss()

    def __call__(self, x_hat, batch):
        labels = batch.argmax(1)
        ret = self.cross_entropy_fn(x_hat, labels)
        return ret

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, input_array, y=None):
        return self
    def transform(self, input_array, y=None):
        return input_array * 1

    def inverse_transform(self, input_array, y=None):
        return input_array * 1

def serialize(npa):
    return Binary(zlib.compress(pickle.dumps(npa, protocol=2)))

def deserialize(buf):
    return pickle.loads(zlib.decompress(buf)).astype(np.float32)

def set_np_printoptions():
    np.set_printoptions(
        suppress=True, # no unnecessary scientific notation
        linewidth=319, # prevents too many newlines
        formatter=dict(float=lambda x: "%.3g" % x), # remove extra spaces
        threshold=np.inf
    )

def inspect_tset(tset):
    print(tset[0:2])

def dump_npa(npa,prefix="",suffix=""):
    buf = serialize(npa)
    filename = tempfile.mktemp(prefix=prefix, suffix=suffix)
    with open(filename, "wb+") as f:
        f.write(buf)
        f.flush()
    return filename

def log_npa_artifact(npa,prefix="some_npa",suffix=".bin"):
    npa_filename = dump_npa(
        npa,
        prefix=prefix,
        suffix=suffix
    )
    mlflow.log_artifact(npa_filename)

def load_npa(filename):
    with open(filename, 'rb') as f:
        buf = f.read()
        npa = deserialize(buf)
        return npa

def glob_artifacts(run_id,prefix="",suffix=""):
    run = mlflow.get_run(run_id)
    auri = run.info.artifact_uri
    apath = urllib.parse.urlparse(auri).path
    search = os.path.join(apath, prefix+"*"+suffix)
    found = glob.glob(search)
    found = sorted(found)
    return found

def load_npa_artifact(experiment_name, run_id,prefix="",suffix=""):
    found = glob_artifacts(run_id, prefix=prefix, suffix=suffix)
    npa = load_npa(found[0])
    return npa

def is_seq(stuff):
    return type(stuff) in (tuple, list)

def fn_across(stuff, fn):
    if is_seq(stuff):
        return fn(list(map(lambda curr: fn_across(curr, fn),stuff)))
    else:
        return fn(stuff)

def min_across(stuff):
    return fn_across(stuff, np.min)

def max_across(stuff):
    return fn_across(stuff, np.max)

def str_shapes(stuff):
    if is_seq(stuff):
        return "["+" ".join([str_shapes(curr) for curr in stuff])+"]"
    else:
        return str(stuff.shape)

def load_model_config(config_name):
    if os.path.exists(config_name):
        # a filename is given
        filename = config_name
    else:
        # config name, then resolved to a filename is given
        directory = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'model_config/'))
        filename = os.path.join(directory,config_name+".yaml")
    with open(filename, 'r') as f:
        ret = yaml.load(f)
    ret['config_name'] = config_name
    ret['config_filename'] = filename
    return ret

def dict_to_obj(typename, d):
    T = collections.namedtuple(typename,d.keys())
    obj = T(d)
    return obj

def strnow_iso():
    now = str(datetime.datetime.now())
    return now

def strnow_compact():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def is_vectoriform(stuff):
    return type(stuff) in (np.ndarray, list, tuple)

def is_empty(stuff):
    assert is_vectoriform(stuff), f"is_empty: object of type {type(stuff)} is not vectoriform"
    if type(stuff) in (list,tuple):
        return len(stuff) == 0
    if type(stuff) is np.ndarray:
        if len(stuff.shape) == 0:
            return True
        elif stuff.shape[0] == 0:
            return True
        else:
            return False
    raise Exception (f"is_empty: unhandled case for type {type(stuff)}")
