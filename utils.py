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

MONGODB_CONN="mongodb://mongouser:XGkS1wDyb4922@localhost:27017/learning_sets"
from sklearn.base import BaseEstimator, TransformerMixin

class Collection(dict):
    def __iter__(self):
        return iter(self.values())

    @property
    def names(self):
        return self.keys()

    def __getattr__(self,name):
        assert name in self.names,f"{name} not in collection's names"
        return self[name]

class Tsets(enum.Enum):
    VAL = 'val' # validation set
    TRAIN = 'train' # training set

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
    return pickle.loads(zlib.decompress(buf))

def load_tsets(rel_name, with_set_index=False):
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    coll = db['npas_tsets']
    document = coll.find({'rel': rel_name}).sort('_id', pymongo.DESCENDING).limit(1)[0]
    train_dataset = deserialize(document['train_npa']).astype(np.float32)
    test_dataset = deserialize(document['test_npa']).astype(np.float32)
    if with_set_index is False:
        train_dataset = train_dataset[:,1:]
        test_dataset = test_dataset[:,1:]
    return train_dataset, test_dataset

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