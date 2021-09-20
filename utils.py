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

MONGODB_CONN="mongodb://mongouser:XGkS1wDyb4922@localhost:27017/learning_sets"

def serialize(npa):
    return Binary(zlib.compress(pickle.dumps(npa, protocol=2)))

def deserialize(buf):
    return pickle.loads(zlib.decompress(buf))

def load_tsets(rel_name):
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    coll = db['npas_tsets']
    document = coll.find({'rel': rel_name}).sort('_id', pymongo.DESCENDING).limit(1)[0]
    train_dataset = deserialize(document['train_npa']).astype(np.float32)
    test_dataset = deserialize(document['test_npa']).astype(np.float32)
    return train_dataset, test_dataset

def set_np_printoptions():
    np.set_printoptions(
        suppress=True, # no unnecessary scientific notation
        linewidth=319, # prevents too many newlines
        formatter=dict(float=lambda x: "%.0g" % x) # remove extra spaces
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

def load_npa(filename):
    with open(filename, 'rb') as f:
        buf = f.read()
        npa = deserialize(buf)
        return npa

def load_npa_artifact(experiment_name, run_id,prefix="",suffix=""):
    run = mlflow.get_run(run_id)
    auri = run.info.artifact_uri
    apath = urllib.parse.urlparse(auri).path
    search = os.path.join(apath, prefix+"*"+suffix)
    filename = glob.glob(search)[0]
    npa = load_npa(filename)
    return npa