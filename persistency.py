import functools
import numpy as np

import tsets
import utils

import pymongo
MONGODB_CONN="mongodb://mongouser:XGkS1wDyb4922@localhost:27017/learning_sets"

@functools.cache
def mongo_db():
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    return db

def load_tsets_document(rel):
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    coll = db['npas_tsets']
    document = coll.find({'rel': rel.name}).sort('_id', pymongo.DESCENDING).limit(1)[0]
    return document

def load_tsets(rel, with_set_index=False):
    document = load_tsets_document(rel)
    del document['rel']
    ret = tsets.Tsets(rel, **document, with_set_index=with_set_index)
    return ret