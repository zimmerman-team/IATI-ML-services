import functools
import pymongo
import gridfs

from common import config
from common import tsets
from common import utils


@functools.cache
def mongo_db():
    client = pymongo.MongoClient(config.mongo_uri())
    db = client['learning_sets']
    return db


@functools.cache
def gridfs_instance():
    db = mongo_db()
    gf = gridfs.GridFS(db)
    return gf


def load_tsets_document(rel):
    client = pymongo.MongoClient(config.mongo_uri())
    db = client['learning_sets']
    coll = db['npas_tsets']
    document = coll.find({'rel': rel.name}).sort('_id', pymongo.DESCENDING).limit(1)[0]
    return document


def load_tsets(rel, with_set_index=False):
    document = load_tsets_document(rel)
    del document['rel']
    ret = tsets.Tsets(rel, **document, with_set_index=with_set_index)
    return ret


def remove_npa(filename):
    db = mongo_db()
    gf = gridfs_instance()
    files_found = db['fs.files'].find({'filename': filename})
    for curr in files_found:
        gf.delete(curr['_id'])
        db['fs.files'].remove({'_id': curr['_id']})
        db['fs.chunks'].remove({'files_id': curr['_id']})
    assert not gf.exists({'filename': filename}), f"remove_npa was unable to remove entirely {filename}"


def save_npa(filename, npa):
    gf = gridfs_instance()
    remove_npa(filename)
    serialized = utils.serialize(npa)
    f = gf.new_file(
        filename=filename,
        chunk_size=8*(1024**2) # 8M
    )
    f.write(serialized)
    f.close()
    return f._id


def get_npa_file_id(filename):
    db = mongo_db()
    found = db['fs.files'].find_one({'filename': filename})
    assert found is not None, f"get_npa_file_id: could not find an entry with filename={filename}"
    return found['_id']


def load_npa(file_id=None, filename=None):
    # one and only one of the two arguments needs to be specified
    assert sum([file_id is None, filename is None]) == 1, "load_npa: specify only one of file_id,filename args"
    if filename is not None:
        file_id = get_npa_file_id(filename)
    gf = gridfs_instance()
    serialized = gf.get(file_id).read()
    npa = utils.deserialize(serialized)
    return npa
