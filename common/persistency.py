import functools
import pymongo
import gridfs
import logging

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

    def create_index(coll_name,spec):
        try:
            db[coll_name].create_index(spec)
        except:
            # already existing index
            pass

    for spec in [
        [('filename', 1), ('uploadDate', 1)],
        [('filename', 1)],
        [('uploadDate', 1)]
        ]:
        create_index('fs.files',spec)
    for spec in [
        [('files_id', 1), ('n', 1)],
        [('files_id', 1)],
        [('n', 1)],
    ]:
        create_index('fs.chunks',spec)
    return gf


def load_tsets_document(rel):
    db = mongo_db()
    coll = db['npas_tsets']
    document = coll.find({'rel': rel.name}).sort('creation_time', pymongo.DESCENDING).limit(1)[0]
    return document


def load_tsets(rel, with_set_index=False, cap=None):
    """
    :param rel: relation's data to be loaded
    :param with_set_index: set as True to include the set index
    :param cap: limit to a certain amount of datapoints. Useful to quickly debug a new model
    :return: the Tsets object containing the dataset splits
    """
    document = load_tsets_document(rel)
    print(f"tsets creation time: {document['creation_time']}")
    del document['rel']
    ret = tsets.Tsets(
        rel,
        **document,
        with_set_index=with_set_index,
        cap=cap
    )
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
