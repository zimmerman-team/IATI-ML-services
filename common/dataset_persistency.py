import functools
import pymongo
import gridfs
import logging
import threading

from common import config, splits, utils

# key-indexed by thread it
cached_clients = dict()

class MongoDB(object):
    """
    returns a connection to the mongodb.
    It's cached as only one is needed.
    """

    def __init__(self,caching=True):
        self.caching = caching

    def __enter__(self):
        global cached_clients

        # every thread will have its own client

        tid = threading.get_ident()

        if (not self.caching) or (tid not in cached_clients.keys()):
            client = pymongo.MongoClient(
                config.mongo_uri(),
                connect=False # will connect at the first operation
            )
            cached_clients[tid] = client
        else:
            client = cached_clients[tid]

        db = client['learning_sets']
        return db

    def __exit__(self, *args):
        global cached_clients
        tid = threading.get_ident()
        if tid in cached_clients.keys():
            client = cached_clients.pop(tid)
            client.close()
            del client

@functools.cache
def gridfs_instance():
    """
    returns an instance of the interface to GridFS.
    It's cached as only one is needed.
    :return: the gridfs.GridFS instance
    """
    with MongoDB() as db:
        gf = gridfs.GridFS(db)

        def create_index(coll_name, _spec):
            """
            just a wrapper for mongodb index creation.
            :param coll_name: name of the mongodb collection
            :param _spec: the relspecs_classes.Spec instance
            :return: None
            """
            try:
                db[coll_name].create_index(_spec)
            except pymongo.errors.PyMongoError as e:
                # already existing index
                pass

        for spec in [
            [('filename', 1), ('uploadDate', 1)],
            [('filename', 1)],
            [('uploadDate', 1)]
        ]:
            create_index('fs.files', spec)
        for spec in [
            [('files_id', 1), ('n', 1)],
            [('files_id', 1)],
            [('n', 1)],
        ]:
            create_index('fs.chunks', spec)
        return gf


def load_splits_document(spec):
    """
    Loads a document from the mongo db, containing the
    training and test sets.
    :param spec: the relspecs_classes.Spec instance
    :return: mongodb document of the last splits for this spec
    """
    with MongoDB() as db:
        coll = db['npas_splits']
        document = coll.find({'spec': spec.name}).sort('creation_time', pymongo.DESCENDING).limit(1)[0]
        return document


def load_splits(spec, with_set_index=False, cap=None):
    """
    :param spec: specification of the data to be loaded
    :param with_set_index: set as True to include the set index
    :param cap: limit to a certain amount of datapoints. Useful to quickly debug a new model
    :return: the Splits object containing the dataset splits
    """
    document = load_splits_document(spec)
    logging.info(f"splits creation time: {document['creation_time']}")
    del document['spec']
    ret = splits.Splits(
        spec,
        **document,
        with_set_index=with_set_index,
        cap=cap
    )
    return ret


def remove_npa(filename):
    """
    removes a file containing the numpy array data, from gridfs
    :param filename: name of the file to be deleted
    :return: None
    """
    with MongoDB() as db:
        gf = gridfs_instance()
        files_found = db['fs.files'].find({'filename': filename})
        for curr in files_found:
            gf.delete(curr['_id'])
            db['fs.files'].remove({'_id': curr['_id']})
            db['fs.chunks'].remove({'files_id': curr['_id']})
        assert not gf.exists({'filename': filename}), f"remove_npa was unable to remove entirely {filename}"


def save_npa(filename, npa):
    """
    save a file containing the numpy array data, to gridfs
    :param filename: name of the file where the npa will be saved`
    :param npa: npa containing a dataset
    :return: the id of the newly-created file
    """
    gf = gridfs_instance()
    remove_npa(filename)
    serialized = utils.serialize(npa)
    chunk_size = 8 * (1024 ** 2)  # 8M
    f = gf.new_file(
        filename=filename,
        chunk_size=chunk_size
    )
    f.write(serialized)
    f.close()
    return f._id


def get_npa_file_id(filename):
    """
    returns the id of a file containing the numpy array dataset,
    given a filename.
    :param filename: filename associated to the npa
    :return: the id of the searched file
    """
    with MongoDB() as db:
        found = db['fs.files'].find_one({'filename': filename})
        assert found is not None, f"get_npa_file_id: could not find an entry with filename={filename}"
        return found['_id']


def load_npa(file_id=None, filename=None):
    """
    Loads a numpy array containing a dataset, given either a filename or a file_id
    :param file_id: optional, id of the file containing the npa
    :param filename: optional, name of the file containing the npa
    :return:
    """
    # one and only one of the two arguments needs to be specified
    assert sum([file_id is None, filename is None]) == 1, "load_npa: specify only one of file_id,filename args"
    if filename is not None:
        file_id = get_npa_file_id(filename)
    gf = gridfs_instance()
    serialized = gf.get(file_id).read()
    npa = utils.deserialize(serialized)
    return npa
