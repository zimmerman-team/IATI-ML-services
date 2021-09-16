from bson.binary import Binary
import pickle
import zlib

def serialize(npa):
    return Binary(zlib.compress(pickle.dumps(npa, protocol=2)))

def deserialize(buf):
    return pickle.loads(zlib.decompress(buf))

