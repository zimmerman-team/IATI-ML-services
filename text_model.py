import functools
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import nltk
import os

MODEL_FILENAME = "doc2vec_model_dbow.bin"

class ModelWrapper():
    def __init__(self):
        directory = os.path.abspath(__file__)
        full_path = os.path.join([directory,MODEL_FILENAME])
        self.model = Doc2Vec.load(full_path)
    def encode(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [curr.lower() for curr in tokens]
        self.model.infer_vector(text)
    def decode(self, code):
        pass #FIXME TODO
    @property
    def n_features(self):
        return self.model.vector_size
    @property
    def empty_vector(self):
        return np.array([0.0]*self.n_features)

@functools.lru_cache(maxsize=None)
def instance():
    wrapper = ModelWrapper()
    return wrapper