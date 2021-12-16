import functools
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import nltk
import os

MODEL_FILENAME = "doc2vec_model_dbow.bin"


class ModelWrapper(object):
    """
    Wrapper for the text model.
    Allows to encode a paragraph into a fixed-length vector.
    """

    def __init__(self):
        """
        Constructor: loads the model with his trained parameters
        located in the same directory as this python module.
        """
        directory = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(directory, MODEL_FILENAME)
        self.model = Doc2Vec.load(full_path)

    def encode(self, text):
        """
        Does typical NLP preliminary work (tokenization etc)
        and then calls the text model to return a fixed-size
        embedding vector.
        :param text:
        :return:
        """
        # logging.info("encoding "+str(text))
        tokens = nltk.word_tokenize(text)
        tokens = [curr.lower() for curr in tokens]
        ret = self.model.infer_vector(tokens)
        # logging.info("text encode returning "+str(ret))
        return ret

    def decode(self, code):
        """
        Given an embedding code, creates a paragraph.
        :param code:
        :return:
        """
        pass  # FIXME TODO

    @property
    def n_features(self):
        """
        size of the fixed-length embedding vector
        :return:
        """
        return self.model.vector_size

    @property
    def empty_vector(self):
        """
        returns a new empty (with zeroes) embedding vector
        :return:
        """
        return np.array([0.0]*self.n_features)


@functools.lru_cache(maxsize=None)
def instance():
    """
    returns the Wrapper of the text model.
    It's cached as there is the need for only one instance.
    :return:
    """
    wrapper = ModelWrapper()
    return wrapper
