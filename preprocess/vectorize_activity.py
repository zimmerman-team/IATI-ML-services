from common import relspecs, utils
from models import models_storage
import numpy as np


class ActivityData(utils.Collection):
    """
    Representation of a IATI activity datapoint
    """
    def __init__(self, activity_data):
        for rel in relspecs.specs:
            self[rel.name] = rel.extract_from_activity_data(activity_data)


class ActivityVectorizer(object):
    """
    Takes activity data and transforms them into a fixed-length vector.
    This is required in order to train the final activity autoencoder,
    as well as other machine learning models on activities.
    """
    def __init__(self, model_modulename):
        self.model_storage = models_storage.ModelsStorage(model_modulename)
        self.model_storage.load_all_models()

    def process(self, activity):

        vectorized_fields = []
        for spec_name, encoded_set in activity.items():
            encoded_set_npa = np.array(encoded_set)
            vectorized_field = self.model_storage[spec_name].encoder(encoded_set_npa)
            vectorized_fields.append(vectorized_field)
        ret = np.hstack(vectorized_fields)
        return ret
