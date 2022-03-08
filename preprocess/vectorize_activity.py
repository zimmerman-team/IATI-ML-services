from common import relspecs, utils
from models import models_storage
import numpy as np
import logging


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
        # all the models need to be loaded beforehand otherwise it would be
        # too costly to load them for every activity
        self.model_storage.load_all_models()

    def process(self, activity_sets, activity_fixed_length_fields_npa):

        vectorized_fields = []
        for spec_name, encoded_set in activity_sets.items():
            encoded_set_npa = np.array(encoded_set)
            logging.info(f"encoded_set_npa.shape {encoded_set_npa.shape}")

            # query the model of a set
            print(f"encoded_set_npa {encoded_set_npa}")
            vectorized_field = self.model_storage[spec_name].encoder(encoded_set_npa)

            vectorized_fields.append(vectorized_field)
        vectorized_fields.append(activity_fixed_length_fields_npa)
        ret = np.hstack(vectorized_fields)
        return ret
