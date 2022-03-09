from common import relspecs, utils
from models import models_storage
import numpy as np
import logging
import torch


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
            if encoded_set is None:
                # no input  data for this model
                # FIXME: more object-oriented way to create this?
                vectorized_field = np.zeros((1,model.kwargs['latent_dim']))
            else:
                encoded_set_npa = utils.create_set_npa(relspecs.specs[spec_name],encoded_set)
                logging.debug(f"encoded_set_npa.shape {encoded_set_npa.shape}")

                model = self.model_storage[spec_name]
                target_set, target_mask = model._make_target(torch.Tensor(encoded_set_npa))
                vectorized_field_torch = model.encoder(target_set, mask=target_mask)
                vectorized_field = vectorized_field_torch.detach().numpy()

            vectorized_fields.append(vectorized_field)
        vectorized_fields.append(activity_fixed_length_fields_npa)
        ret = np.hstack(vectorized_fields)
        logging.debug(f"ActivityVectorizer.process ret.shape {ret.shape}")
        return ret
