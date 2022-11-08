import niftycollection.collection
from common import specs_config, utils
from models import models_storage
import numpy as np
import logging
import torch


class ActivityData(niftycollection.collection.Collection):
    """
    Representation of a IATI activity datapoint
    """
    def __init__(self, activity_data):
        for rel in specs_config.specs:
            self[rel.name] = rel.extract_from_activity_data(activity_data)


class ActivityVectorizer(object):
    """
    Takes activity data and transforms them into a fixed-length vector.
    This is required in order to train the final activity autoencoder,
    as well as other machine learning models on activities.
    """
    def __init__(self, model_modulename_rels):
        self.model_storage = models_storage.ModelsStorage(model_modulename_rels)
        # all the models need to be loaded beforehand otherwise it would be
        # too costly to load them for every activity
        self.model_storage.load_all_models()

    @property
    def rel_latent_dim(self):
        return self.model_storage.get_rel_latent_dim()

    def process(self, activity_sets, activity_fixed_length_fields_npa):

        vectorized_fields = []

        # the fixed-length "normal" fields come first
        vectorized_fields.append(activity_fixed_length_fields_npa)

        for spec_name, encoded_set in activity_sets.items():
            model = self.model_storage[spec_name]
            if model is None:
                available = self.model_storage.names
                msg = f"model storage does not have a model for spec {spec_name}. Available models are for:{available}. Key-value pairs: {self.model_storage.items()}"
                logging.error(msg)
                raise Exception(msg)
            if encoded_set is None:
                # no input  data for this model
                vectorized_field = model.default_z_npa_for_missing_inputs
            else:
                encoded_set_npa = utils.create_set_npa(specs_config.specs[spec_name], encoded_set)

                target_set, target_mask = model._make_target(torch.Tensor(encoded_set_npa))
                vectorized_field_torch = model.encoder(target_set, mask=target_mask)
                vectorized_field = vectorized_field_torch.detach().numpy()
                #logging.warning(f"encoded_set_npa.shape for {spec_name} : {encoded_set_npa.shape} - vectorized_field.shape {vectorized_field.shape}")

            vectorized_fields.append(vectorized_field)
        ret = np.hstack(vectorized_fields)
        logging.debug(f"ActivityVectorizer.process ret.shape {ret.shape}")
        return ret
