from common import relspecs, utils
from models import models_storage
import numpy as np

class Activity(utils.Collection):
    def __init__(self, activity_data):
        for rel in relspecs:
            self[rel.name] = rel.extract_from_activity_data(activity_data)


class ActivityVectorizer(object):
    def __init__(self):
        self.model_storage = models_storage.DSPNAEModelsStorage()

    def vectorize_activity(self, activity):
        vectorized_fields = []
        for rel in relspecs:
            field_data = activity[rel.name]
            vectorized_field = self.model_storage[rel.name].encoder(field_data)
            vectorized_fields.append(vectorized_field)
        ret = np.hstack(vectorized_fields)
        return ret