import os

from common import utils, relspecs
from models import dspn_autoencoder

class DSPNAEModelsStorage(utils.Collection): # FIXME: abstraction?
    """
    We need to store the DSPNAE models somewhere and to recall them
    easily. This class offers a straightforward interface to load
    the models.
    """
    def __init__(self):

        for rel in relspecs.rels:
            model_filename = os.path.join(
                "trained_models",
                f"DSPNAE_{rel.name}"
            )
            model = dspn_autoencoder.DSPNAE.load_from_checkpoint(model_filename)
            self[rel.name] = model
