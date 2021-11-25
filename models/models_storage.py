import logging
import os
import sys
import pickle

project_root_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path.insert(0, project_root_dir)

from common import utils, relspecs
from models import dspn_autoencoder

class DSPNAEModelsStorage(utils.Collection): # FIXME: abstraction?
    """
    We need to store the DSPNAE models somewhere and to recall them
    easily. This class offers a straightforward interface to load
    the models.
    """
    def __init__(self):
        os.chdir(project_root_dir)
        for rel in relspecs.rels:
            model = self.load(rel)
            self[rel.name] = model

    def load(self, rel):
        model_filename = os.path.join(
            "trained_models",
            f"DSPNAE_{rel.name}.ckpt"
        )

        # FIXME: duplicated from GenericModel
        kwargs_filename = f"DSPNAE_{rel.name}_kwargs.pickle"
        with open(kwargs_filename) as f:
            kwargs = pickle.load(f)
        logging.info(f"loading {model_filename}..")
        if not os.path.exists(model_filename):
            logging.warning(f"could not find model saved in {model_filename}")
            return

        model = dspn_autoencoder.DSPNAE(**kwargs)
        model.load_from_checkpoint(model_filename)
        return model


def test():
    ms = DSPNAEModelsStorage()
    print("ms",ms)
    print("done.")

if __name__ == "__main__":
    test()