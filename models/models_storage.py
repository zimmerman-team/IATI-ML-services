import logging
import os
import sys
import pickle
import pytorch_lightning as pl
import glob
import re
project_root_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path.insert(0, project_root_dir)

from common import utils, relspecs, config
from models import dspn_autoencoder



class DSPNAEModelsStorage(utils.Collection): # FIXME: classname-parameterized?
    """
    We need to store the DSPNAE models somewhere and to recall them
    easily. This class offers a straightforward interface to load
    the models.
    """
    def __init__(self):
        os.chdir(project_root_dir)

    def load_all_models(self):
        os.chdir(project_root_dir)
        for rel in relspecs.rels:
            model = self.load(rel)
            self[rel.name] = model

    def create_write_callback(self, model):
        """
        creates a model checkpoint dumping callback for the training
        of the given model.
        Also allows for model versioning, which is being taken care
        by the pytorch_lightning library.
        :param model: model to be saved
        :return: the callback object to be used as callback=[callbacks]
            parameter in the pytorch_lightning Trainer
        """
        model_filename = model.name
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="train_loss",
            dirpath=config.trained_models_dirpath,
            filename=model_filename,
            save_top_k=1,
            save_last=False,
            mode="min",
        )
        return checkpoint_callback

    def generate_kwargs_filename(self, model):

        # using tha version from the last model filename
        # as it is saved before the kwargs dump
        version = self.last_version(model.rel)

        ret = os.path.join(
            config.trained_models_dirpath,
            f"{model.name}-v{version}.kwargs.pickle"
        )
        return ret

    def dump_kwargs(self,model):
        kwargs_filename = self.generate_kwargs_filename(model)
        with open(kwargs_filename, 'wb') as f:
            pickle.dump(model.kwargs, f)

    def filenames(self,rel,extension):
        filenames_glob = os.path.join(
            "trained_models",
            f"DSPNAE_{rel.name}*.{extension}"
        )
        print("filenames_glob",filenames_glob)
        ret = {}
        filenames = glob.glob(filenames_glob)
        for curr in filenames:
            m = re.match(f'.*-v(\d+).{extension}', curr)
            if m:
                # has version number
                print(curr, m.groups())
                version = int(m.groups()[0])
            else:
                # no version number in filename: this was the first
                print(f'filename {curr} not matching versioned pattern')
                version = 0
            ret[version] = curr
        print("ret",ret)
        return ret

    def kwargs_filenames(self,rel):
        return self.filenames(rel,'kwargs.pickle')

    def models_filenames(self,rel):
        return self.filenames(rel,'ckpt')

    def last_version(self, rel):
        filenames = self.models_filenames(rel)
        versions = filenames.keys()
        if len(versions) == 0:
            # there was no model stored
            return None
        last_version = max(versions)
        return last_version

    def most_recent_kwargs_filename(self, rel):
        last_version = self.last_version(rel)
        if last_version is None:
            # there was no model stored
            return None
        filenames = self.kwargs_filenames(rel)
        if last_version not in filenames.keys():
            raise Exception(f"cannot find kwargs file for version {last_version}")
        return filenames[last_version]

    def most_recent_model_filename(self, rel):
        filenames = self.models_filenames(rel)
        last_version = self.last_version(rel)
        if last_version is None:
            # there was no model stored
            return None
        return filenames[last_version]

    def rel_has_stored_model(self,rel):
        kwargs_filename = self.most_recent_kwargs_filename(rel)
        model_filename = self.most_recent_model_filename(rel)
        if None in (kwargs_filename, model_filename):
            return False
        if os.path.exists(kwargs_filename) and os.path.exists(model_filename):
            return True
        return False

    def load(self, rel):
        """
        :param rel: the relation this model has been trained on
        :return:
        """
        if not self.rel_has_stored_model(rel):
            logging.warning(f"no model for rel {rel.name}")
            return

        kwargs_filename = self.most_recent_kwargs_filename(rel)
        with open(kwargs_filename, 'rb') as f:
            kwargs = pickle.load(f)

        model_filename = self.most_recent_model_filename(rel)
        if model_filename is None or not os.path.exists(model_filename):
            logging.warning(f"could not find model saved in {model_filename}")
            return

        # FIXME: duplicated from GenericModel
        logging.info(f"loading {model_filename}..")

        # FIXME: kwargs provided twice?
        model = dspn_autoencoder.DSPNAE(**kwargs)
        model.load_from_checkpoint(model_filename, **kwargs)
        return model

def test():
    ms = DSPNAEModelsStorage()
    ms.load_all_models()
    print("ms",ms)
    print("done.")

if __name__ == "__main__":
    test()