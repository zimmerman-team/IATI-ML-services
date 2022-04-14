import logging
import os
import shutil
import tempfile
import sys
import pickle
import pytorch_lightning as pl
import glob
import re
import copy
import logging

project_root_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path.insert(0, project_root_dir)

from common import utils, specs_config, config
import models

from collections import namedtuple

# Simple class representing a record in our database.
MemoRecord = namedtuple("MemoRecord", "key, task")

class KwargsPickler(pickle.Pickler):
    # we need a custom pickler class to remove unpickable objects
    def persistent_id(self, obj):
        if callable(obj):
            # functions such as Field.__init__.<locals>._loss_function cannot be pickled
            return ("unpickable function",None)
        if issubclass(type(obj), specs_config.Spec):
            # the Spec types may have machine-learning (torch) functions, that cannot be pickled.
            # so their name is being stored instead
            return ("Spec",obj.name)
        return None

class KwargsUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        type_, key = pid
        if type_ == "unpicklable function":
            return None
        elif type_ == "Spec":
            # only the name of the spec was stored, so retrieve it from the collection
            return specs_config.specs[key]
        pickle.UnpicklingError(f"unsupported persistent object {pid}")

class ModelStorageMissingModelsException(Exception):
    pass

class ModelsStorage(utils.Collection):
    """
    We need to store the DSPNAE models somewhere and to recall them
    easily. This class offers a straightforward interface to load
    the models.
    """
    def __init__(self, model_modulename):
        self.model_modulename = model_modulename
        self.model_module = getattr(models, self.model_modulename)
        os.chdir(project_root_dir)

    def get_rel_latent_dim(self):
        # FIXME: hacky?
        # Also: @property does not work in utils.Collection. FIXME?
        first_rel_name, first_model = next(iter(self.items()))
        if first_model is None:
            raise ModelStorageMissingModelsException()
        ret = first_model.kwargs['latent_dim']
        return ret

    def load_all_models(self):
        """
        loads the trained models for all the relations
        :return:
        """
        os.chdir(project_root_dir)
        for rel in specs_config.rels:
            logging.debug(f"loading model for {rel}..")
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
        print("model_filename",model_filename)
        print("config.trained_models_dirpath",config.trained_models_dirpath)
        if not os.path.exists(config.trained_models_dirpath):
            os.mkdir(config.trained_models_dirpath)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=None,#"train_loss",
            dirpath=config.trained_models_dirpath,
            filename=model_filename,
            save_top_k=1,
            save_last=False,
            mode="min",
            verbose=True,
            save_on_train_epoch_end=True,
            every_n_epochs=1
        )
        return checkpoint_callback

    def generate_kwargs_filename(self, model):
        """
        We need to store the parameters to the constructor as well,
        which are unfortunately not included in pytorch_lightning's
        model parameters persistency system.
        This method is returning the filename of the persisted
        construction parameters.
        :param model:
        :return:
        """

        # using tha version from the last model filename
        # as it is saved before the kwargs dump
        version = self.last_version(model.spec)
        if version is None:
            version = "UNKNOWN_VERSION"

        ret = os.path.join(
            config.trained_models_dirpath,
            f"{model.name}-v{version}.kwargs.pickle"
        )
        return ret

    def dump_kwargs(self, model):
        """
        dumps the construction parameters of the given model
        to a persistent file.
        :param model:
        :return:
        """
        kwargs_filename = self.generate_kwargs_filename(model)

        if os.path.isfile(kwargs_filename):
            # do not fully remove the file but save it to a temporary directory for eventual debugging
            utils.soft_remove(kwargs_filename)

        with open(kwargs_filename, 'wb') as f:

            # copy the kwargs in case of unpickable element removal before pickling
            kwargs = model.kwargs.copy()

            p = KwargsPickler(f)
            p.dump(kwargs)

    def filenames(self, rel, extension):
        """
        generic function that returns the names of files containing persisted aspects
        of a model trained on the data from a specific relation.
        :param rel:
        :param extension: file extension
        :return: a dictionary of filenames indexed on the version of the given aspect
        """
        filenames_glob = os.path.join(
            config.trained_models_dirpath,
            f"{self.model_modulename}__{rel.name}*.{extension}" # two dashes to be able to split model name and rel_name
        )
        ret = {}
        filenames = glob.glob(filenames_glob)
        for curr in filenames:
            m = re.match(f'.*-v(\d+).{extension}', curr)
            if m:
                # has version number
                version = int(m.groups()[0])
            else:
                # no version number in filename: this was the first
                version = 0
            ret[version] = curr
        return ret

    def kwargs_filenames(self, rel):
        """
        Returns all filenames of the persisted construction parameters for models
        of a given relation.
        :param rel:
        :return: a dictionary of filenames indexed on the version of the construction parameters
        """
        return self.filenames(rel, 'kwargs.pickle')

    def models_filenames(self, rel):
        """
        Returns all filenames of the persisted trained model parameters of models
        of a given relation.
        :param rel:
        :return: a dictionary of filenames indexed on the version of the construction parameters
        """
        return self.filenames(rel, 'ckpt')

    def last_version(self, rel):
        """
        Finds out the latest version of models trained for the given relation.
        :param rel:
        :return:
        """
        filenames = self.models_filenames(rel)
        versions = filenames.keys()
        if len(versions) == 0:
            # there was no model stored
            return None
        last_version = max(versions)
        return last_version

    def most_recent_kwargs_filename(self, rel):
        """
        returns the most recent filename containing construction parameters
        for models trained on a given relation.
        :param rel:
        :return:
        """
        last_version = self.last_version(rel)
        if last_version is None:
            # there was no model stored
            return None
        filenames = self.kwargs_filenames(rel)
        if last_version not in filenames.keys():
            raise Exception(f"cannot find kwargs file for version {last_version}")
        return filenames[last_version]

    def most_recent_model_filename(self, rel):
        """
        returns the most recent filename of persisted model paramters of a given relation.
        :param rel:
        :return:
        """
        filenames = self.models_filenames(rel)
        last_version = self.last_version(rel)
        if last_version is None:
            # there was no model stored
            return None
        return filenames[last_version]

    def rel_has_stored_model(self, rel):
        """
        returns True if there is a trained model belonging to a specific relation
        :param rel:
        :return:
        """
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
            logging.warning(f"no model for rel {rel.name}. "\
                +f"kwargs_filename={self.most_recent_kwargs_filename(rel)};"\
                +f" model_filename={self.most_recent_model_filename(rel)};"\
                +f" last_version={self.last_version(rel)};"\
                +f" model_modulename={self.model_modulename}")
            return

        kwargs_filename = self.most_recent_kwargs_filename(rel)
        logging.info(f"kwarg filename {kwargs_filename}")
        with open(kwargs_filename, 'rb') as f:
            unpickler = KwargsUnpickler(f)
            kwargs = unpickler.load()

        model_filename = self.most_recent_model_filename(rel)
        logging.info(f"model filename {model_filename}")
        if model_filename is None or not os.path.exists(model_filename):
            logging.warning(f"could not find model saved in {model_filename}")
            return

        # FIXME: duplicated from GenericModel
        logging.info(f"loading {model_filename}..")

        # FIXME: kwargs provided twice?
        model = self.model_module.Model(**kwargs)
        model.load_from_checkpoint(model_filename, **kwargs)
        return model


def test():
    """
    When this script is run directly from command-line, a test
    is being performed.
    :return:
    """
    if len(sys.argv < 2):
        logging.error('need to have a command line argument (model_modulename)')
        sys.exit(-1)
    ms = ModelsStorage(model_modulename=sys.argv[1])
    ms.load_all_models()
    print("ms", ms)
    print("done.")


if __name__ == "__main__":
    test()
