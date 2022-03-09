import logging
import sys
from bson.binary import Binary
import pickle
import zlib
import numpy as np
import mlflow
import urllib
import os
import shutil
import tempfile
import glob
import torch
import enum
import yaml
import collections
import datetime
import copy
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

class Collection(dict):
    """
    In a utils.Collection, which inherits from `dict`,dynamic properties
    can be set as `c['the_property'] = the_value`
    and accessed as in `c.the_property`.
    Moreover the `.names` property returns the properties that
    have been set.
    """

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            lst = args[0]
            assert type(lst) is list, \
                "first argument of the collection has to be a list of elements"
            for curr in lst:
                if curr.name in self.names:
                    raise Exception(f"element {curr.name} already in this Collection")
                self[curr.name] = curr
        super().__init__(**kwargs)

    def __iter__(self):
        """
        Iterating on the utils.Collection results in the values
        of the properties set, not the names of the properties.
        :return: iterator on the values of the properties that have been set
        """
        return iter(self.values())

    @property
    def names(self):
        """
        :return: list of string names of the properties set
        """
        return self.keys()

    def __getattr__(self, name):
        """
        in a utils.Collection, dynamic properties can be set as
        `c['the_property'] = the_value`
        and accessed as in `c.the_property`
        :param name: the name of the property being accessed
        :return: the value of the property
        """
        if name[0] == '_':
            return self.__getattribute__(name)
        assert name in self.names, f"{name} not in collection's names"
        return self[name]

    def __add__(self, addendum):

        # new object is going to be returned,
        # without altering the source one
        new = copy.deepcopy(self)
        for curr in addendum:
            new.add(curr)
        return new

    def add(self, item):
        self[item.name] = item


class Tsets(enum.Enum):
    TRAIN = 'train'  # training set
    VAL = 'val'  # validation set


class OneHotCrossEntropyLoss(object):
    def __init__(self, weight=None):
        # print("OneHostCrossEntropyLoss weight:",weight)
        if weight is not None:
            self.cross_entropy_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(weight))
        else:
            self.cross_entropy_fn = torch.nn.CrossEntropyLoss()

    def __call__(self, x_hat, batch):
        labels = batch.argmax(1)
        ret = self.cross_entropy_fn(x_hat, labels)
        return ret


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1

    def inverse_transform(self, input_array, y=None):
        return input_array * 1


def serialize(npa):
    return Binary(zlib.compress(pickle.dumps(npa, protocol=2)))


def deserialize(buf):
    return pickle.loads(zlib.decompress(buf)).astype(np.float32)


def set_np_printoptions():
    np.set_printoptions(
        suppress=True,  # no unnecessary scientific notation
        linewidth=319,  # prevents too many newlines
        formatter=dict(float=lambda x: "%.3g" % x),  # remove extra spaces
        threshold=np.inf
    )


def inspect_tset(tset):
    print(tset[0:2])


def dump_npa(npa, prefix="", suffix=""):
    buf = serialize(npa)
    filename = tempfile.mktemp(prefix=prefix, suffix=suffix)
    with open(filename, "wb+") as f:
        f.write(buf)
        f.flush()
    return filename


def log_npa_artifact(npa, prefix="some_npa", suffix=".bin"):
    npa_filename = dump_npa(
        npa,
        prefix=prefix,
        suffix=suffix
    )
    mlflow.log_artifact(npa_filename)


def load_npa(filename):
    with open(filename, 'rb') as f:
        buf = f.read()
        npa = deserialize(buf)
        return npa


def glob_artifacts(run_id, prefix="", suffix=""):
    run = mlflow.get_run(run_id)
    auri = run.info.artifact_uri
    apath = urllib.parse.urlparse(auri).path
    search = os.path.join(apath, prefix+"*"+suffix)
    found = glob.glob(search)
    found = sorted(found)
    return found


def load_npa_artifact(experiment_name, run_id, prefix="", suffix=""):
    found = glob_artifacts(run_id, prefix=prefix, suffix=suffix)
    npa = load_npa(found[0])
    return npa


def is_seq(stuff):
    return type(stuff) in (tuple, list)


def fn_across(stuff, fn):
    if is_seq(stuff):
        return fn(list(map(lambda curr: fn_across(curr, fn), stuff)))
    else:
        return fn(stuff)


def min_across(stuff):
    return fn_across(stuff, np.min)


def max_across(stuff):
    return fn_across(stuff, np.max)


def str_shapes(stuff):
    if is_seq(stuff):
        return "["+" ".join([str_shapes(curr) for curr in stuff])+"]"
    else:
        return str(stuff.shape)


def load_model_config(config_name, dynamic_config=None):
    if os.path.exists(config_name):
        # a filename is given
        filename = config_name
    else:
        # config name, then resolved to a filename is given
        directory = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'model_config/'))
        filename = os.path.join(directory, config_name+".yaml")
    with open(filename, 'r') as f:
        ret = yaml.load(f, Loader=yaml.Loader)

    # default values for missing parameters
    ret['cap_dataset'] = ret.get('cap_dataset', None)
    ret['gradient_clip_val'] = 1000
    ret['config_name'] = config_name
    ret['config_filename'] = filename
    ret['dspn_loss_fn'] = "smooth_l1"

    # dynamic config generation will override the yaml file config
    if dynamic_config is not None:
        for k, v in dynamic_config.items():
            logging.info(f"configuration item {k} dynamically set at {v}")
            ret[k] = v
    return ret


def dict_to_obj(typename, d):
    T = collections.namedtuple(typename, d.keys())
    obj = T(d)
    return obj


def strnow_iso():
    now = str(datetime.datetime.now())
    return now


def strnow_compact():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def is_vectoriform(stuff):
    return type(stuff) in (np.ndarray, list, tuple)


def is_empty(stuff):
    assert is_vectoriform(stuff), f"is_empty: object of type {type(stuff)} is not vectoriform"
    if type(stuff) in (list, tuple):
        return len(stuff) == 0
    if type(stuff) is np.ndarray:
        if len(stuff.shape) == 0:
            return True
        elif stuff.shape[0] == 0:
            return True
        else:
            return False
    raise Exception(f"is_empty: unhandled case for type {type(stuff)}")


def debug(*args):
    msg = " ".join([str(a) for a in args])
    logging.debug(msg)

def glue(tensor_list):
    """
    given a list of tensor, being the values of the fields,
    returns a glued-up tensor to be used in ML model training (or query).
    :param tensor_list:
    :return:
    """
    if type(tensor_list) is list:
        assert len(tensor_list) > 0
        first = tensor_list[0]
        if type(first) is torch.Tensor:
            ret = torch.hstack(tensor_list)
        elif type(first) is np.ndarray:
            ret = np.hstack(tensor_list)
        else:
            raise Exception("elements in the list must be either numpy arrays or torch tensors")
    else:
        # possibly already glued?
        ret = tensor_list
    return ret

def soft_remove(filename):
    basename = os.path.basename(filename)
    new_filename = tempfile.mktemp() + basename
    # move the file out of the way to the temporary directory
    shutil.move(filename, new_filename)


def setup_logging():
    log_level_nr = getattr(logging,config.log_level,logging.INFO)
    # setting log lever for stdout
    logging.basicConfig( level=log_level_nr)

    # the logs will also end up in a file
    log_filename = os.path.join(os.getcwd(),"logs", strnow_compact()+'.log')
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        level=log_level_nr
    )
    print("logging level desired (nr):",log_level_nr,
          " - obtained:",logging.getLevelName(logging.getLogger().level),
          " - also onto file:",log_filename
          )
    logging.debug("test DEBUG message")
    logging.info("test INFO message")
    logging.warning("test WARNING message")

def get_args():
    """
    Simple command-line arguments extraction system
    :return:
    """
    args = {}
    for arg in sys.argv:
        if arg.startswith("--"):
            k = arg.split('=')[0][2:]
            v = arg.split('=')[1]
            args[k] = v
    return args

def setup_main(dynamic_config={}):
    # need to make sure that logs/* and mlruns/* are generated
    # in the correct project root directory, as well as
    # config files are loaded from model_config/
    project_root_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..'  # parent directory of models/ or common/
    ))
    os.chdir(project_root_dir)

    # gets args from command line that end up in the model run's configuration
    # and overrides the eventual given dynamic_config with the passed arguments
    # as in --rel_name=activity_date for example
    args = get_args()
    for arg, val in args.items():
        dynamic_config[arg] = val
        if arg in config.entries_names():
            # config file entries will be overriden by command-line entries
            config.set_entry(arg,val)

    try:
        os.mkdir("logs")
    except FileExistsError:
        pass

    setup_logging()


def create_set_npa(spec,data):
    """
    FIXME: move to some other module?
    :param spec:
    :param data:
    :return:
    """
    set_npas = []
    keys = spec.fields_names
    if data is None:
        data = dict(**{k:[] for k in keys}) # FIXME: inefficient to create this for every None?
    for k in keys:  # we need to always have a same ordering of the fields!
        if len(data[k]) > 0 and type(data[k][0]) is list:
            floats = list(map(lambda v: list(map(lambda x: float(x), v)), data[k]))
        else:  # not something that is dummified: simple numerical value field
            floats = list(map(lambda x: [float(x)], data[k]))
        field_npa = np.array(floats)
        set_npas.append(field_npa)
    if len(set(map(lambda curr: curr.shape[0], set_npas))) > 1:
        logging.info("keys:" + str(keys))
        logging.info("set_npas shapes:" + str([curr.shape for curr in set_npas]))
    set_npa = np.hstack(set_npas)
    return set_npa

