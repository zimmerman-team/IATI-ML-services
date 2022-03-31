import os
import sys
import yaml
import socket

from common import utils
# some configuration options may have default values, which are going to be eventually
# replaced by respective entries in the config file.
defaults = dict(
    models_dag_training_tasks_concurrency=1,
    train_fraction=0.9,
    tame_tqdm=True,
    trained_models_dirpath='trained_models/'
)

_conf_dict = defaults.copy() # entries may be replaced by load()

def entries_names():
    """
    :return: the list of names of the config attributes that are set
    """
    global _conf_dict
    return list(_conf_dict.keys())

def mongo_uri():
    """
    connection URI, constructed from config file values
    :return:
    """
    # NOTE: these apparently missing module-wide variables are actually
    # set dynamically via populate()/set_entry(..)
    return f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_db}"

def get_airflow_sqlalchemy_conn():
    global _conf_dict
    password = _conf_dict['airflow_pg_password']
    return f"postgresql+psycopg2://airflow_user:{password}@localhost:5432/airflow_db"

def vm_uri():
    """
    virtual machine URI, constructed from config file values
    :return:
    """
    return f"{vm_user}@{vm_host}"

class NoSuchConfItemException(Exception):
    pass

class NestedConfig(object):

    def __init__(self, _conf_dict=dict()):
        self.__dict__['_conf_dict'] = _conf_dict

    def __setattr__(self, key, value):
        self._conf_dict[key] = value

    def __getattr__(self, key):
        if key in self._conf_dict.keys():
            return self._conf_dict[key]
        else:
            raise NoSuchConfItemException(key)

def _is_list_of_dicts(val):
    return type(val) is list \
           and len(val) > 0 \
           and type(val[0]) is dict

def set_entry(name, val, where=None):
    """
    :param name: string name of the entry
    :param val: value of the entry
    :return: None
    """
    if where is None:
        # defaults to root (the config module)
        where = sys.modules[__name__]
    _conf_dict = getattr(where, '_conf_dict')

    if _is_list_of_dicts(val):
        new_obj = NestedConfig()
        for curr_dict in val:
            for k,v in curr_dict.items():
                # recursion
                set_entry(k,v,where=new_obj)
        val = new_obj # replace the yaml dict with the NestedConfig object

    setattr(where, name, val)
    _conf_dict[name] = val
    if name == "train_fraction":
        test_fraction = 1.0 - val
        set_entry("test_fraction",test_fraction)

def populate():
    """
    populates the attributes of the config module with the config elements
    """
    global _conf_dict
    for curr in _conf_dict:
        val = _conf_dict[curr]
        set_entry(curr, val)
    home = os.path.expanduser('~')
    set_entry('home',home)
    set_entry('airflow_sqlalchemy_conn',get_airflow_sqlalchemy_conn())

def d_options():
    global _conf_dict
    ret = ""
    for name,val in _conf_dict.items():
        ret += f"-Dm4_{name.upper()}={val} "
    return ret

def spaced_options():
    global _conf_dict
    ret = ""
    for name,val in _conf_dict.items():
        ret += f"{name.upper()} {val}---"
    return ret

def load():
    """
    Loads the configuration found in config/{hostname}.yaml
    This will allow to have several configuration files for the different
    hostnames the code is deployed to.
    """
    global _conf_dict
    dirpath = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../config")
    hostname = socket.gethostname()
    filename = os.path.join(dirpath, f'{hostname}.yaml')
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"cannot find {filename}")
    print(f"loading {filename}.. ", end="")
    f = open(filename, 'r')
    yaml_conf = yaml.load(f, Loader=yaml.Loader)
    for k,v in yaml_conf.items():
        set_entry(k,v)
    print("done.")


# automatically load the configuration on module import
load()
populate()
