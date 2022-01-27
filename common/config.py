import os
import sys
import yaml
import socket

# some configuration options may have default values, which are going to be eventually
# replaced by respective entries in the config file.
defaults = dict(
    models_dag_training_tasks_concurrency=1,
    train_fraction=0.9,
    tame_tqdm=True
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
    return f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_db}"


def vm_uri():
    """
    virtual machine URI, constructed from config file values
    :return:
    """
    return f"{vm_user}@{vm_host}"

def set_entry(name, val):
    """
    :param name: string name of the entry
    :param val: value of the entry
    :return: None
    """
    global _conf_dict
    setattr(sys.modules[__name__], name, val)
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
