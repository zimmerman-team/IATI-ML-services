import os
import sys
import yaml
import socket

conf_dict = None  # will be filled by load()


def mongo_uri():
    return f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_db}"


def vm_uri():
    return f"{vm_user}@{vm_host}"


def populate():
    """
    populates the attributes of the config module with the config elements
    """
    global conf_dict
    for curr in conf_dict:
        val = conf_dict[curr]
        setattr(sys.modules[__name__], curr, val)


def load():
    """
    Loads the configuration found in config/{hostname}.yaml
    This will allow to have several configuration files for the different
    hostnames the code is deployed to.
    """
    global conf_dict
    dirpath = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/../config")
    hostname = socket.gethostname()
    filename = os.path.join(dirpath, f'{hostname}.yaml')
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"cannot find {filename}")
    print(f"loading {filename}.. ", end="")
    f = open(filename, 'r')
    conf_dict = yaml.load(f, Loader=yaml.Loader)
    print("done.")


# automatically load the configuration on module import
load()
populate()
