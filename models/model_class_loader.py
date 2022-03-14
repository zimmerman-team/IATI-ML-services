import sys
import os

project_root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [project_root_path]+sys.path

import models

models_training_on = {
    'rels': [
        'dspn_autoencoder',
        'idspn_autoencoder',
        'item_autoencoder'
    ],
    'activity': [
        'activity_autoencoder'
    ]
}

def is_model_module(modelname):
    if hasattr(getattr(models, modelname), 'Model'):
        return True
    return False

def all_modelnames():
    ret = []
    for curr in dir(models):
        if is_model_module(curr):
            ret.append(curr)
    return ret

def get_model_class(modelname):
    assert is_model_module(modelname)
    module = getattr(models, modelname)
    modelclass = module.Model
    return modelclass

def does_model_train_on(modelname, dataset_type):
    global models_training_on_rels
    assert is_model_module(modelname)
    assert dataset_type in models_training_on.keys()
    return modelname in models_training_on[dataset_type]
