import os
import sys
import logging

project_root_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path.insert(0, project_root_dir)

import models
from models import models_storage
from models import run
from common import specs_config

rel_name = "budget"
model_modulename = "dspn_autoencoder"

def train_and_store_model():
    global rel_name, modulename
    config_name = 'dspn_deepnarrow_short'
    dynamic_config = dict(
        model_name_suffix="modelstoragetest",
        rel_name=rel_name
    )
    model_module = getattr(models, model_modulename)
    run.run(model_module.Model, config_name, dynamic_config=dynamic_config)

def load_stored_model():
    global rel_name, model_modulename
    storage = models_storage.ModelsStorage(model_modulename=model_modulename)
    model = storage.load(specs_config.rels[rel_name])
    logging.warning(f"loaded_model: {model}")

def main():
    train_and_store_model()
    load_stored_model()
    logging.warning("all done.")

if __name__ == "__main__":
    main()
