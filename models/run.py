import mlflow
import torch
import pytorch_lightning as pl
import numpy as np
import logging
import os
import sys
import argparse

from common import utils, relspecs, persistency
from models import diagnostics, measurements as ms

def get_args():
    args = {}
    for arg in sys.argv:
        if arg.startswith(("--")):
            k = arg.split('=')[0][2:]
            v = arg.split('=')[1]
            args[k] = v
    return args

class MeasurementsCallback(pl.callbacks.Callback):
    rel = None
    collected = {}
    # FIXME: this is for the refactoring: measurements = make_measurements()

    def __init__(self, *args, **kwargs):
        self.rel = kwargs.pop('rel')
        self.model = kwargs.pop('model')
        super().__init__(*args, **kwargs)
        self.measurements = self.model.make_measurements()

    def on_train_batch_end(self, _, lm, outputs, batch, batch_idx, dataloader_idx):
        self.measurements.collect(
            lm,
            utils.Tsets.TRAIN.value,
            (ms.DatapointMeasurement,
             ms.BatchMeasurement)
        )

    def on_validation_batch_end(self, _, lm, outputs, batch, batch_idx, dataloader_idx):
        self.measurements.collect(
            lm,
            utils.Tsets.VAL.value,
            (ms.DatapointMeasurement,
             ms.BatchMeasurement)
        )

    def _epoch_end(self, which_tset, trainer, lm):
        is_last_epoch = lm.current_epoch == trainer.max_epochs - 1
        epoch_nr = lm.current_epoch

        measurements_types = [ms.EpochMeasurement]
        if is_last_epoch:
            measurements_types.append(ms.LastEpochMeasurement)
        self.measurements.collect(
            lm,
            which_tset,
            measurements_types
        )
        if 'z' in self.measurements:  # FIXME: this is not abstracted
            z = self.measurements['z'].vstack(which_tset)
            corr, corr_metric, mask = diagnostics.correlation(z)
            mlflow.log_metric(f"{which_tset}_latent_corr_metric", corr_metric)
            diagnostics.log_correlation_heatmap_artifact("latent", corr, corr_metric, mask, which_tset, epoch_nr)

    def on_train_epoch_end(self, trainer, lm):
        self._epoch_end( 'train', trainer, lm )

    def on_validation_epoch_end(self, trainer, lm):
        self._epoch_end( 'val', trainer, lm )

    def teardown(self, trainer, lm, stage=None):
        print("teardown stage", stage)
        self.measurements.print_debug_info()
        for m in self.measurements.plottable:
            for which_tset in utils.Tsets:
                stacked_npa = m.vstack(which_tset.value)
                print(m.name,which_tset.value)
                if len(stacked_npa) == 0:
                    logging.warning(f"{m.name} {which_tset} was empty")
                    continue
                utils.log_npa_artifact(
                    stacked_npa,
                    prefix=f"{m.name}_{which_tset.value}",
                    suffix=".bin"
                )
                diagnostics.log_heatmaps_artifact(
                    m.name,
                    stacked_npa,
                    which_tset.value,
                    rel=self.rel,
                    type_=m.plot_type
                )
                diagnostics.log_barplots_artifact(
                    m.name,
                    stacked_npa[[-1],:],  # consider only last epoch
                    which_tset.value,
                    rel=self.rel,
                    type_=m.plot_type
                )

def run(Model,config_name, dynamic_config={}):

    # need to make sure that logs/* and mlruns/* are generated
    # in the correct project root directory, as well as
    # config files are loaded from model_config/
    project_root_dir= os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..' # parent directory of models/
    ))
    os.chdir(project_root_dir)

    # gets args from command line that end up in the model run's configuration
    # and overrides the eventual given dynamic_config with the passed arguments
    # as in --rel_name=activity_date for example
    args = get_args()
    for arg, val in args.items():
        dynamic_config[arg] = val

    log_filename = os.path.join("logs",utils.strnow_compact()+'.log')
    logging.basicConfig(filename=log_filename, filemode='w', level=logging.DEBUG)
    model_config = utils.load_model_config(config_name, dynamic_config=dynamic_config)
    mlflow.set_experiment(model_config['experiment_name'])
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=model_config['config_name']):
        mlflow.log_params(model_config)
        print("__file__",__file__)
        mlflow.log_artifact(__file__)
        mlflow.log_artifact(model_config['config_filename'])
        rel = relspecs.rels[model_config['rel_name']]

        tsets = persistency.load_tsets(
            rel,
            with_set_index=Model.with_set_index,
            cap=model_config['cap_dataset']
        )

        for curr in tsets.tsets_names:
            mlflow.log_param(f"{curr}_datapoints",tsets[curr].shape[0])

        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create a model from the Model class
        # load it to the specified device, either gpu or cpu
        model = Model(
            input_shape=tsets.item_dim,
            rel=rel,
            item_dim=tsets.item_dim,
            **model_config
        ).to(device)

        train_loader = model.make_train_loader(tsets)
        test_loader = model.make_test_loader(tsets)

        trainer = pl.Trainer(
            limit_train_batches=1.0,
            callbacks=[MeasurementsCallback(rel=rel,model=model)],
            max_epochs=model_config['max_epochs']
        )
        trainer.fit(model, train_loader, test_loader)
        print("current mlflow run:",mlflow.active_run().info.run_id, " - all done.")
        #log_net_visualization(model,torch.zeros(model_config['batch_size'], tsets.item_dim))
