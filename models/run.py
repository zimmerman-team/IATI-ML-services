import mlflow
import mlflow.pytorch
import torch
import pytorch_lightning as pl
import numpy as np
import logging
import os
import sys
import argparse
import pickle
from common import utils, specs_config, dataset_persistency, config, timer, splits
from models import diagnostics, measurements as ms, models_storage



class MeasurementsCallback(pl.callbacks.Callback):
    """
    A pytorch_lightning model will use this callback to extract measurements.
    """
    spec = None
    collected = {}
    _train_epoch_timer = timer.Timer()
    # FIXME: this is for the refactoring: measurements = make_measurements()

    def __init__(self, *args, **kwargs):
        """
        constructor. The arguments need to provide the relation and the model
        :param args:
        :param kwargs:
        """
        self.spec = kwargs.pop('spec')
        self.model = kwargs.pop('model')
        super().__init__(*args, **kwargs)
        self.measurements = self.model.make_measurements()

    def on_train_batch_end(self, _, lm, outputs, batch, batch_idx, dataloader_idx):
        """
        Called after a batch has been used for training.
        :param _:
        :param lm:
        :param outputs:
        :param batch:
        :param batch_idx:
        :param dataloader_idx:
        :return:
        """
        self.measurements.collect(
            lm,
            'train',
            (ms.DatapointMeasurement,
             ms.BatchMeasurement)
        )

    def on_validation_batch_end(self, _, lm, outputs, batch, batch_idx, dataloader_idx):
        """
        Called after a batch has been used for validation.
        :param _:
        :param lm:
        :param outputs:
        :param batch:
        :param batch_idx:
        :param dataloader_idx:
        :return:
        """
        self.measurements.collect(
            lm,
            'test',
            (ms.DatapointMeasurement,
             ms.BatchMeasurement)
        )

    def _epoch_end(self, which_split, trainer, lm):
        """
        Run at the end of an epoch, both for training and validation sets.
        :param which_split:
        :param trainer:
        :param lm:
        :return:
        """
        is_last_epoch = lm.current_epoch == trainer.max_epochs - 1
        epoch_nr = lm.current_epoch
        mlflow.log_metric("epoch_nr",epoch_nr)

        measurements_types = [ms.EpochMeasurement]
        if is_last_epoch:
            measurements_types.append(ms.LastEpochMeasurement)
        self.measurements.collect(
            lm,
            which_split,
            measurements_types
        )
        if 'z' in self.measurements:  # FIXME: this is not abstracted
            z = self.measurements['z'].vstack(which_split)
            corr, corr_metric, mask = diagnostics.correlation(z)
            mlflow.log_metric(f"{which_split}_latent_corr_metric", corr_metric)
            diagnostics.log_correlation_heatmap_artifact("latent", corr, corr_metric, mask, which_split, epoch_nr)
        logging.debug(f"_epoch_end {which_split}")


    def on_train_epoch_start(self, trainer, lm):
        """
        Called at the beginning of a training epoch.
        Used just to reset the training epoch timer
        :param trainer:
        :param lm:
        :return:
        """
        self._train_epoch_timer.reset()

    def on_train_epoch_end(self, trainer, lm):
        """
        Called at the end of a training epoch.
        :param trainer:
        :param lm:
        :return:
        """
        self._epoch_end('train', trainer, lm)
        elapsed_time = self._train_epoch_timer.elapsed_time
        mlflow.log_metric("train_epoch_elapsed_time",elapsed_time)

    def on_validation_epoch_end(self, trainer, lm):
        """
        Called at the end of a validation epoch.
        :param trainer:
        :param lm:
        :return:
        """
        self._epoch_end('val', trainer, lm)

    def teardown(self, trainer, lm, stage=None):
        """
        Called at the very end of the training.
        Finally plots the evolution of certain metrics into an image file which is logged as artifact.
        :param trainer: unused parameter
        :param lm: also unused parameter
        :param stage: either 'fit' or 'test'
        :return:
        """
        print("teardown stage", stage)
        self.measurements.print_debug_info()
        for m in self.measurements.plottable:
            if m.plot_type in ['field', 'losses', 'latent']:  # FIXME: refactor this
                for which_split in splits.Splits.splits_names:
                    stacked_npa = m.vstack(which_split.value)
                    print(m.name, which_split.value)
                    if len(stacked_npa) == 0:
                        logging.warning(f"{m.name} {which_split} was empty")
                        continue
                    utils.log_npa_artifact(
                        stacked_npa,
                        prefix=f"{m.name}_{which_split.value}",
                        suffix=".bin"
                    )
                    diagnostics.log_heatmaps_artifact(
                        m.name,
                        stacked_npa,
                        which_split.value,
                        spec=self.spec,
                        type_=m.plot_type
                    )
                    diagnostics.log_barplots_artifact(
                        m.name,
                        stacked_npa[[-1], :],  # consider only last epoch
                        which_split.value,
                        spec=self.spec,
                        type_=m.plot_type
                    )
            else: # FIXME: refactor those ifs, there has to be a better way
                pass



# FIXME: the run function seems to be a lot of stuff, can
#   probably fix it with some additional abstraction?
def run(Model, config_name, dynamic_config={}):
    """
    Entry point to run the training of a model.
    Allows to pass parameters in dynamic_config
    in order to override config settings from a
    configuration file.
    :param Model:
    :param config_name:
    :param dynamic_config:
    :return:
    """

    utils.setup_main(dynamic_config=dynamic_config)

    model_config = utils.load_model_config(config_name, dynamic_config=dynamic_config)
    logging.debug("model_config: "+str(model_config))
    mlflow.set_experiment(model_config['experiment_name'])
    mlflow.pytorch.autolog()
    run_name = f"{model_config['config_name']}_{model_config['spec_name']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(model_config)
        logging.debug(f"__file__ {__file__}")
        mlflow.log_artifact(__file__)
        mlflow.log_artifact(model_config['config_filename'])

        # the following class method is called because according to the type of model
        # it will require different kind of data.
        spec = Model.get_spec_from_model_config(model_config)

        _splits = dataset_persistency.load_splits(
            spec,
            with_set_index=Model.with_set_index,
            cap=model_config['cap_dataset']
        )
        mlflow.log_param('tsets_creation_time', _splits.creation_time)
        for curr in _splits.splits_names:
            mlflow.log_param(f"{curr}_datapoints", _splits[curr].shape[0])

        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create a model from the Model class
        # load it to the specified device, either gpu or cpu
        model = Model(
            input_shape=_splits.item_dim,
            spec=spec,
            item_dim=_splits.item_dim,
            **model_config
        ).to(device)
        mlflow.log_artifact(model.source_filename)
        mlflow.log_param("source_filename",model.source_filename)
        mlflow.log_param("modulename",model.modulename)
        train_loader = model.make_train_loader(_splits)
        test_loader = model.make_test_loader(_splits)
        model_write_callback = model.storage.create_write_callback(model)
        callbacks = [
            MeasurementsCallback(spec=spec, model=model),
            model_write_callback
        ]
        if config.tame_tqdm:
            callbacks.append(pl.callbacks.TQDMProgressBar(refresh_rate=100))
        trainer = pl.Trainer(
            limit_train_batches=1.0,
            callbacks=callbacks,
            max_epochs=model_config['max_epochs'],
            precision=16, # mixed precision training will improve performances
            gradient_clip_val=model_config['gradient_clip_val'],
            #enable_checkpointing=True
        )
        trainer.fit(model, train_loader, test_loader)

        model.storage.dump_kwargs(model)

        print("current mlflow run:", mlflow.active_run().info.run_id, " - all done.")
        # log_net_visualization(model,torch.zeros(model_config['batch_size'], tsets.item_dim))

        # return the model to the caller so that it can eventually be used for other purposes
        return model