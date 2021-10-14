import mlflow
import torch
import pytorch_lightning as pl
import numpy as np
import logging

from common import utils, relspecs, persistency
from models import diagnostics, measurements as ms

def make_measurements():
    ret = ms.MeasurementsCollection([
        ms.DatapointMeasurement("x_hat", dst=dict(
            output_mean_per_feature=ms.mean,
            output_var_per_feature=ms.var,
            output_last_epoch=ms.random_sampling
        )),
        ms.DatapointMeasurement("z",diagnostics.correlation, dst=dict(
            latent_last_epoch=ms.random_sampling
        )),

        ms.LastEpochMeasurement("output_last_epoch"),
        ms.LastEpochMeasurement("latent_last_epoch"),

        ms.BatchMeasurement('diff'),
        ms.BatchMeasurement('diff_reduced', dst=dict(
            mae_per_feature=ms.mae
        )),
        ms.BatchMeasurement('losses', dst=dict(
            mean_losses=ms.mean
        )),
        ms.BatchMeasurement('guess_correct', dst=dict(
            mean_guess_correct=ms.mean
        )),
        ms.BatchMeasurement('latent_l1_norm', dst=dict(
            mean_latent_l1_norm=ms.mean
        )),

        ms.EpochMeasurement("output_mean_per_feature"),
        ms.EpochMeasurement("output_var_per_feature"),
        ms.EpochMeasurement("mae_per_feature"),
        ms.EpochMeasurement("mean_losses"),
        ms.EpochMeasurement("mean_guess_correct"),
        ms.EpochMeasurement("mean_latent_l1_norm")
    ])
    return ret

class MeasurementsCallback(pl.callbacks.Callback):
    rel = None
    collected = {}
    # FIXME: this is for the refactoring: measurements = make_measurements()

    def __init__(self, *args, **kwargs):
        self.rel = kwargs.pop('rel')
        super().__init__(*args, **kwargs)
        self.measurements = make_measurements()

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
        z = self.measurements['z'].vstack(which_tset)

        corr, corr_metric,mask = diagnostics.correlation(z)
        mlflow.log_metric(f"{which_tset}_latent_corr_metric",corr_metric)
        diagnostics.log_correlation_heatmap_artifact("latent",corr,corr_metric,mask,which_tset,epoch_nr)

    def on_train_epoch_end(self, trainer, lm):
        self._epoch_end( 'train', trainer, lm )

    def on_validation_epoch_end(self, trainer, lm):
        self._epoch_end( 'val', trainer, lm )

    def teardown(self, trainer, lm, stage=None):
        print("teardown stage", stage)
        self.measurements.print_debug_info()
        for collected_name, heatmap_type in dict( # FIXME refactor
                mae_per_feature='fields',
                output_var_per_feature='fields',
                output_mean_per_feature='fields',
                mean_losses='losses',
                mean_guess_correct='losses',
                output_last_epoch='fields',
                latent_last_epoch='latent',
                mean_latent_l1_norm='losses'
        ).items():
            for which_tset in utils.Tsets:
                curr = self.measurements[collected_name].data[which_tset.value]
                print(collected_name,which_tset.value)
                if curr is None or len(curr)==0:
                    logging.warning("f{collected_name} {which_tset} was empty")
                    continue
                if type(curr) is list:
                    stacked_npa = np.vstack(curr)
                elif type(curr) is np.ndarray:
                    stacked_npa = curr
                else:
                    raise Exception("collected type "+str(type(curr))+" not understood")
                utils.log_npa_artifact(
                    stacked_npa,
                    prefix=f"{collected_name}_{which_tset.value}",
                    suffix=".bin"
                )
                diagnostics.log_heatmaps_artifact(
                    collected_name,
                    stacked_npa,
                    which_tset.value,
                    rel=self.rel,
                    type_=heatmap_type
                )
                diagnostics.log_barplots_artifact(
                    collected_name,
                    stacked_npa[[-1],:], # consider only last epoch
                    which_tset.value,
                    rel=self.rel,
                    type_=heatmap_type
                )

def run(Model,config_name):
    run_config = utils.load_run_config(config_name)
    mlflow.set_experiment(run_config['experiment_name'])
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=run_config['config_name']):
        mlflow.log_params(run_config)
        mlflow.log_artifact(__file__)
        mlflow.log_artifact(run_config['config_filename'])
        rel = relspecs.rels[run_config['rel_name']]

        # FIXME: create tset objects so I don't have to propagate 'with_set_index' everywhere
        tsets = persistency.load_tsets(rel,with_set_index=False)

        for curr in tsets.tsets_names:
            mlflow.log_param(f"{curr}_datapoints",tsets[curr].shape[0])
        input_cardinality = tsets.train.shape[1]

        train_loader = torch.utils.data.DataLoader(
            tsets.train_scaled,
            batch_size=run_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=False
        )

        test_loader = torch.utils.data.DataLoader(
            tsets.test_scaled,
            batch_size=run_config['batch_size'],
            shuffle=False,
            num_workers=4
        )

        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create a model from the Model class
        # load it to the specified device, either gpu or cpu
        model = Model(
            input_shape=input_cardinality,
            rel=rel,
            **run_config
        ).to(device)

        trainer = pl.Trainer(
            limit_train_batches=1.0,
            callbacks=[MeasurementsCallback(rel=rel)],
            max_epochs=run_config['max_epochs']
        )
        trainer.fit(model, train_loader, test_loader)
        print("current mlflow run:",mlflow.active_run().info.run_id, " - all done.")
        #log_net_visualization(model,torch.zeros(run_config['batch_size'], input_cardinality))
