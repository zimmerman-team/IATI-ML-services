import logging

import torch
import numpy as np
import mlflow
import pytorch_lightning as pl
import hiddenlayer
import gc
import functools
import sys
import sklearn.preprocessing
import tempfile

from models import diagnostics
from models import measurements as ms
from common import utils, relspecs, persistency

utils.set_np_printoptions()

class AE(pl.LightningModule):

    def __init__(self, **kwargs):
        self.rel = kwargs.pop('rel',None)
        super().__init__()
        self.encoder_input_layer = torch.nn.Linear(
            in_features=kwargs["input_shape"],
            out_features=kwargs["layers_width"]
        )
        self.encoder_hidden_layers = [
            torch.nn.Linear(
                in_features=kwargs["layers_width"],
                out_features=kwargs["layers_width"]
            )
            for x in range(kwargs["depth"]-2)
        ]
        self.encoder_output_layer = torch.nn.Linear(
            in_features=kwargs["layers_width"],
            out_features=kwargs["bottleneck_width"]
        )
        self.decoder_input_layer = torch.nn.Linear(
            in_features=kwargs["bottleneck_width"],
            out_features=kwargs["layers_width"]
        )
        self.decoder_hidden_layers = [
            torch.nn.Linear(
                in_features=kwargs["layers_width"],
                out_features=kwargs["layers_width"]
            )
            for x in range(kwargs["depth"]-2)
        ]
        self.activation_function = getattr(torch.nn, kwargs["activation_function"])()
        if kwargs['divide_output_layer']:
            # instead of considering the output as a single homogeneous vector
            # its dimensionality is divided in many output layers, each belonging
            # to a specific field.
            # In this way, it's possible, for example, to apply a SoftMax activation
            # function to a categorical output section
            self.decoder_output_layers = [ # FIXME: smell
                dict(
                    layer=torch.nn.Linear(
                        in_features=kwargs["layers_width"],
                        out_features=field.n_features
                    ),
                    activation_function=(
                        field.output_activation_function or torch.nn.Identity()
                    )
                )
                for field
                in self.rel.fields
            ]
        else:
            self.decoder_output_layers = [
                dict(
                    layer=torch.nn.Linear(
                        in_features=kwargs["layers_width"],
                        out_features=kwargs["input_shape"]
                    ),
                    activation_function=torch.nn.Identity()
                )
            ]

        self.kwargs = kwargs

    def encoder(self, features):
        activation = self.encoder_input_layer(features)
        activation = self.activation_function(activation)
        for curr in self.encoder_hidden_layers:
            activation = curr(activation)
            activation = self.activation_function(activation)
        code = self.encoder_output_layer(activation)
        return code

    def decoder(self, code):
        activation = self.decoder_input_layer(code)
        activation = self.activation_function(activation)
        for curr in self.decoder_hidden_layers:
            activation = curr(activation)
            activation = self.activation_function(activation)
        reconstructed = []
        for curr in self.decoder_output_layers:
            activation_out = curr["layer"](activation)
            activation_out = curr['activation_function'](activation_out)
            reconstructed.append(activation_out)
        reconstructed = self._glue(reconstructed)
        return reconstructed

    def forward(self, features):
        self.code = self.encoder(features)
        self.reconstructed = self.decoder(self.code)
        return self.reconstructed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=self.kwargs['weight_decay']
        )
        return optimizer

    def _divide(self,tensor): # FIXME: maybe something more OO?
        if self.kwargs['divide_output_layer']:
            # actually do the division
            ret = self.rel.divide(tensor)
        else:
            # the "divided" output will just be a list with a single un-divided tensor
            ret = [tensor]
        return ret

    def _glue(self, tensor_list):
        return self.rel.glue(tensor_list)

    def _loss(self,batch,x_hats,z):
        losses = []
        guess_correct = []
        batch_divided = self._divide(batch)

        # FIXME: debug with run_config['divided_output_layer'] = False
        for curr, curr_x_hat, batch_div, field in zip(self.decoder_output_layers, x_hats, batch_divided, self.rel.fields):
            loss_fn = field.loss_function \
                      or torch.nn.functional.mse_loss
            """
            print("curr_x_hat",curr_x_hat.shape,curr_x_hat)
            print("batch_div",batch_div.shape,batch_div)
            print("loss_fn",loss_fn)
            """
            curr_loss = loss_fn(curr_x_hat, batch_div)
            guess_correct.append(field.guess_correct(curr_x_hat.detach().numpy(), batch_div.detach().numpy()))
            losses.append(curr_loss)

        loss = functools.reduce(lambda a, b: a + b, losses)

        self.latent_l1_norm = self.kwargs.pop('latent_l1_norm',0)
        loss += torch.norm(z,p=1)*self.latent_l1_norm
        self.losses = [curr.detach().numpy() for curr in losses]
        self.guess_correct = guess_correct
        return loss

    def _divide_or_glue(self, stuff):
        if type(stuff) is list:
            # stuff is already divided for various fields
            divided = stuff
            glued = self._glue(stuff)
        else:
            # stuff is already a glued-up tensor
            divided = self._divide(stuff)
            glued = stuff
        return divided, glued

    def _step(self,batch,batch_idx,which_tset):
        x_hat_divided, x_hat_glued = self._divide_or_glue(self(batch))
        z = self.encoder(batch)
        diff = batch - x_hat_glued
        mae = torch.mean(torch.abs(diff))
        mse = torch.mean((diff) ** 2)
        loss = self._loss(batch,x_hat_divided,z)
        self.log(f"{which_tset}_loss", loss)
        self.log(f"{which_tset}_mae", mae)
        self.log(f"{which_tset}_mse", mse)

        self.diff = diff.detach().numpy()
        self.x_hat = x_hat_glued.detach().numpy()
        self.z = z.detach().numpy()
        self.diff_reduced = np.mean(np.abs(self.diff), axis=0)
        return loss

    def training_step (self, batch, batch_idx):
        return self._step(batch,batch_idx,'train')

    def validation_step (self, batch, batch_idx):
        return self._step(batch,batch_idx,'val')

"""
def make_measurements():
    ret = ms.MeasurementsCollection([
        ms.DatapointMeasurement("x_hat"),
        ms.DatapointMeasurement("z",diagnostics.correlation),

        ms.LastEpochMeasurement("output_last_epoch"),
        ms.LastEpochMeasurement("latent_last_epoch"),

        ms.BatchMeasurement('diff'),
        ms.BatchMeasurement('diff_reduced'),
        ms.BatchMeasurement('losses'),
        ms.BatchMeasurement('guess_correct'),
        ms.BatchMeasurement('latent_l1_norm'),

        ms.EpochMeasurement("output_mean_per_feature"),
        ms.EpochMeasurement("output_var_per_feature",ms.var),
        ms.EpochMeasurement("mae_per_feature",ms.mae),
        ms.EpochMeasurement("mean_losses"),
        ms.EpochMeasurement("mean_guess_correct"),
        ms.EpochMeasurement("mean_latent_l1_norm")
    ])
    return ret
"""

class MeasurementsCallback(pl.callbacks.Callback):
    rel = None
    collected = {}
    # FIXME: this is for the refactoring: measurements = make_measurements()

    def __init__(self, *args, **kwargs):
        self.rel = kwargs.pop('rel')
        super().__init__(*args, **kwargs)
        self._init_collected()

    def _init_collected(self):
        for collection in (
                'x_hat',
                'z',
                'output_last_epoch',
                'latent_last_epoch',
                # indexed by batch idx:
                'diff',
                'diff_reduced',
                'losses',
                'guess_correct',
                'latent_l1_norm',
                # indexed by epoch:
                'output_mean_per_feature',
                'output_var_per_feature',
                'mae_per_feature',
                'mean_losses',
                'mean_guess_correct',
                'mean_latent_l1_norm'
        ):
            self.collected[collection] = {}
            for which_tset in ('val', 'train'):
                self.collected[collection][which_tset] = []

    def _collect(self,lm,which_tset):
        self.collected['diff_reduced'][which_tset].append(lm.diff_reduced)
        self.collected['diff'][which_tset].append(lm.diff)
        self.collected['x_hat'][which_tset].append(lm.x_hat)
        self.collected['z'][which_tset].append(lm.z)
        self.collected['losses'][which_tset].append(lm.losses)
        self.collected['guess_correct'][which_tset].append(lm.guess_correct)
        self.collected['latent_l1_norm'][which_tset].append(lm.latent_l1_norm)

    def on_train_batch_end(self, _, lm, outputs, batch, batch_idx, dataloader_idx):
        # FIXME: this is for the refactoring: self.measurements.collect(lm,utils.Tsets.TRAIN)
        self._collect(lm, 'train')

    def on_validation_batch_end(self, _, lm, outputs, batch, batch_idx, dataloader_idx):
        # FIXME: this is for the refactoring: self.measurements.collect(lm,utils.Tsets.VAL)
        self._collect(lm, 'val')

    def _epoch_end(self, which_tset, trainer, lm):
        is_last_epoch = lm.current_epoch == trainer.max_epochs - 1
        epoch_nr = lm.current_epoch
        diff_reduced = np.vstack(self.collected['diff_reduced'][which_tset])
        x_hats = np.vstack(self.collected['x_hat'][which_tset])
        stacked_losses = np.vstack(self.collected['losses'][which_tset])
        stacked_guess_correct = np.vstack(self.collected['guess_correct'][which_tset])
        stacked_latent_l1_norm = np.vstack(self.collected['latent_l1_norm'][which_tset])
        mae_per_feature = np.mean(np.abs(diff_reduced),axis=0)
        output_var_per_feature = np.var(x_hats,axis=0)
        output_mean_per_feature = np.mean(x_hats,axis=0)
        # FIXME: this is a mean over the batch losses which are probably summed together
        mean_losses = np.mean(stacked_losses,axis=0)
        mean_guess_correct = np.mean(stacked_guess_correct,axis=0)
        mean_latent_l1_norm = np.mean(stacked_latent_l1_norm,axis=0)
        self.collected['mae_per_feature'][which_tset].append(mae_per_feature)
        self.collected['output_var_per_feature'][which_tset].append(output_var_per_feature)
        self.collected['output_mean_per_feature'][which_tset].append(output_mean_per_feature)
        self.collected['mean_losses'][which_tset].append(mean_losses)
        self.collected['mean_guess_correct'][which_tset].append(mean_guess_correct)
        self.collected['mean_latent_l1_norm'][which_tset].append(mean_latent_l1_norm)
        # empty the validation diffs, ready for next epoch
        self.collected['diff_reduced'][which_tset] = []
        self.collected['diff'][which_tset] = []
        z = np.vstack(self.collected['z'][which_tset])

        corr, corr_metric,mask = diagnostics.correlation(z)
        mlflow.log_metric(f"{which_tset}_latent_corr_metric",corr_metric)
        diagnostics.log_correlation_heatmap_artifact("latent",corr,corr_metric,mask,which_tset,epoch_nr)
        if is_last_epoch:
            ar = np.arange(x_hats.shape[0])
            rc = np.random.choice(ar,size=100)
            output_last_epoch = x_hats[rc,:]
            latent_last_epoch = z[rc,:]
            self.collected['output_last_epoch'][which_tset] = output_last_epoch
            self.collected['latent_last_epoch'][which_tset] = latent_last_epoch
        self.collected['x_hat'][which_tset] = []
        self.collected['z'][which_tset] = []
        self.collected['losses'][which_tset] = []
        self.collected['guess_correct'][which_tset] = []
        gc.collect()

    def on_train_epoch_end(self, trainer, lm):
        self._epoch_end( 'train', trainer, lm )

    def on_validation_epoch_end(self, trainer, lm):
        self._epoch_end( 'val', trainer, lm )

    def teardown(self, trainer, lm, stage=None):
        print("teardown stage", stage)
        for collected_name, heatmap_type in dict(
                mae_per_feature='fields',
                output_var_per_feature='fields',
                output_mean_per_feature='fields',
                mean_losses='losses',
                mean_guess_correct='losses',
                output_last_epoch='fields',
                latent_last_epoch='latent',
                mean_latent_l1_norm='losses'
        ).items():
            for which_tset in self.collected[collected_name]:
                curr = self.collected[collected_name][which_tset]
                print(collected_name,which_tset)
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
                    prefix=f"{collected_name}_{which_tset}",
                    suffix=".bin"
                )
                diagnostics.log_heatmaps_artifact(
                    collected_name,
                    stacked_npa,
                    which_tset,
                    rel=self.rel,
                    type_=heatmap_type
                )
                diagnostics.log_barplots_artifact(
                    collected_name,
                    stacked_npa[[-1],:], # consider only last epoch
                    which_tset,
                    rel=self.rel,
                    type_=heatmap_type
                )

def log_net_visualization(model, features):
    hl_graph = hiddenlayer.build_graph(model, features)
    hl_graph.theme = hiddenlayer.graph.THEMES["blue"].copy()
    filename = tempfile.mktemp(suffix=".png")
    hl_graph.save(filename, format="png")
    mlflow.log_artifact(filename)

def main():
    mlflow.set_experiment("autoencoder_baseline")
    mlflow.pytorch.autolog()
    config_name = sys.argv[1]
    run_config = utils.load_run_config(config_name)
    with mlflow.start_run(run_name=run_config['config_name'])
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

        # create a model from `AE` autoencoder class
        # load it to the specified device, either gpu or cpu
        model = AE(
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
        print("current mlflow run:",mlflow.active_run().info.run_id)
        #log_net_visualization(model,torch.zeros(run_config['batch_size'], input_cardinality))

if __name__ == "__main__":
    main()
