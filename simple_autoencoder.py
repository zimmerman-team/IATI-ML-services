import torch
import numpy as np
import mlflow
import pytorch_lightning as pl
import hiddenlayer
import gc
import functools

import diagnostics
import utils
import sklearn.preprocessing
import tempfile
import relspecs

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

    def forward(self, features):
        activation = self.encoder_input_layer(features)
        activation = self.activation_function(activation)
        for curr in self.encoder_hidden_layers:
            activation = curr(activation)
            activation = self.activation_function(activation)
        code = self.encoder_output_layer(activation)
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

    def _loss(self,batch,x_hats):
        losses = []
        guess_correct = []
        batch_divided = self._divide(batch)

        # FIXME: debug with run_params['divided_output_layer'] = False
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
        diff = batch - x_hat_glued
        mae = torch.mean(torch.abs(diff))
        mse = torch.mean((diff) ** 2)
        loss = self._loss(batch,x_hat_divided)
        self.log(f"{which_tset}_loss", loss)
        self.log(f"{which_tset}_mae", mae)
        self.log(f"{which_tset}_mse", mse)

        self.diff = diff.detach().numpy()
        self.x_hat = x_hat_glued.detach().numpy()
        self.diff_reduced = np.mean(np.abs(self.diff), axis=0)
        return loss

    def training_step (self, batch, batch_idx):
        return self._step(batch,batch_idx,'train')

    def validation_step (self, batch, batch_idx):
        return self._step(batch,batch_idx,'val')

class ValidationErrorAnalysisCallback(pl.callbacks.Callback):
    rel = None
    collected = {}

    def __init__(self, *args, **kwargs):
        self.rel = kwargs.pop('rel')
        super().__init__(*args, **kwargs)
        self._init_collected()

    def _init_collected(self):
        for collection in (
                'x_hat',
                'output_last_epoch',
                # indexed by batch idx:
                'diffs',
                'diffs_reduced',
                'losses',
                'guess_correct',
                # indexed by epoch:
                'output_mean_per_feature',
                'output_var_per_feature',
                'mae_per_feature',
                'mean_losses',
                'mean_guess_correct'
            ):
            self.collected[collection] = {}
            for which_tset in ('val','train'):
                self.collected[collection][which_tset] = []

    def _collect(self,lm,which_tset):
        self.collected['diffs_reduced'][which_tset].append(lm.diff_reduced)
        self.collected['diffs'][which_tset].append(lm.diff)
        self.collected['x_hat'][which_tset].append(lm.x_hat)
        self.collected['losses'][which_tset].append(lm.losses)
        self.collected['guess_correct'][which_tset].append(lm.guess_correct)

    def on_train_batch_end(self, _, lm, outputs, batch, batch_idx, dataloader_idx):
        self._collect(lm,'train')

    def on_validation_batch_end(self, _, lm, outputs, batch, batch_idx, dataloader_idx):
        self._collect(lm,'val')

    def _epoch_end(self, which_tset, is_last_epoch=False):
        diffs_reduced = np.vstack(self.collected['diffs_reduced'][which_tset])
        x_hats = np.vstack(self.collected['x_hat'][which_tset])
        stacked_losses = np.vstack(self.collected['losses'][which_tset])
        stacked_guess_correct = np.vstack(self.collected['guess_correct'][which_tset])
        mae_per_feature = np.mean(np.abs(diffs_reduced),axis=0)
        output_var_per_feature = np.var(x_hats,axis=0)
        output_mean_per_feature = np.mean(x_hats,axis=0)
        # FIXME: this is a mean over the batch losses which are probably summed together
        mean_losses = np.mean(stacked_losses,axis=0)
        mean_guess_correct = np.mean(stacked_guess_correct,axis=0)
        self.collected['mae_per_feature'][which_tset].append(mae_per_feature)
        self.collected['output_var_per_feature'][which_tset].append(output_var_per_feature)
        self.collected['output_mean_per_feature'][which_tset].append(output_mean_per_feature)
        self.collected['mean_losses'][which_tset].append(mean_losses)
        self.collected['mean_guess_correct'][which_tset].append(mean_guess_correct)
        # empty the validation diffs, ready for next epoch
        self.collected['diffs_reduced'][which_tset] = []
        self.collected['diffs'][which_tset] = []
        if is_last_epoch:
            ar = np.arange(x_hats.shape[0])
            rc = np.random.choice(ar,size=100)
            output_last_epoch = x_hats[rc,:]
            self.collected['output_last_epoch'][which_tset] = output_last_epoch
        self.collected['x_hat'][which_tset] = []
        self.collected['losses'][which_tset] = []
        gc.collect()

    def on_train_epoch_end(self, trainer, lm):
        self._epoch_end('train', is_last_epoch=lm.current_epoch==trainer.max_epochs-1)

    def on_validation_epoch_end(self, trainer, lm):
        self._epoch_end('val', is_last_epoch=lm.current_epoch==trainer.max_epochs-1)

    def teardown(self, trainer, lm, stage=None):
        print("teardown stage", stage)
        for collected_name, heatmap_type in dict(
                mae_per_feature='fields',
                output_var_per_feature='fields',
                output_mean_per_feature='fields',
                mean_losses='losses',
                output_last_epoch='fields',
                mean_guess_correct='losses'
        ).items():
            for which_tset in self.collected[collected_name]:
                curr = self.collected[collected_name][which_tset]
                print(collected_name,which_tset)
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
    run_params = dict(
        layers_width=196,
        bottleneck_width=196,
        activation_function="ELU",
        depth=2,
        weight_decay=1e-4,
        max_epochs=500,
        rel_name='budget',
        batch_size=1024,
        divide_output_layer=True
    ) # to yaml file?
    mlflow.log_params(run_params)
    mlflow.log_artifact(__file__)
    rel = relspecs.rels[run_params['rel_name']]

    # FIXME: create tset objects so I don't have to propagate 'with_set_index' everywhere
    train_dataset,test_dataset = utils.load_tsets(rel.name,with_set_index=False)

    mlflow.log_param("train_datapoints",train_dataset.shape[0])
    mlflow.log_param("test_datapoints",test_dataset.shape[0])
    input_cardinality = train_dataset.shape[1]

    train_dataset_scaled, test_dataset_scaled = rel.make_and_fit_scalers(train_dataset,test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_scaled,
        batch_size=run_params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset_scaled,
        batch_size=run_params['batch_size'],
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
        **run_params
    ).to(device)

    trainer = pl.Trainer(
        limit_train_batches=1.0,
        callbacks=[ValidationErrorAnalysisCallback(rel=rel)],
        max_epochs=run_params['max_epochs']
    )
    trainer.fit(model, train_loader, test_loader)
    print("current mlflow run:",mlflow.active_run().info.run_id)
    #log_net_visualization(model,torch.zeros(run_params['batch_size'], input_cardinality))

if __name__ == "__main__":
    main()
