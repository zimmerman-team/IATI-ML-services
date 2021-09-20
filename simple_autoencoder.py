import torch
import numpy as np
import mlflow
import pytorch_lightning as pl
import hiddenlayer

import diagnostics
import utils
import sklearn.preprocessing
import tempfile
import relspecs

utils.set_np_printoptions()

class AE(pl.LightningModule):

    # the _diff_reduced fields contain per-feature mae of the last batch
    train_diff_reduced = None
    val_diff_reduced = None

    def __init__(self, **kwargs):
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
        self.decoder_output_layer = torch.nn.Linear(
            in_features=kwargs["layers_width"],
            out_features=kwargs["input_shape"]
        )
        self.activation_function = getattr(torch.nn, kwargs["activation_function"])()
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
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=self.kwargs['weight_decay']
        )
        return optimizer

    def training_step (self, batch, batch_idx):
        x_hat = self(batch)
        diff = batch-x_hat
        mae = torch.mean(torch.abs(diff))
        #argmaxes = torch.argmax(diff,dim=1)
        #print("location of maximum difference",argmaxes,diff[:,argmaxes])
        loss = torch.nn.functional.mse_loss(x_hat, batch)
        mse = torch.mean(diff**2)
        self.log("train_loss",loss)
        self.log("train_mae",mae)
        self.log("train_mse",mse)
        return loss

    def validation_step (self, batch, batch_idx):
        x_hat = self(batch)
        diff = batch-x_hat
        loss = torch.nn.functional.mse_loss(x_hat, batch)
        mae = torch.mean(torch.abs(diff))
        mse = torch.mean((diff)**2)
        self.log("val_loss",loss)
        self.log("val_mae",mae)
        self.log("val_mse",mse)
        self.diff_reduced = torch.mean(torch.abs(diff),dim=0)
        return loss

class ValidationErrorAnalysisCallback(pl.callbacks.Callback):
    rel = None

    # indexed by batch idx
    val_diffs = []

    # indexed by epoch
    val_mae_per_feature = []

    def __init__(self, *args, **kwargs):
        self.rel = kwargs.pop('rel')
        super().__init__(*args, **kwargs)

    def on_validation_batch_end(self, _, lm, outputs, batch, batch_idx, dataloader_idx):
        self.val_diffs.append(lm.diff_reduced)

    def on_validation_epoch_end(self, trainer, lm):
        diffs = torch.vstack(self.val_diffs)
        mae_per_feature = torch.mean(torch.abs(diffs),dim=0)
        self.val_mae_per_feature.append(mae_per_feature)
        self.log("mae_per_feature_",mae_per_feature)
        # empty the validation diffs, ready for next epoch
        self.val_diffs = []

    def teardown(self, trainer, lm, stage=None):
        print("teardown stage", stage)
        val_mae_per_feature_npa = torch.vstack(self.val_mae_per_feature).numpy()
        filename = utils.dump_npa(
            val_mae_per_feature_npa,
            prefix="val_mae_per_feature",
            suffix=".bin"
        )
        mlflow.log_artifact(filename)
        diagnostics.log_split_heatmap_artifact(val_mae_per_feature_npa,self.rel)

def log_net_visualization(model, features):
    hl_graph = hiddenlayer.build_graph(model, features)
    hl_graph.theme = hiddenlayer.graph.THEMES["blue"].copy()
    filename = tempfile.mktemp(suffix=".png")
    hl_graph.save(filename, format="png")
    mlflow.log_artifact(filename)

def main():
    mlflow.set_experiment("autoencoder_baseline")
    mlflow.pytorch.autolog()
    batch_size = 256
    model_params = dict(
        layers_width=64,
        bottleneck_width=5,
        activation_function="ELU",
        depth=5,
        weight_decay=5e-3,
        max_epochs=2,
        rel_name='budget'
    ) # to yaml file?
    mlflow.log_param("batch_size",batch_size)
    mlflow.log_params(model_params)
    rel = relspecs.rels[model_params['rel_name']]
    train_dataset,test_dataset = utils.load_tsets(rel.name)

    mlflow.log_param("train_datapoints",train_dataset.shape[0])
    mlflow.log_param("test_datapoints",test_dataset.shape[0])
    input_cardinality = train_dataset.shape[1]

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_dataset)
    train_dataset_scaled = scaler.transform(train_dataset)
    test_dataset_scaled = scaler.transform(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_scaled, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset_scaled, batch_size=batch_size, shuffle=False, num_workers=4
    )

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(
        input_shape=input_cardinality,
        **model_params
    ).to(device)

    trainer = pl.Trainer(
        limit_train_batches=0.5,
        callbacks=[ValidationErrorAnalysisCallback(rel=rel)],
        max_epochs=model_params['max_epochs']
    )
    trainer.fit(model, train_loader, test_loader)
    log_net_visualization(model,torch.zeros(batch_size, input_cardinality))

if __name__ == "__main__":
    main()
