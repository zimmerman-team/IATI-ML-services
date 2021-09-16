import torch
import pymongo
import pickle
import numpy as np
import mlflow
import pytorch_lightning as pl
import hiddenlayer
import tempfile
import utils

MONGODB_CONN="mongodb://mongouser:XGkS1wDyb4922@localhost:27017/learning_sets"

class AE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_input_layer = torch.nn.Linear(
            in_features=kwargs["input_shape"],
            out_features=kwargs["encoder_width"]
        )
        self.encoder_hidden_layers = [
            torch.nn.Linear(
                in_features=kwargs["encoder_width"],
                out_features=kwargs["encoder_width"]
            )
            for x in range(kwargs["depth"]-2)
        ]
        self.encoder_output_layer = torch.nn.Linear(
            in_features=kwargs["encoder_width"],
            out_features=kwargs["bottleneck_width"]
        )
        self.decoder_input_layer = torch.nn.Linear(
            in_features=kwargs["bottleneck_width"],
            out_features=kwargs["decoder_width"]
        )
        self.decoder_hidden_layers = [
            torch.nn.Linear(
                in_features=kwargs["decoder_width"],
                out_features=kwargs["decoder_width"]
            )
            for x in range(kwargs["depth"]-2)
        ]
        self.decoder_output_layer = torch.nn.Linear(
            in_features=kwargs["decoder_width"],
            out_features=kwargs["input_shape"]
        )
        self.activation_function = getattr(torch.nn, kwargs["activation_function"])()

    def forward(self, features):
        activation = self.encoder_input_layer(features)
        activation = self.activation_function(activation)
        for curr in self.encoder_hidden_layers:
            activation = curr(activation)
            activation = self.activation_function(activation)
        code = self.encoder_output_layer(activation)
        activation = self.decoder_input_layer(code)
        activation = self.activation_function(activation)
        for curr in self.encoder_hidden_layers:
            activation = curr(activation)
            activation = self.activation_function(activation)
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step (self, batch, batch_idx):
        x_hat = self(batch)
        loss = torch.nn.functional.mse_loss(x_hat, batch)
        self.log("train_loss",loss)
        return loss

    def validation_step (self, batch, batch_idx):
        x_hat = self(batch)
        loss = torch.nn.functional.mse_loss(x_hat, batch)
        self.log("val_loss",loss)
        return loss

def log_net_visualization(model, features):
    hl_graph = hiddenlayer.build_graph(model, features)
    hl_graph.theme = hiddenlayer.graph.THEMES["blue"].copy()
    filename = tempfile.mktemp(suffix=".png")
    hl_graph.save(filename, format="png")
    mlflow.log_artifact(filename)

def main():
    mlflow.pytorch.autolog()
    batch_size = 256
    model_params = dict(
        encoder_width=192,
        decoder_width=192,
        bottleneck_width=128,
        activation_function="Tanh",
        depth=5
    )
    mlflow.log_param("batch_size",batch_size)
    mlflow.log_params(model_params)

    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    coll = db['npas_tsets']
    document = coll.find({'rel':'budget'}).sort('_id', pymongo.DESCENDING).limit(1)[0]
    train_dataset = utils.deserialize(document['train_npa']).astype(np.float32)
    test_dataset = utils.deserialize(document['test_npa']).astype(np.float32)

    print("train_dataset.shape",train_dataset.shape)
    mlflow.log_param("train_datapoints",train_dataset.shape[0])
    mlflow.log_param("test_datapoints",test_dataset.shape[0])
    input_cardinality = train_dataset.shape[1]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(
        input_shape=input_cardinality,
        **model_params
    ).to(device)

    trainer = pl.Trainer(limit_train_batches=0.5)
    trainer.fit(model, train_loader, test_loader)
    log_net_visualization(model,torch.zeros(batch_size, input_cardinality))

if __name__ == "__main__":
    main()
