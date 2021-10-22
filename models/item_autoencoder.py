import torch
import numpy as np
import functools
import sys
import os

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path]+sys.path
from models import diagnostics, run, measurements as ms, generic_model
from common import utils, relspecs, persistency
utils.set_np_printoptions()


class ItemAE(generic_model.GenericModel):

    DataLoader = torch.utils.data.DataLoader
    with_set_index = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_input_layer = torch.nn.Linear(
            in_features=kwargs["input_shape"],
            out_features=kwargs["layers_width"]
        )

        for i in self.depth_range():
            setattr(self, f'encoder_hidden_layer_{i}',
                    torch.nn.Linear(
                        in_features=kwargs["layers_width"],
                        out_features=kwargs["layers_width"]
                    ))
        self.encoder_output_layer = torch.nn.Linear(
            in_features=kwargs["layers_width"],
            out_features=kwargs["bottleneck_width"]
        )
        self.decoder_input_layer = torch.nn.Linear(
            in_features=kwargs["bottleneck_width"],
            out_features=kwargs["layers_width"]
        )

        for i in self.depth_range():
            setattr(self, f'decoder_hidden_layer_{i}',
                    torch.nn.Linear(
                        in_features=kwargs["layers_width"],
                        out_features=kwargs["layers_width"]
                    ))
        self.activation_function = getattr(torch.nn, kwargs["activation_function"])
        if kwargs['divide_output_layer']:
            # instead of considering the output as a single homogeneous vector
            # its dimensionality is divided in many output layers, each belonging
            # to a specific field.
            # In this way, it's possible, for example, to apply a SoftMax activation
            # function to a categorical output section
            self.decoder_output_layers = [  # FIXME: smell
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

    def encoder(self, features):
        activation = self.encoder_input_layer(features)
        activation = self.activation_function()(activation)
        for i in self.depth_range():
            curr = getattr(self, f'encoder_hidden_layer_{i}')
            activation = curr(activation)
            activation = self.activation_function()(activation)
        code = self.encoder_output_layer(activation)
        return code

    def decoder(self, code):
        activation = self.decoder_input_layer(code)
        activation = self.activation_function()(activation)
        for i in self.depth_range():
            curr = getattr(self, f'decoder_hidden_layer_{i}')
            activation = curr(activation)
            activation = self.activation_function()(activation)
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

    def _loss(self,batch,x_hats,z):
        losses = []
        guess_correct = []
        batch_divided = self._divide(batch)

        # FIXME: debug with run_config['divided_output_layer'] = False
        for curr, curr_x_hat, batch_div, field in zip(
                self.decoder_output_layers,
                x_hats,
                batch_divided,
                self.rel.fields
        ):
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

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')


if __name__ == "__main__":
    config_name = sys.argv[1]
    run.run(ItemAE, config_name)
