import torch
import numpy as np
import functools
import sys
import os

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path]+sys.path
from models import diagnostics, run, measurements as ms, generic_model
from common import utils, relspecs, persistency
from models import measurements as ms
utils.set_np_printoptions()


class ItemAE(generic_model.GenericModel):

    with_set_index = False

    def make_train_loader(self, tsets):
        train_loader = torch.utils.data.DataLoader(
            tsets.train_scaled,
            batch_size=self.kwargs['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=False
        )
        return train_loader

    def make_test_loader(self, tsets):
        test_loader = torch.utils.data.DataLoader(
            tsets.test_scaled,
            batch_size=self.kwargs['batch_size'],
            shuffle=False,
            num_workers=4
        )
        return test_loader

    def make_measurements(self):
        ret = ms.MeasurementsCollection([
            ms.DatapointMeasurement("x_hat", dst=dict(
                output_mean_per_feature=ms.mean,
                output_var_per_feature=ms.var,
                output_last_epoch=ms.random_sampling
            )),
            ms.DatapointMeasurement("z", dst=dict(
                latent_last_epoch=ms.random_sampling
            )),

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

            ms.EpochMeasurement("output_mean_per_feature", plot_type='fields'),
            ms.EpochMeasurement("output_var_per_feature", plot_type='fields'),
            ms.EpochMeasurement("mae_per_feature", plot_type='fields'),
            ms.EpochMeasurement("mean_losses", plot_type='losses'),
            ms.EpochMeasurement("mean_guess_correct", plot_type='losses'),
            ms.EpochMeasurement("mean_latent_l1_norm", plot_type='losses', mlflow_log=True),

            ms.LastEpochMeasurement("output_last_epoch", plot_type='fields'),
            ms.LastEpochMeasurement("latent_last_epoch", plot_type='latent'),

        ])
        return ret

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = generic_model.Encoder(**kwargs)
        self.decoder = generic_model.Decoder(**kwargs)


    def forward(self, features):
        self.code = self.encoder.forward(features)
        self.reconstructed = self.decoder.forward(self.code)
        return self.reconstructed

    def _loss(self,batch,x_hats,z):
        losses = []
        guess_correct = []
        batch_divided = self._divide(batch)

        # FIXME: debug with run_config['divided_output_layer'] = False
        for curr, curr_x_hat, batch_div, field in zip(
                self.decoder.output_layers,
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
