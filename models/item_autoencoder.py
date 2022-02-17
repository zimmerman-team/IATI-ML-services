import torch
import numpy as np
import functools
import sys
import os

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path]+sys.path
from models import run,  generic_model
from common import utils
from models import measurements as ms
utils.set_np_printoptions()


class Model(generic_model.GenericModel):
    """
    Represents the Item-based AutoEncoder (baseline model)
    """

    with_set_index = False

    def make_train_loader(self, tsets):
        """
        Creates a DataLoader object (torch library) for the training set.
        :param tsets: the object containing the training and
            validation set data.
        :return: the data loader
        """
        train_loader = torch.utils.data.DataLoader(
            tsets.train_scaled,  # training set needs to be scaled
            batch_size=self.kwargs['batch_size'],  # working with batches
            shuffle=True,  # shuffles datapoints at every epoch
            num_workers=4,
            pin_memory=False
        )
        return train_loader

    def make_test_loader(self, tsets):
        """
        Creates a DataLoader object (torch library) for the validation set.
        :param tsets: the object containing the training and
            validation set data.
        :return: the data loader
        """
        test_loader = torch.utils.data.DataLoader(
            tsets.test_scaled,  # scaled validation data
            batch_size=self.kwargs['batch_size'],  # uses same batch size as training
            shuffle=False,  # no need to shuffle data when querying the model for validation
            num_workers=4
        )
        return test_loader

    def make_measurements(self):
        """
        defines all the metrics that need to be extracted from then model
        :return: a collection of Measurements (MeasurementsCollection), which are
            defining how the metrics are going to be calculated
        """
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
        """
        Creates the model. Instantiates the encoder and decoder parts.
        :param kwargs: mostly configuration. For example, depth and width
            of encoder and decoder
        """
        super().__init__(**kwargs)
        self.encoder = generic_model.Encoder(**kwargs)
        self.decoder = generic_model.Decoder(**kwargs)

    def forward(self, features):
        """
        Defines forward autoencoding path of the processing of the data.
        A code is being extracted from the input features.
        The decoder re-expands the reduced dimensions of the code
        into the original dimensionality of the datapoints.
        :param features: input data
        :return: the reconstructed input data
        """
        self.code = self.encoder.forward(features)
        self.reconstructed = self.decoder.forward(self.code)
        return self.reconstructed

    def _loss(self, batch, x_hats, z):
        """
        loss calculation
        :param batch: input data
        :param x_hats: reconstructions
        :param z: latent representations
        :return: the loss scalar
        """
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
            loss_fn = field.loss_function() \
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

        self.latent_l1_norm = self.kwargs.pop('latent_l1_norm', 0)

        # applies L1 norm to the latent codes in order to encourage sparsity
        loss += torch.norm(z, p=1)*self.latent_l1_norm

        self.losses = [curr.detach().numpy() for curr in losses]
        self.guess_correct = guess_correct
        return loss

    def _step(self, batch, batch_idx, which_tset):
        """
        Returns the quantity to be minimized, for every batch.
        :param batch: the data batch
        :param batch_idx: # FIXME: UNUSED? remove?
        :param which_tset: # either 'train' or 'val
        :return:
        """
        x_hat_divided, x_hat_glued = self._divide_or_glue(self(batch))
        z = self.encoder(batch)
        diff = batch - x_hat_glued
        mae = torch.mean(torch.abs(diff))
        mse = torch.mean(diff ** 2)
        loss = self._loss(batch, x_hat_divided, z)
        self.log(f"{which_tset}_loss", loss)
        self.log(f"{which_tset}_mae", mae)
        self.log(f"{which_tset}_mse", mse)

        # the following instance variables are eventually going
        # to be extracted by the Measurement system
        # that will ultimately lead to logged metrics on mlflow
        self.diff = diff.detach().numpy()
        self.x_hat = x_hat_glued.detach().numpy()
        self.z = z.detach().numpy()
        self.diff_reduced = np.mean(np.abs(self.diff), axis=0)

        # PyTorch Lightning will take care of minimization and parameter update
        return loss


def main():
    """
    The default behavior of running this script
    directly is training the ItemAE model.
    :return:
    """
    # FIXME: this part seems to be duplicated across all model modules
    config_name = sys.argv[1]
    run.run(Model, config_name)


if __name__ == "__main__":
    main()
