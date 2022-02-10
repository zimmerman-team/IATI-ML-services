import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import logging
import torch.multiprocessing as mp
import numpy as np

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path, os.path.join(path, 'dspn_annotated')]+sys.path
from models import generic_model, run, measurements as ms
import dspn
import dspn.model
import dspn.dspn
import dspn.utils
from models import diagnostics
from common import utils, config, chunking_dataset
from models import models_storage, generic_model


def my_hungarian_loss(predictions, targets, thread_pool):
    # predictions and targets shape :: (n, c, s)
    predictions, targets = dspn.utils.outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets.expand_as(predictions), reduction="none").mean(1)

    squared_error_np = squared_error.detach().cpu().numpy()
    indices = thread_pool.map(dspn.utils.hungarian_loss_per_sample, squared_error_np)
    losses = [
        sample[row_idx, col_idx].mean()
        for sample, (row_idx, col_idx) in zip(squared_error, indices)
    ]
    total_loss = torch.mean(torch.stack(list(losses)))
    return total_loss,indices

class MyFSEncoder(generic_model.AEModule):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        logging.debug("MyFSEncoder kwargs:"+str(kwargs))
        super().__init__(**kwargs)
        input_channels = kwargs['item_dim']
        output_channels = kwargs['latent_dim']
        dim = kwargs['layers_width']
        # here `dim` is hidden_dim in the caller,
        # and output_channels is latent_dim;
        # input_channels is set_channels
        # (see build_net(..))

        layers = [
            # 1-sized convolutions (Network-in-Network)
            # Conv1d's 1st and 2nd args are in_channels and out_channels
            # 3rd argument (set at value 1): kernel size
            nn.Conv1d(input_channels + 1, dim, 1),
            self.activation_function(),
        ]
        for i in self.depth_range():
            layers += [
                nn.Conv1d(dim, dim, 1),
                self.activation_function()
            ]
        layers.append( nn.Conv1d(dim, output_channels, 1) )
        self.conv = nn.Sequential( *layers )

        self.pool = dspn.model.FSPool(output_channels, 20, relaxed=False)

    def forward(self, x, mask=None):
        # in the caller, `x` is `current_set`
        mask = mask.unsqueeze(1)
        x = torch.cat([x, mask], dim=1)  # include mask as part of set
        x = self.conv(x) # output topology is probably set_size x latent_dim
        x = x / x.size(2)  # normalise so that activations aren't too high with big sets
        x, _ = self.pool(x)
        return x


class MyDSPN(nn.Module):
    """ Deep Set Prediction Networks
    Yan Zhang, Jonathon Hare, Adam Prügel-Bennett
    NeurIPS 2019
    https://arxiv.org/abs/1906.06565
    """

    def __init__(self, encoder, set_channels, max_set_size, channels, iters, lr, loss_fn):
        """
        encoder: Set encoder module that takes a set as input and returns a representation thereof.
            It should have a forward function that takes two arguments:
            - a set: FloatTensor of size (batch_size, input_channels, maximum_set_size). Each set
            should be padded to the same maximum size with 0s, even across batches.
            - a mask: FloatTensor of size (batch_size, maximum_set_size). This should take the value 1
            if the corresponding element is present and 0 if not.

        channels: Number of channels of the set to predict.

        max_set_size: Maximum size of the set.

        iter: Number of iterations to run the DSPN algorithm for.

        lr: Learning rate of inner gradient descent in DSPN.
        """
        super().__init__()
        self.encoder = encoder
        self.iters = iters
        self.lr = lr
        self.channels = channels
        self.loss_fn = loss_fn

        self.starting_set = nn.Parameter(torch.rand(1, set_channels, max_set_size))
        self.starting_mask = nn.Parameter(0.5 * torch.ones(1, max_set_size))

    def forward(self, target_repr):
        """
        Conceptually, DSPN simply turns the target_repr feature vector into a set.

        target_repr: Representation that the predicted set should match. FloatTensor of size (batch_size, repr_channels).
        Note that repr_channels can be different from self.channels.
        This can come from a set processed with the same encoder as self.encoder (auto-encoder), or a different
        input completely (normal supervised learning), such as an image encoded into a feature vector.
        """
        # copy same initial set over batch
        current_set = self.starting_set.expand(
            target_repr.size(0), *self.starting_set.size()[1:]
        )
        current_mask = self.starting_mask.expand(
            target_repr.size(0), self.starting_mask.size()[1]
        )
        # make sure mask is valid
        current_mask = current_mask.clamp(min=0, max=1)

        # info used for loss computation
        intermediate_sets = [current_set]
        intermediate_masks = [current_mask]
        # info used for debugging
        repr_losses = []
        grad_norms = []

        # optimise repr_loss for fixed number of steps
        for i in range(self.iters):
            # regardless of grad setting in train or eval, each iteration requires torch.autograd.grad to be used
            with torch.enable_grad():
                if not self.training:
                    current_set.requires_grad = True
                    current_mask.requires_grad = True

                # compute representation of current set
                predicted_repr = self.encoder(current_set, current_mask)
                # how well does the representation matches the target
                repr_loss = self.loss_fn(
                    predicted_repr, target_repr, reduction="mean"
                )
                # change to make to set and masks to improve the representation
                set_grad, mask_grad = torch.autograd.grad(
                    inputs=[current_set, current_mask],
                    outputs=repr_loss,
                    only_inputs=True,
                    create_graph=True,
                )
            # update set with gradient descent
            current_set = current_set - self.lr * set_grad
            current_mask = current_mask - self.lr * mask_grad
            current_mask = current_mask.clamp(min=0, max=1)
            # save some memory in eval mode
            if not self.training:
                current_set = current_set.detach()
                current_mask = current_mask.detach()
                repr_loss = repr_loss.detach()
                set_grad = set_grad.detach()
                mask_grad = mask_grad.detach()
            # keep track of intermediates
            intermediate_sets.append(current_set)
            intermediate_masks.append(current_mask)
            repr_losses.append(repr_loss)
            grad_norms.append(set_grad.norm())

        return intermediate_sets, intermediate_masks, repr_losses, grad_norms

class DSPNAE(generic_model.GenericModel):
    """
    DSPNAE is an acronym for Deep Set Prediction Network AutoEncoder
    """

    @classmethod
    def storage(cls):
        """
        :return: stored model retrieval system for DSPN AutoEncoders
        """
        return models_storage.DSPNAEModelsStorage()

    with_set_index = True
    losses = []
    class CollateFn(object):
        """
        CollateFn is being used to use the start-item-index
        and end-item-index to extract the set of contiguous items belonging to
        the same set - and this will be returned by the DataLoader.
        """

        def __init__(self, data):
            self.data = data

        def __call__(self, intervals):
            assert (len(intervals) == 1)
            interval = intervals[0]  # because it's a batch of size 1
            start_item_index, end_item_index = interval[0:2]

            # NOTE: set_data does not have the set_index column
            set_data = self.data[start_item_index:end_item_index]

            ret = torch.tensor(set_data)
            logging.debug(f"CollateFn.__call__ ret.shape {ret.shape}")
            return ret

    def make_train_loader(self, tsets):
        """
        For the deep sets it's important the the datapoints returned by the
        DataLoader are of items belonging to the same set.
        For this reason the input data to the dataloader cannot be a set of
        items, otherwise items belonging to different sets would end up
        being mixed-up.
        Instead, the input "datapoints" are just information about
        intervals of datapoints that belong to the same index.
        This information is returned by `tset.set_intervals(..)`
        Expanding on this: each input datapoint contains a start-item-index
        and an end-item-index of items in the actual original dataset (which
        is tsets.train_scaled, or tsets.test_scaled in make_test_loader(..)).
        Subsequently, CollateFn is being used to use the start-item-index
        and end-item-index to extract the set of contiguous items belonging to
        the same set - and this will be returned by the DataLoader.
        :param tsets: train/test dataset splits
        :return: the DataLoader
        """
        all_intervals = tsets.sets_intervals('train')
        # NOTE: shuffling is performed in the ChunkingDataset instead of the DataLoader
        #   because the ChunkingDataset is presented as iterable and cannot be indexed
        #   directly
        chunking_intervals = chunking_dataset.ChunkingDataset(
            all_intervals,
            shuffle=True,
            chunk_len=self.kwargs.get('epoch_chunk_len',1000),
            log_mlflow=True # do the mlflow logging for the training set
        )
        train_loader = torch.utils.data.DataLoader(
            chunking_intervals,
            shuffle=False,
            num_workers=config.data_loader_num_workers,
            pin_memory=False,
            collate_fn=self.CollateFn(tsets.train_scaled_without_set_index)
        )
        return train_loader

    def make_test_loader(self, tsets):
        """
        Please see description of DSPNAE.make_train_loader(..)
        :param tsets: train/test dataset splits
        :return: the DataLoader
        """
        all_intervals = tsets.sets_intervals('test')
        train_chunk_len = self.kwargs.get('epoch_chunk_len', 1000)
        test_chunk_len = int(float(train_chunk_len) * config.test_fraction)
        print("config.test_fraction",config.test_fraction)
        print("test_chunk_len",test_chunk_len)
        chunking_intervals = chunking_dataset.ChunkingDataset(
            all_intervals,
            shuffle=False,
            chunk_len=test_chunk_len,
            log_mlflow=False # don't do mlflow logging test/validation chunking
        )
        test_loader = torch.utils.data.DataLoader(
            chunking_intervals,
            shuffle=False,
            num_workers=config.data_loader_num_workers,
            collate_fn=self.CollateFn(tsets.test_scaled_without_set_index)
        )
        return test_loader

    def make_measurements(self):
        """
        Returns measurements for various metrics and values
        collected during training.
        :return:
        """
        ret = ms.MeasurementsCollection([
            ms.DatapointMeasurement(
                "z",
                dst=dict(
                    latent_last_epoch=ms.random_sampling
                )
            ),
            ms.BatchMeasurement('batch_loss', dst=dict(
                loss=ms.mean,
                len_losses=ms.len_
            ), mlflow_log=False),
            ms.BatchMeasurement('batch_mask_error', dst=dict(
                mask_error=ms.mean
            ), mlflow_log=False),
            ms.BatchMeasurement('batch_avg_gradn', dst=dict(
                avg_gradn=ms.mean
            ), mlflow_log=False),
            ms.BatchMeasurement('batch_repr_loss', dst=dict(
                avg_repr_loss=ms.mean
            ), mlflow_log=False),
            ms.EpochMeasurement("mask_error", mlflow_log=True),
            ms.EpochMeasurement("avg_gradn", mlflow_log=True),
            ms.EpochMeasurement("avg_repr_loss", mlflow_log=True),
            ms.EpochMeasurement("loss", plot_type='losses', mlflow_log=True),
            ms.EpochMeasurement("len_losses", mlflow_log=True),

            # tuple of (orig,reconstructed) sets, the first of every epoch
            # FIXME: maybe plot_type should be a lambda?
            ms.EpochMeasurement("target_and_reconstructed_set", plot_type='reconstructed_set'),
            ms.EpochMeasurement("reconstructed_set_avg",mlflow_log=True),
            ms.EpochMeasurement("reconstructed_set_std",mlflow_log=True),
            ms.EpochMeasurement("reconstructed_mask_avg",mlflow_log=True),
            ms.EpochMeasurement("reconstructed_mask_std",mlflow_log=True),

            ms.LastEpochMeasurement("latent_last_epoch", plot_type='latent'),
        ])
        return ret

    def __init__(self, **kwargs):
        """
        Construct the AutoEncoder model, composed by an encoder
        and a Deep Set Prediction Network decoder
        :param kwargs:
        """
        super().__init__(**kwargs)
        assert 'max_set_size' in self.kwargs, "must set max_set_size for this model"
        self.max_set_size = self.kwargs['max_set_size']
        self.hungarian_loss_thread_pool = mp.Pool(4)
        self.encoder = MyFSEncoder(**kwargs)

        dspn_loss_fn = getattr(torch.nn.functional,kwargs.get('dspn_loss_fn','smooth_l1')+"_loss")
        self.decoder = MyDSPN(
            self.encoder,
            kwargs['item_dim'],
            self.max_set_size,
            kwargs['item_dim'],
            kwargs['dspn_iter'],  # number of iteration on the decoder # FIXME: to conf
            float(kwargs['dspn_lr']),  # inner learning rate
            dspn_loss_fn
        )

    def _make_target(self, loaded_set):
        """
        Creates the training datapoint, composed by a set data component
        and a mask component, which determines which entries in the
        target_set tensor are set datapoints and which are padding.
        # FIXME: enable arbitrary batch_size, not only set as 1
        :param loaded_set:
        :return:
        """
        set_size = loaded_set.size(0)
        item_dims = loaded_set.size(1)
        # target_set dimensionality: (batch_size, item_dims, set_size)
        # here we assume a batch_size=1
        target_set = torch.zeros(1, item_dims, self.max_set_size)
        src = torch.swapaxes(loaded_set, 0, 1)
        src = src[:, 0:self.max_set_size]  # capping set size to max_set_size
        target_set[0, 0:item_dims, 0:set_size] = src

        # target_mask dimensionality: (batch_size, set_size)
        target_mask = torch.zeros(1, self.max_set_size)
        target_mask[0, 0:set_size] = 1
        return target_set, target_mask

    def forward(self, target_set, target_mask):
        """
        Forward computation of the autoencoding path:
        A latent code is generated by the encoder
        and then the decoder is reconstructing the original
        datapoint.
        :param target_set:
        :param target_mask:
        :return:
        """
        self.z = self.encoder(target_set, mask=target_mask)
        ret = self.decoder(self.z)
        intermediate_sets, intermediate_masks, repr_losses, grad_norms = ret

        # we take the last iteration of intermediate_sets
        self.reconstructed = intermediate_sets[-1]
        return ret

    def _step(self, batch, batch_idx, which_tset):
        """
        Processes a batch instance and produces the loss measure.
        :param batch:
        :param batch_idx:
        :param which_tset: 'train' or 'test'
        :return:
        """
        # utils.debug("batch.size",batch.size())
        # "batch" dimensionality: (set_size, item_dims)
        target_set,target_mask = self._make_target(batch)

        # copied from dspn.train.main.run()
        (progress, masks, repr_losses, gradn) = self(target_set,target_mask)
        #print("gradn",gradn)

        # utils.debug("target_set.shape",target_set.shape)
        # if using mask as feature, concat mask feature into progress
        target_set_with_mask = torch.cat(
            [target_set, target_mask.unsqueeze(dim=1)], dim=1
        )
        # utils.debug("target_set_with_mask.shape after cat with mask",target_set_with_mask.shape)

        unsqueezed_masks = [
            m.unsqueeze(dim=1)
            for m
            in masks
        ]
        progress_cat = [
            torch.cat([p, unsqueezed_mask], dim=1)
            for p, unsqueezed_mask in zip(progress, unsqueezed_masks)
        ]

        reconstructed_set = progress[-1]
        reconstructed_set_with_mask = progress_cat[-1]
        reconstructed_mask = masks[-1]

        #set_loss = dspn.utils.chamfer_loss(
        #    torch.stack(progress), target_set_with_mask.unsqueeze(0)
        #)

        total_loss,indices = my_hungarian_loss(
            reconstructed_set_with_mask,
            target_set_with_mask,
            thread_pool=self.hungarian_loss_thread_pool
        )
        set_loss = total_loss.unsqueeze(0)
        loss = set_loss.mean()
        # print("indices",indices)
        # for measurements:
        # self.log(f"{which_tset}_loss", loss)
        self.batch_loss = loss # setting instance variable to be collected by Measurements
        self.batch_mask_error = (unsqueezed_masks[-1]-target_mask).abs().mean()
        self.batch_avg_gradn = gradn[-1].abs().mean() #FIXME: maybe gradn is not clear, better grad_norm?
        self.batch_repr_loss = repr_losses[-1] #FIXME: maybe gradn is not clear, better grad_norm?
        if batch_idx == 0: # FIXME: this code seems quite refactorable
            self.reconstructed_set_avg = reconstructed_set.abs().mean().detach().numpy()
            self.reconstructed_set_std = reconstructed_set.std().detach().numpy()
            self.reconstructed_mask_avg = reconstructed_mask.abs().mean().detach().numpy()
            self.reconstructed_mask_std = reconstructed_mask.std().detach().numpy()
            self.target_and_reconstructed_set = (target_set,reconstructed_set)
            #import ipdb; ipdb.set_trace()

        #import ipdb; ipdb.set_trace()
        return loss

def main():
    """
    The default behavior for launching this script directly
    is the training of a DSPN AutoEncoder
    :return:
    """
    config_name = sys.argv[1]
    run.run(DSPNAE, config_name)


if __name__ == "__main__":
    main()
