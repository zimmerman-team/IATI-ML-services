import torch
import torch.nn as nn
import sys
import os
import logging
import torch.multiprocessing as mp

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path, os.path.join(path, 'dspn_annotated')]+sys.path
from models import generic_model, run, measurements as ms
import dspn
import dspn.model
import dspn.dspn
from models import diagnostics
from common import utils, config, chunking_dataset
from models import models_storage, generic_model

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
            ret = torch.tensor(self.data[start_item_index:end_item_index])
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
            collate_fn=self.CollateFn(tsets.train_scaled)
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
            collate_fn=self.CollateFn(tsets.test_scaled)
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
            ms.EpochMeasurement("loss", plot_type='losses', mlflow_log=True),
            ms.EpochMeasurement("len_losses", mlflow_log=True),
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
        self.decoder = dspn.dspn.DSPN(
            self.encoder,
            kwargs['item_dim'],
            self.max_set_size,
            kwargs['item_dim'],
            kwargs['dspn_iter'],  # number of iteration on the decoder # FIXME: to conf
            kwargs['dspn_lr'],  # inner learning rate
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
        (progress, masks, evals, gradn) = self(target_set,target_mask)

        # utils.debug("target_set.shape",target_set.shape)
        # if using mask as feature, concat mask feature into progress
        target_set_with_mask = torch.cat(
            [target_set, target_mask.unsqueeze(dim=1)], dim=1
        )
        # utils.debug("target_set_with_mask.shape after cat with mask",target_set_with_mask.shape)
        progress = [
            torch.cat([p, m.unsqueeze(dim=1)], dim=1)
            for p, m in zip(progress, masks)
        ]

        #set_loss = dspn.utils.chamfer_loss(
        #    torch.stack(progress), target_set_with_mask.unsqueeze(0)
        #)

        set_loss = dspn.utils.hungarian_loss(
            progress[-1], target_set_with_mask, thread_pool=self.hungarian_loss_thread_pool
        ).unsqueeze(0)
        loss = set_loss.mean()

        # for measurements:
        # self.log(f"{which_tset}_loss", loss)
        self.batch_loss = loss # setting instance variable to be collected by Measurements
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
