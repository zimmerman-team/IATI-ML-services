import torch
import logging
import math

from models import generic_model
from common import chunking_dataset, config, specs_config

class DeepSetGeneric(generic_model.GenericModel):
    with_set_index = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'max_set_size' in self.kwargs, "must set max_set_size for this model"
        self.max_set_size = self.kwargs['max_set_size']


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

    @classmethod
    def get_spec_from_model_config(cls, model_config):
        spec = specs_config.specs[model_config['spec_name']]
        return spec

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
        test_chunk_len = int(math.ceil(float(train_chunk_len) * config.test_fraction))
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

