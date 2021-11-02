import torch
import sys
import os
from typing import Union

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path]+sys.path
from models import generic_model, run, measurements as ms

class InvariantModel(torch.nn.Module):
    def __init__(self, phi, rho):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        items_repr = self.phi.forward(x)
        set_repr = torch.sum(items_repr, dim=0, keepdim=True)
        out = self.rho.forward(set_repr)
        return out

class DeepSetsAutoencoder(generic_model.GenericModel):

    with_set_index = True

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
            interval = intervals[0] # because it's a batch of size 1
            print("interval", interval)
            start_item_index,end_item_index = interval[0:2]
            ret = torch.tensor(self.data[start_item_index:end_item_index])
            print("ret.shape", ret.shape)
            return ret


    def make_train_loader(self, tsets):
        """
        For the deep sets it's important the the datapoints returned by the
        DataLoader are of items belonging to the same set.
        For this reason the input data to the dataloader cannot be a set of
        items, otherwise items belonging to different sets would end up
        being mixed-up.
        Instead, the input "datapoints" are just information about
        intervals of datapoints that belong to the same indexIn other .
        This information is returned by `tset.set_intervals(..)`
        Expanding on this: each input datapoint contain a start-item-index
        and an end-item-index of items in the actual original dataset (which
        is tsets.train_scaled, or tsets.test_scaled in make_test_loader(..)).
        Subsequently, CollateFn is being used to use the start-item-index
        and end-item-index to extract the set of contiguous items belonging to
        the same set - and this will be returned by the DataLoader.
        :param tsets: train/test dataset splits
        :return:
        """
        train_loader = torch.utils.data.DataLoader(
            tsets.sets_intervals('train'),
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            collate_fn=self.CollateFn(tsets.train_scaled)
        )
        return train_loader


    def make_test_loader(self, tsets):
        """
        Please see description of DeepSetsAutoencoder.make_train_loader(..)
        :param tsets: train/test dataset splits
        :return:
        """
        test_loader = torch.utils.data.DataLoader(
            tsets.sets_intervals('test'),
            shuffle=False,
            num_workers=4,
            collate_fn=self.CollateFn(tsets.test_scaled)
        )
        return test_loader


    def make_measurements(self):
        ret = ms.MeasurementsCollection([
            ])
        return ret


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.la = torch.nn.Linear(
            in_features=kwargs['input_shape'],
            out_features=1
        )

    def forward(self, features):
        return self.la(features)


    def training_step(self, batch, batch_idx):
        print("training_set",batch.shape,batch_idx)
        return 1


    def validation_step(self, batch, batch_idx):
        return 1


if __name__ == "__main__":
    config_name = sys.argv[1]
    run.run(DeepSetsAutoencoder, config_name)
