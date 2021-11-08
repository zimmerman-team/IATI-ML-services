import torch
import sys
import os
from typing import Union

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path,os.path.join(path,'dspn_annotated')]+sys.path
from models import generic_model, run, measurements as ms
import dspn
import dspn.model
import dspn.dspn

class InvariantModel(torch.nn.Module): #FIXME: delete?
    def __init__(self, phi, rho):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        items_repr = self.phi.forward(x)
        set_repr = torch.sum(items_repr, dim=0, keepdim=True)
        out = self.rho.forward(set_repr)
        return out

class DSPNAE(generic_model.GenericModel):
    """
    DSPNAE is an acronym for Deep Set Prediction Network AutoEncoder
    """

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
            start_item_index,end_item_index = interval[0:2]
            ret = torch.tensor(self.data[start_item_index:end_item_index])
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
        :return: the DataLoader
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
        :return: the DataLoader
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
        assert 'max_set_size' in self.kwargs, "must set max_set_size for this model"
        self.max_set_size = self.kwargs['max_set_size']
        self.encoder = dspn.model.FSEncoder(
            kwargs['item_dim'],
            kwargs['latent_dim'],
            kwargs['layers_width']
        )
        self.decoder = dspn.dspn.DSPN(
            self.encoder,
            kwargs['item_dim'],
            self.max_set_size,
            kwargs['item_dim'],
            10, # number of iteration on the decoder # FIXME: to conf
            800, # inner learning rate # FIXME: to conf
        )

    def _make_target(self,loaded_set):
        set_size = loaded_set.size(0)
        item_dims = loaded_set.size(1)
        # target_set dimensionality: (batch_size, item_dims, set_size)
        # here we assume a batch_size=1
        target_set = torch.zeros(1,item_dims,self.max_set_size)
        src = torch.swapaxes(loaded_set,0,1)
        src = src[:,0:self.max_set_size] # capping set size to max_set_size
        target_set[0,0:item_dims,0:set_size] = src

        # target_mask dimensionality: (batch_size, set_size)
        target_mask = torch.zeros(1,self.max_set_size)
        target_mask[0, 0:set_size] = 1
        return target_set,target_mask

    def forward(self, loaded_set):
        # loaded_set dimensionality: (set_size, item_dims)

        target_set,target_mask = self._make_target(loaded_set)

        self.code = self.encoder(target_set, target_mask)
        ret = self.decoder(self.code)
        intermediate_sets, intermediate_masks, repr_losses, grad_norms = ret
        self.reconstructed = intermediate_sets[-1]
        return ret

    def _step(self, batch, batch_idx, which_tset):
        # copied from dspn.train.main.run()
        tmp = self( batch )
        (progress, masks, evals, gradn) = tmp

        target_set,target_mask = self._make_target(batch)
        # if using mask as feature, concat mask feature into progress
        target_set = torch.cat(
            [target_set, target_mask.unsqueeze(dim=1)], dim=1
        )
        progress = [
            torch.cat([p, m.unsqueeze(dim=1)], dim=1)
            for p, m in zip(progress, masks)
        ]

        set_loss = dspn.utils.chamfer_loss(
            torch.stack(progress), target_set.unsqueeze(0)
        )

        return set_loss.mean()

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')


if __name__ == "__main__":
    config_name = sys.argv[1]
    run.run(DSPNAE, config_name)
