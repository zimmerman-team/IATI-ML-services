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

class DeepSets(generic_model.GenericModel):

    with_set_index = True

    class CollateFn(object):
        def __init__(self, data):
            self.data = data

        def __call__(self, intervals):
            assert (len(intervals) == 1)
            interval = intervals[0]
            print("intervals", interval)
            ret = torch.tensor(self.data[interval[0]:interval[1]])
            print("ret.shape", ret.shape)
            return ret


    def make_train_loader(self, tsets):
        train_loader = torch.utils.data.DataLoader(
            tsets.sets_intervals('train'),
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            collate_fn=self.CollateFn(tsets.train_scaled)
        )
        return train_loader


    def make_test_loader(self, tsets):
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
    run.run(DeepSets, config_name)
