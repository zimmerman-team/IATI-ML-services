import sys
import functools
import os
import torch
import torch.nn as nn
import numpy as np

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path]+sys.path

from models import run,deepset_generic, generic_model, measurements as ms
multisetequivariance = __import__("multiset-equivariance")
#import multisetequivariance
#import ipdb; ipdb.set_trace()

def my_hungarian_loss(pred, target, num_workers=0):
    pdist = nn.functional.smooth_l1_loss(
        pred.unsqueeze(1).expand(-1, target.size(1), -1, -1),
        target.unsqueeze(2).expand(-1, -1, pred.size(1), -1),
        reduction='none').mean(3)

    pdist_ = pdist.detach().cpu().numpy()

    num_workers = min(pred.size(0), num_workers)
    if num_workers > 0:
        indices = multisetequivariance.losses.ray_lsa(pdist_, num_workers)
    else:
        indices = np.array([multisetequivariance.losses.linear_sum_assignment(p) for p in pdist_])
    mask_diff = pred[:,indices[:,0],-1] - target[:,indices[:,1],-1]
    batch_mask_error = mask_diff.abs().mean().detach().numpy()
    indices = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
    total_loss = losses.mean(1)

    return total_loss, batch_mask_error

class MyFSEncoder(generic_model.AEModule):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)

        input_channels = kwargs['item_dim']
        output_channels = kwargs['latent_dim']
        dim = kwargs['layers_width']

        layers = [
            nn.Linear(input_channels + 1, dim),
        ]
        for i in self.depth_range():
            layers += [
                nn.Linear(dim, dim),
                self.activation_function()
            ]

        layers.append( nn.Linear(dim, output_channels) )
        self.mlp = nn.Sequential( *layers )
        self.pool = multisetequivariance.models.FSPool(output_channels, self.kwargs['max_set_size'])

    def forward(self, x):
        x = self.mlp(x)
        x = self.pool(x)
        return x

class Model(deepset_generic.DeepSetGeneric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = self.make_net(kwargs)


    def make_measurements(self):
        """
        Returns measurements for various metrics and values
        collected during training.
        :return:
        """
        ret = ms.MeasurementsCollection([
            ms.BatchMeasurement('batch_loss', dst=dict(
                loss=ms.mean
            ), mlflow_log=False),
            ms.BatchMeasurement('batch_mask_error', dst=dict(
                mask_error=ms.mean
            ), mlflow_log=False),
            ms.BatchMeasurement('batch_avg_gradn', dst=dict(
                avg_gradn=ms.mean
            ), mlflow_log=False),
            ms.EpochMeasurement("mask_error", mlflow_log=True),
            ms.EpochMeasurement("avg_gradn", mlflow_log=True),
            ms.EpochMeasurement("loss", plot_type='losses', mlflow_log=True)
        ])
        return ret

    def make_net(self, conf):
        pool = "fs"
        input_enc_kwargs = dict(
            d_in=conf['item_dim'] + 1, # +1 because mask
            d_hid=conf['layers_width'],
            d_latent=conf['latent_dim'],
            set_size = self.max_set_size,
            pool=pool
        )
        inner_obj_kwargs = dict(
            d_in=conf['item_dim'] + 1, # +1 because mask
            d_hid=conf['layers_width'],
            d_latent=conf['latent_dim'],
            set_size=self.max_set_size,
            pool=pool,
            objective_type='mse_regularized'
        )
        dspn_kwargs = dict(
            learn_init_set=True,
            set_dim=conf['item_dim'] + 1 , # +1 because mask
            set_size=self.max_set_size,
            momentum=0.9,
            lr=conf['dspn_lr'],
            iters=conf['dspn_iter'],
            grad_clip=40,
            projection=None,
            implicit=True
        )

        net = multisetequivariance.models.DSPNBaseModel(
            input_enc_kwargs=input_enc_kwargs,
            inner_obj_kwargs=inner_obj_kwargs,
            dspn_kwargs=dspn_kwargs
        )
        net.input_to_z = MyFSEncoder(**self.kwargs)
        return net

    def forward(self, x):
        input = x
        output = self.net(input)
        if isinstance(output, tuple):
            output, set_grad = output
        else:
            set_grad = None
        return output, set_grad

    def _step(self, batch, batch_idx, which_split):
        #print("dimensionalities","batch",batch.shape)
        target_set,target_mask = self._make_target(batch)

        target_set_with_mask = torch.cat(
            [target_set, target_mask.unsqueeze(dim=1)], dim=1
        )
        target_set_with_mask_swapaxes = torch.swapaxes(target_set_with_mask,1,2)
        #print("dimensionalities","target_set",target_set.shape,"target_set_with_mask",target_set_with_mask.shape,"target_set_with_mask_swapaxes",target_set_with_mask_swapaxes.shape)
        #batch_unsqueezed = torch.unsqueeze(batch,2)
        #print("dimensionalities","batch_unsqueezed",batch_unsqueezed.shape)
        output, set_grad = self(target_set_with_mask_swapaxes)
        self.batch_avg_gradn = set_grad.norm()
        #print("dimensionalities","output",output.shape,"set_grad",set_grad.shape,"batch",batch.shape,"target_set_with_mask_swapaxes",target_set_with_mask_swapaxes.shape)
        loss, self.batch_mask_error = my_hungarian_loss(output, target_set_with_mask_swapaxes, num_workers=4)
        loss = loss.mean(0)
        self.batch_loss = loss # setting instance variable to be collected by Measurements

        return loss


def main():
    """
    The default behavior for launching this script directly
    is the training of a DSPN AutoEncoder
    :return:
    """
    config_name = sys.argv[1]
    run.run(Model, config_name)

if __name__ == "__main__":
    main()
