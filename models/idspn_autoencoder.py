import sys
import functools
import os
import torch

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path]+sys.path

from models import deepset_generic, run, measurements as ms
multisetequivariance = __import__("multiset-equivariance")
#import multisetequivariance
#import ipdb; ipdb.set_trace()
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
            ms.EpochMeasurement("loss", plot_type='losses', mlflow_log=True)
        ])
        return ret

    def make_net(self, conf):
        self.batch_loss = loss # setting instance variable to be collected by Measurements
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
        return net

    def forward(self, x):


        input = x
        output = self.net(input)
        if isinstance(output, tuple):
            output, set_grad = output
        else:
            set_grad = None
        return output, set_grad

    def _step(self, batch, batch_idx, which_tset):
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
        #print("dimensionalities","output",output.shape,"set_grad",set_grad.shape,"batch",batch.shape,"target_set_with_mask_swapaxes",target_set_with_mask_swapaxes.shape)
        loss = multisetequivariance.losses.hungarian_loss(output, target_set_with_mask_swapaxes, num_workers=4).mean(0)

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
