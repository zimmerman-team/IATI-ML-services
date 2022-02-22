import sys
import functools
import os

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path]+sys.path

from models import deepset_generic, run, measurements as ms
multisetequivariance = __import__("multiset-equivariance")
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
        ret = ms.MeasurementsCollection([])
        return ret

    def make_net(self, conf):
        input_enc_kwargs = dict(
            d_latent=conf['latent_dim']
        )
        inner_obj_kwargs = dict(
            d_in=conf['item_dim'],
            d_hid=conf['layers_width'],
            d_latent=conf['latent_dim'],
            set_size=self.max_set_size,
            pool='fs',
            objective_type='mse_regularized'
        )
        dspn_kwargs = dict(
            learn_init_set=True,
            set_dim=conf['item_dim'],
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
        input, gt_output = x
        output = self.net(input)
        if isinstance(output, tuple):
            output, set_grad = output
        else:
            set_grad = None
        return output, gt_output, set_grad

    def _step(self, batch, batch_idx, which_tset):
        output, gt_output, set_grad = self(batch)
        loss = multisetequivariance.losses.hungarian_loss(output, gt_output, num_workers=4).mean(0)
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
