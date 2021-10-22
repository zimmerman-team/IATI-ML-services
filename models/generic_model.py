import pytorch_lightning as pl
import torch


class GenericModel(pl.LightningModule):

    DataLoader = None  # please set in subclass
    with_set_index = None  # please set in subclass

    def depth_range(self):
        return range(self.kwargs["depth"]-2)

    def __init__(self, **kwargs):
        self.rel = kwargs.pop('rel', None)
        self.kwargs = kwargs
        super().__init__()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=self.kwargs['weight_decay']
        )
        return optimizer

    def _divide(self, tensor):  # FIXME: maybe something more OO?
        if self.kwargs['divide_output_layer']:
            # actually do the division
            ret = self.rel.divide(tensor)
        else:
            # the "divided" output will just be a list with a single un-divided tensor
            ret = [tensor]
        return ret

    def _glue(self, tensor_list):
        return self.rel.glue(tensor_list)

    def _divide_or_glue(self, stuff):
        if type(stuff) is list:
            # stuff is already divided for various fields
            divided = stuff
            glued = self._glue(stuff)
        else:
            # stuff is already a glued-up tensor
            divided = self._divide(stuff)
            glued = stuff
        return divided, glued
