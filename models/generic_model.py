import pytorch_lightning as pl
import torch


class AEModule(torch.nn.Module):


    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.activation_function = getattr(torch.nn, self.kwargs["activation_function"])
        self.rel = kwargs.get('rel', None)
        super().__init__()


    def activate(self, x):
        return self.activation_function()(x)


    def depth_range(self):
        return range(self.kwargs["depth"]-2)

class Encoder(AEModule):


    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.encoder_input_layer = torch.nn.Linear(
            in_features=self.kwargs["input_shape"],
            out_features=self.kwargs["layers_width"]
        )

        for i in self.depth_range():
            setattr(self, f'encoder_hidden_layer_{i}',
                    torch.nn.Linear(
                        in_features=self.kwargs["layers_width"],
                        out_features=self.kwargs["layers_width"]
                    ))
        self.encoder_output_layer = torch.nn.Linear(
            in_features=self.kwargs["layers_width"],
            out_features=self.kwargs["bottleneck_width"]
        )


    def forward(self, features):
        activation = self.encoder_input_layer(features)
        activation = self.activate(activation)
        for i in self.depth_range():
            curr = getattr(self, f'encoder_hidden_layer_{i}')
            activation = curr(activation)
            activation = self.activate(activation)
        code = self.encoder_output_layer(activation)
        return code


class Decoder(AEModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder_input_layer = torch.nn.Linear(
            in_features=self.kwargs["bottleneck_width"],
            out_features=self.kwargs["layers_width"]
        )

        for i in self.depth_range():
            setattr(self, f'decoder_hidden_layer_{i}',
                    torch.nn.Linear(
                        in_features=self.kwargs["layers_width"],
                        out_features=self.kwargs["layers_width"]
                    ))
        if self.kwargs['divide_output_layer']:
            # instead of considering the output as a single homogeneous vector
            # its dimensionality is divided in many output layers, each belonging
            # to a specific field.
            # In this way, it's possible, for example, to apply a SoftMax activation
            # function to a categorical output section
            self.output_layers = [  # FIXME: smell
                dict(
                    layer=torch.nn.Linear(
                        in_features=self.kwargs["layers_width"],
                        out_features=field.n_features
                    ),
                    activation_function=(
                            field.output_activation_function or torch.nn.Identity()
                    )
                )
                for field
                in self.rel.fields
            ]
        else:
            self.output_layers = [
                dict(
                    layer=torch.nn.Linear(
                        in_features=self.kwargs["layers_width"],
                        out_features=self.kwargs["input_shape"]
                    ),
                    activation_function=torch.nn.Identity()
                )
            ]


    def forward(self, code):
        activation = self.decoder_input_layer(code)
        activation = self.activate(activation)
        for i in self.depth_range():
            curr = getattr(self, f'decoder_hidden_layer_{i}')
            activation = curr(activation)
            activation = self.activate(activation)
        reconstructed = []
        for curr in self.output_layers:
            activation_out = curr["layer"](activation)
            activation_out = self.activate(activation_out)
            reconstructed.append(activation_out)
        reconstructed = self._glue(reconstructed)
        return reconstructed


    def _glue(self, tensor_list):
        return self.rel.glue(tensor_list)


class GenericModel(pl.LightningModule):

    with_set_index = None  # please set in subclass


    def make_train_loader(self,tsets):
        raise Exception("implement in subclass")


    def make_test_loader(self, tsets):
        raise Exception("implement in subclass")


    def __init__(self, **kwargs):
        self.rel = kwargs.get('rel', None)
        self.kwargs = kwargs
        super().__init__()


    def configure_optimizers(self):
        """
        method required by Pytorch Lightning
        :return: the optimizer function used to minimize the loss function
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=self.kwargs['weight_decay']
        )
        return optimizer


    def _divide(self, tensor):  # FIXME: maybe something more OO?
        """
        Given a glued-up tensor of data/features, splits it, by column groups,
        into a list of tensors each one belonging to a field
        :param tensor: the data/features tensor
        :return:
        """
        if self.kwargs['divide_output_layer']:
            # actually do the division
            ret = self.rel.divide(tensor)
        else:
            # the "divided" output will just be a list with a single un-divided tensor
            ret = [tensor]
        return ret


    def _divide_or_glue(self, stuff):
        """
        Either splits an input tensor in chunked tensors by column groups
        each belonging to a field; or glues a list of tensors in the column
        dimension.
        :param stuff: either a tensor or a list of tensors
        :return: the divided tensor list *and* the glued-up tensor
        """
        if type(stuff) is list:
            # stuff is already divided for various fields
            divided = stuff
            glued = self._glue(stuff)
        else:
            # stuff is already a glued-up tensor
            divided = self._divide(stuff)
            glued = stuff
        return divided, glued
