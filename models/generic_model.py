import pytorch_lightning as pl
import torch
import os
import sys
import logging
import inspect
import numpy as np

project_root_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path.insert(0, project_root_dir)

from common import config, timer
from models import models_storage
from functools import cached_property

class AEModule(torch.nn.Module):
    """
    Superclass of both Encoder and Decoder, which are very similar components.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.activation_function = getattr(torch.nn, self.kwargs["activation_function"])
        # FIXME: maybe this requires multiple inheritance
        self.spec = kwargs.get('spec', None)
        super().__init__()

    def activate(self, x):
        return self.activation_function()(x)

    def depth_range(self):
        return range(self.kwargs["depth"]-2)


class Encoder(AEModule):
    """
    An encoder takes a datapoint and produces a compressed
    low-dimensional latent code.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_input_layer = torch.nn.Linear(
            in_features=self.kwargs["item_dim"],
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
            out_features=self.kwargs["latent_dim"]
        )

    def forward(self, features):
        """
        Specification of the computation path of the encoder
        :param features: the datapoint(s)
        :return: the compressed low-dimensional latent code
        """
        activation = self.encoder_input_layer(features)
        activation = self.activate(activation)
        for i in self.depth_range():
            curr = getattr(self, f'encoder_hidden_layer_{i}')
            activation = curr(activation)
            activation = self.activate(activation)
        code = self.encoder_output_layer(activation)
        return code


class Decoder(AEModule):
    """
    A decoder expands a compressed low-dimensional latent code
    into the original datapoint.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder_input_layer = torch.nn.Linear(
            in_features=self.kwargs["latent_dim"],
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
                in self.spec.fields
            ]
        else:
            self.output_layers = [
                dict(
                    layer=torch.nn.Linear(
                        in_features=self.kwargs["layers_width"],
                        out_features=self.kwargs["item_dim"]
                    ),
                    activation_function=torch.nn.Identity()
                )
            ]

    def forward(self, code):
        """
        :param code: the compressed low-dimensional latent code
        :return: the reconstructed datapoint
        """
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
        return self.spec.glue(tensor_list)


class GenericModel(pl.LightningModule):
    """
    Superclass of ItemAE and DSPNAE
    """

    with_set_index = None  # please set in subclass
    _timer = timer.Timer()

    @classmethod
    def get_spec_from_model_config(cls, model_config):
        """
        this class method is called because according to the type of model
        it will require different kind of data.
        :param model_config:
        :return:
        """
        raise Exception("classmethod `get_spec_from_model_config` needs to be implemented in subclass")

    @property
    def source_filename(self):
        try:
            filename = inspect.getfile(self.__class__)
        except TypeError:
            # in case of the obscure *** TypeError: <class '__main__.Model'> is a built-in class
            # presumably raised when running directly the model module via command-line python
            filename = sys.argv[0]
        return filename

    @property
    def modulename(self):
        """
        Name of the module in which the concrete class is located
        :return:
        """
        ret = inspect.getmodulename(self.source_filename)
        return ret

    @property
    def classname(self):
        """
        Returns the specific class of the model
        :return:
        """
        return self.__class__.__name__

    @property
    def name(self):
        """
        The name of a model is composed by the name of
        the algorithm and the name of the source of data
        :return:
        """

        # double underscores are useful to distinguish the modulename part from the spec.name part
        # as those strings can themselves contain single underscores
        ret = f"{self.modulename}__{self.spec.name}"
        if 'model_name_suffix' in self.kwargs:
            ret += f"_{self.kwargs['model_name_suffix']}"
        return ret

    def make_train_loader(self, splits):
        """
        System to provide batches of datapoints for training
        :param splits:
        :return:
        """
        raise Exception("implement in subclass")

    def make_test_loader(self, splits):
        """
        System to provide batches of datapoints for testing
        :param splits:
        :return:
        """
        raise Exception("implement in subclass")

    def __init__(self, **kwargs):
        """
        Constructor of the model.
        the construction parameters will be stored in the 'kwargs'
        instance variable for easy retrieval.
        The model is also built on a specific relation data source
        ('spec' parameter)
        :param kwargs:
        """
        self.spec = kwargs.get('spec', None)
        self.kwargs = kwargs # model config (hyper)parameters typically end up here
        logging.debug("generic_model kwargs: "+str(kwargs))

        # stored model retrieval system
        self.storage = models_storage.ModelsStorage(self.modulename)

        super().__init__()

    def configure_optimizers(self):
        """
        method required by Pytorch Lightning
        :return: the optimizer function used to minimize the loss function
        """
        optimizer_class = getattr(torch.optim, self.kwargs.get('optimizer','Adam'))
        optimizer = optimizer_class(
            self.parameters(),
            lr=float(self.kwargs.get('learning_rate',1e-3)),
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
            ret = self.spec.divide(tensor)
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

    def training_step(self, batch, batch_idx):
        """
        processes a batch instance for training purposes
        :param batch:
        :param batch_idx:
        :return:
        """

        logging.debug(f"training_step batch.shape {batch.shape}")
        elapsed_time = self._timer.elapsed_time
        if elapsed_time > config.log_step_elapsed_time:
            logging.debug(f"training_step batch_idx {batch_idx} elapsed_time {elapsed_time}")
            self._timer.reset()
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        """
        processes a batch instance for validation
        :param batch:
        :param batch_idx:
        :return:
        """

        elapsed_time = self._timer.elapsed_time
        if elapsed_time > config.log_step_elapsed_time:
            logging.debug(f"validation_step batch_idx {batch_idx} elapsed_time {elapsed_time}")
            self._timer.reset()
        return self._step(batch, batch_idx, 'val')

    @property
    def default_z_npa_for_missing_inputs(self):
        """
        This is the fallback value that z is assumed to have if the network is not queried, as,
        for example the input is an empty set.
        :return: the default fallback value for z
        """
        # FIXME: batch size?
        ret = np.zeros((1, self.kwargs['latent_dim']))
        return ret
