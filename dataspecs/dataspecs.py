import abc
import datetime
import logging
import re

import numpy as np
import sklearn.preprocessing
import torch.nn

import niftycollection.collection
from common import utils
from common.relspecs_classes import get_codelists
from configurator import config
import text_model


class SpecsCollection(niftycollection.collection.Collection):

    @property
    def downloadable(self):
        """
        returns the specifications, but only
        those that are marked as downloadable
        :return:
        """
        return [
            spec
            for spec
            in self
            if spec.download is True
        ]

    @property
    def downloadable_prefixed_fields_names(self):
        """
        The fields returned by IATI.cloud have a prefix that is
        the relation name.
        This function construct and retrurns those prefixed field names.
        :return:
        """
        ret = [
            curr
            for spec
            in self.downloadable
            for curr
            in spec.prefixed_fields_names
        ]
        return ret


class Spec(object):
    """
    Inherited by Rel and Activity.
    Represents the specification (field list and types)
    of a data source.
    """
    def __init__(self, name, fields, download=False, limit=None):
        """
        :param name: name of the spec
        :param fields: list of fields objects
        :param download: boolean - is this spec going to be downloaded?
        :param limit: None or integer - should the amount of records be
                      limited to a certain amount?
        """
        self.name = name
        self.fields = fields
        self.download = download
        self.limit = limit

    def __str__(self):
        """
        compact string representation including fields
        :return:
        """
        return f"<S:{self.name}: {self.fields}>"

    def divide(self, tensor, with_set_index=False):
        """
        given a glued-up tensor, it returns the list of
        the columns, each belonging to each of the fields.
        :param tensor:
        :param with_set_index:
        :return:
        """
        ret = []
        for start, end in self.fields_intervals(with_set_index=with_set_index):
            ret.append(tensor[:, start:end])
        if with_set_index:
            ret[0] = ret[0].astype(np.int32)
        return ret

    def glue(self, tensor_list):
        # FIXME: not sure if this implementation is correct
        # also, there is a _glue in GenericModel. What's the deal with that?
        if type(tensor_list[0]) is np.ndarray:
            # use numpy
            ret = np.hstack(tensor_list)
        else:
            # pytorch
            ret = torch.hstack(tensor_list)
        return ret

    @property
    def extract_field_regex(self):
        """
        regular expression to extract a field name from the
        complete field name from IATI.cloud
        :return:
        """
        raise Exception("implement in subclass")

    def extract_from_field_data(self, v):
        raise Exception("implement in subclass")

    def extract_from_raw_data(self, activity_data):
        """
        given a dictionary with the activity data as returned by IATI.cloud,
        extract the values of all fields that are in the specification.
        :param activity_data:
        :return:
        """
        ret = {}
        for k, v in activity_data.items():
            m = re.match(self.extract_field_regex, k)
            if m is not None:
                rel_field = m.group(1)
                if rel_field in self.fields_names:
                    # logging.info(f"considering field {rel_field}")
                    ret[rel_field] = self.extract_from_field_data(v)
        return ret

    @property
    def scalers(self):
        """
        returns the scalers (normalizers for example) belonging to
        each field
        :return:
        """
        return [curr.scaler for curr in self.fields]

    @property
    def n_fields(self):
        """
        returns the number of fields in this specification.
        :return:
        """
        return len(self.fields)

    @property
    def n_features(self):
        """
        returns the number of individual features (as in ML)
        belonging to this specification.
        :return:
        """
        return sum([curr.n_features for curr in self.fields])

    @property
    def fields_names(self):
        """
        returns the field names of this specification.
        :return:
        """
        return [f.name for f in self.fields]


    def fields_intervals(self, with_set_index=False):
        """
        returns the intervals (start and end indexes)
        for each of the field, aimed at extracting
        a field column from a glued-up tensor.
        :param with_set_index:
        :return:
        """
        start = 0
        intervals = []
        if with_set_index:
            intervals.append((0, 1))
            start = 1
        for field in self.fields:
            end = start + field.n_features
            intervals.append((start, end))
            start = end
        return intervals

    @property
    def codelists_names(self):
        """
        returns all the codelist names extracted
        from the category fields.
        :return:
        """
        ret = []
        for field in self.fields:
            if type(field) is CategoryField:
                ret.append(field.codelist_name)
        return ret


class Rel(Spec):
    """
    relation field that holds a set of items.
    """

    @property
    def prefixed_fields_names(self):
        """
        returns the complete field names as IATI.cloud's standard
        :return:
        """
        return [
            self.name+"_"+curr
            for curr
            in self.fields_names
        ]

    @property
    def extract_field_regex(self):
        """
        regex to extract the field name within this rel
        :return:
        """
        return f'{self.name}_(.*)'

    def extract_from_field_data(self, v):
        # cap the amount of items to config.download_max_set_size
        v = v[:config.download_max_set_size]
        return v


class AbstractField(abc.ABC):
    def __init__(
            self,
            name,
            output_activation_function=None,
            loss_function=(lambda : None)
    ):
        self.name = name
        self._output_activation_function = output_activation_function

        # WARNING: loss_function needs to be a factory function!
        self._loss_function = loss_function

        self._scaler = None

    def __str__(self):
        """
        compact string representation
        :return:
        """
        return f"<F:{self.name} n_features:{self.n_features}>"

    def __repr__(self):
        """
        compact string representation
        :return:
        """
        return str(self)

    @property
    def n_features(self):
        """
        number of features of this field
        :return:
        """
        raise Exception("not implemented")

    @property
    def output_activation_function(self):
        """
        in the decoder, which activation function to use
        for this field
        :return:
        """
        return self._output_activation_function

    @property
    def loss_function(self):
        """
        the loss function specific for this field.
        Needs to be instantiated by calling it with ()!
        :return:
        """
        return self._loss_function

    def guess_correct(self, x_hat, x_orig):
        """
        fraction of correct field value reconstruction guesses
        :param x_hat:
        :param x_orig:
        :return:
        """
        raise Exception("not implemented")

    @property
    def scaler(self):
        """
        Field value scaler (for example, normalization)
        :return:
        """
        return self._scaler

    def make_scaler(self):
        """
        Constructor for the field value scaler.
        The abstract class provides a default.
        :return:
        """
        # default scaler
        return sklearn.preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))

    def make_and_fit_scaler(self, data):
        """
        creates a scaler fitted from some data.
        :param data:
        :return:
        """
        self._scaler = self.make_scaler()
        self._scaler.fit(data)
        return self._scaler


class PositionField(AbstractField):

    def encode(self, entries, set_size, **kwargs):
        ret = []
        for entry in entries:
            groups = re.match('\s*\(\s*(.*)\s*,\s*(.*)\s*\)\s*', entry).groups()
            lat, lon = tuple(groups)
            t = (float(lat), float(lon))
            ret.append(t)

        # FIXME: code duplication from other fields
        short_of = set_size - len(entries)
        if short_of > 0:
            for _ in range(short_of):
                tmp = [0] * self.n_features
                ret.append(tmp)
        return ret

    def guess_correct(self, x_hat, x):
        # FIXME: this was copied from DatetimeField, check that it matches PositionField purpose
        # using only the first 3 values of the timetuple as they refer to Y/M/D
        x_hat_descaled = self.scaler.inverse_transform(x_hat)
        x_descaled = self.scaler.inverse_transform(x)

        # FIXME: justify why norm < 0.1
        correct_ones = np.linalg.norm(x_hat_descaled - x_descaled, axis=1) < 0.1
        correct_ratio = np.mean(correct_ones)
        return correct_ratio

    @property
    def n_features(self):
        return 2 # latitude and longitude


class DatetimeField(AbstractField):

    def encode(self, entries, set_size, **kwargs):
        ret = []
        for entry in entries:
            entry = re.match('(.*)Z', entry).groups()[0]  # python's datetime does not parse the final 'Z'
            dt = datetime.datetime.fromisoformat(entry)
            t = tuple(dt.timetuple())
            ret.append(t)
        short_of = set_size - len(entries)
        if short_of > 0:
            for i in range(short_of):
                tmp = [0] * self.n_features
                ret.append(tmp)
        return ret

    def guess_correct(self, x_hat, x):
        # using only the first 3 values of the timetuple as they refer to Y/M/D
        x_hat_descaled = self.scaler.inverse_transform(x_hat)[:, :3]
        x_descaled = self.scaler.inverse_transform(x)[:, :3]

        # FIXME: justify why norm < 0.5
        correct_ones = np.linalg.norm(x_hat_descaled - x_descaled, axis=1) < 0.5
        correct_ratio = np.mean(correct_ones)
        return correct_ratio

    @property
    def n_features(self):
        return 9  # 9 is the cardinality of the timetuple


class CategoryField(AbstractField):
    def __init__(self, name, codelist_name, **kwargs):
        if 'output_activation_function' not in kwargs:
            kwargs['output_activation_function'] = torch.nn.Softmax(dim=1)

        prevent_constant_prediction = kwargs.pop('prevent_constant_prediction', None)

        self.codelist_name = codelist_name

        if 'loss_function' not in kwargs:
            def _loss_function():
                if prevent_constant_prediction:
                    prevent_constant_prediction_idx = self.codelist.index(prevent_constant_prediction)
                    siz = len(self.codelist)
                    weight = np.ones(siz)/siz
                    weight[prevent_constant_prediction_idx] /= 10
                else:
                    weight = None
                return utils.OneHotCrossEntropyLoss(weight=weight)
            kwargs['loss_function'] = _loss_function

        super().__init__(name, **kwargs)

    @property
    def codelist(self):
        ret = get_codelists()[self.codelist_name]
        return ret

    def encode(self, entries, set_size, **kwargs):
        ret = np.zeros((set_size, len(self.codelist)))
        for index_code, code in enumerate(entries):
            if code is None:
                logging.warning("code is None: this shouldn't happen")
                continue
            elif code not in self.codelist:
                # FIXME: this is way too common: logging.warning(f"code '{code}' not found
                #  in the codelist {self.codelist}")
                pass
            else:
                index_one = self.codelist.index(code)
                ret[index_code, index_one] = 1
        short_of = set_size - len(entries)
        if short_of > 0:
            for i in range(short_of):
                avg = 1.0 / float(len(self.codelist))
                ret[set_size - 1 - i, :] = avg

        ret = ret.tolist()
        return ret

    def guess_correct(self, x_hat, x):
        x_hat_descaled = self.scaler.inverse_transform(x_hat)
        x_descaled = self.scaler.inverse_transform(x)
        n_correct = 0.0
        for curr_x_hat, curr_x in zip(x_hat_descaled, x_descaled):
            n_correct += float(np.argmax(curr_x_hat) == np.argmax(curr_x))

        correct_ratio = n_correct/x_hat_descaled.shape[0]
        return correct_ratio

    @property
    def n_features(self):
        return len(self.codelist)

    def make_scaler(self):
        return utils.IdentityTransformer()


class NumericalField(AbstractField):
    def encode(self, entries, set_size, **kwargs):
        ret = [float(x) for x in entries]
        short_of = set_size - len(entries)
        if short_of > 0:
            for i in range(short_of):
                ret.append(0.0)
        return ret

    @property
    def n_features(self):
        return 1  # it's just one

    def guess_correct(self, x_hat, x):
        # within 10%
        x_min = x*0.95
        x_max = x*1.05

        n_correct = 0.0
        for curr_x_hat, curr_x_min, curr_x_max in zip(x_hat, x_min, x_max):
            n_correct += float(curr_x_max > curr_x_hat > curr_x_min)
        return n_correct/float(x.shape[0])


class LatentField(AbstractField):

    def __init__(self, name, latent_dim, **kwargs):
        self.latent_dim = latent_dim
        super().__init__(name, **kwargs)

    @property
    def n_features(self):
        return self.latent_dim

    def guess_correct(self, x_hat, x):
        # discrepancies are how far the value is from the target.
        # The magnitude is not going to influence the discrepancy as
        # it's going to be normalized by the input value.
        # Since the ratio between the two values is 1 if they are equal,
        # the discrepancy is calculated as distance from one.
        # Moreover, we need to be careful to avoid division by zero,
        # hence the division need to be done between absolute value,
        # where the denominator is offsetted by a small epsilon
        # to ensure 0 is never at the denominator.
        sign_x = np.sign(x)
        sign_x_hat = np.sign(x_hat)
        same_sign = sign_x * sign_x_hat
        ratio = same_sign * (np.abs(x_hat) / ( np.abs(x) + 1e-5 ))
        per_feature_discrepancies = np.abs(1.0 - ratio)

        # mantain rows, collapse columns
        discrepancies = np.mean(per_feature_discrepancies,axis=1)

        # 10% discrepancy is used for identifying a correct vs a
        correct_ones = (discrepancies < 0.1)

        # fraction of correct guesses is calculated by the sum of correct divided by
        # the total cardinality of the input tensor
        ret = np.sum(correct_ones/float(len(correct_ones)))

        return ret


class TextField(AbstractField):
    def encode(self, entries, set_size, **kwargs):
        ret = [text_model.instance().encode(x).tolist() for x in entries]
        short_of = set_size - len(entries)
        if short_of > 0:
            for i in range(short_of):
                ret.append(text_model.instance().empty_vector.tolist())
        return ret

    @property
    def n_features(self):
        return text_model.instance().n_features

    def guess_correct(self, x_hat, x):
        # within 10%
        x_min = x*0.95
        x_max = x*1.05

        n_correct = 0.0
        for curr_x_hat, curr_x_min, curr_x_max in zip(x_hat, x_min, x_max):
            datapoint_correct = True
            for x_hat_loc, x_min_loc, x_max_loc in zip(curr_x_hat, curr_x_min, curr_x_max):
                datapoint_correct = datapoint_correct and x_hat_loc < x_max_loc
                datapoint_correct = datapoint_correct and x_hat_loc > x_min_loc
            n_correct += float(datapoint_correct)
        return n_correct/float(x.shape[0])


class BooleanField(AbstractField):
    def encode(self, entries, set_size, **kwargs):
        ret = [bool(x) for x in entries]
        short_of = set_size - len(entries)
        if short_of > 0:
            for i in range(short_of):
                ret.append(text_model.instance().empty_vector.tolist())
        return ret

    @property
    def n_features(self):
        return 1

    def guess_correct(self, x_hat, x):
        x_hat_descaled = self.scaler.inverse_transform(x_hat)
        x_descaled = self.scaler.inverse_transform(x)
        correct_ones = np.linalg.norm(x_hat_descaled - x_descaled, axis=1) < 0.5
        correct_ratio = np.mean(correct_ones)
        return correct_ratio
