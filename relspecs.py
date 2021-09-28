import abc
import datetime
import re
from abc import abstractmethod
import numpy as np
import functools

import torch.nn

import persistency
import utils
import sklearn.preprocessing

@functools.cache
def get_codelists():
    db = persistency.mongo_db()
    ret = {}
    for curr in db['codelists'].find({}):
        ret[curr['name']] = curr['codelist']
    return ret

class RelsCollection(object):
    rels_dict = {}

    def __init__(self, rels_list):
        for rel in rels_list:
            if rel.name in self.rels_dict.keys():
                raise Exception(f"rel {rel.name} already in this RelsCollection")
            self.rels_dict[rel.name] = rel

    def __getitem__(self, rel_name):
        return self.rels_dict[rel_name]

    def __iter__(self):
        return iter(self.rels_dict.values())

    @property
    def names(self):
        return self.rels_dict.keys()

class Rel(object):
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields

    def divide(self, tensor, with_set_index=False):
        ret = []
        for start, end in self.fields_intervals(with_set_index=with_set_index):
            ret.append(tensor[:, start:end])
        return ret

    def glue(self, tensor_list): # FIXME: maybe to some other module?
        if type(tensor_list) is list:
            assert len(tensor_list) > 0
            first = tensor_list[0]
            if type(first) is torch.Tensor:
                ret = torch.hstack(tensor_list)
            elif type(first) is np.ndarray:
                ret = np.hstack(tensor_list)
            else:
                raise Exception("elements in the list must be either numpy arrays or torch tensors")
        else:
            # possibly already glued?
            ret = tensor_list
        return ret

    @property
    def scalers(self):
        return [ curr.scaler for curr in self.fields]

    def make_and_fit_scalers(self, train_dataset, test_dataset, with_set_index=False):
        train_sections = self.divide(train_dataset, with_set_index=with_set_index)
        test_sections = self.divide(test_dataset, with_set_index=with_set_index)
        train_sections_scaled = []
        test_sections_scaled = []
        for field, train_section, test_section in zip(self.fields, train_sections, test_sections):
            field.make_and_fit_scaler(train_section)
            train_section_scaled = field.scaler.transform(train_section)
            test_section_scaled = field.scaler.transform(test_section)
            train_sections_scaled.append(train_section_scaled)
            test_sections_scaled.append(test_section_scaled)
        train_dataset_scaled = self.glue(train_sections_scaled)
        test_dataset_scaled = self.glue(test_sections_scaled)
        return train_dataset_scaled, test_dataset_scaled

    @property
    def n_fields(self):
        return len(self.fields)

    @property
    def n_features(self):
        return sum([curr.n_features for curr in self.fields])

    @property
    def fields_names(self):
        return [ f.name for f in self.fields ]

    def fields_intervals(self,with_set_index=False):
        start = 0
        intervals = []
        if with_set_index:
            intervals.append((0,1))
            start=1
        for field in self.fields:
            end = start + field.n_features
            intervals.append((start,end))
            start = end
        return intervals

    @property
    def codelists_names(self):
        ret = []
        for field in self.fields:
            if type(field) is CategoryField:
                ret.append(field.codelist_name)
        return ret

class AbstractField(abc.ABC):
    def __init__(
            self,
            name,
            output_activation_function=None,
            loss_function=None
    ):
        self.name = name
        self._output_activation_function = output_activation_function
        self._loss_function = loss_function
        self._scaler = None

    @property
    def n_features(self):
        raise Exception("not implemented")

    @property
    def output_activation_function(self):
        return self._output_activation_function

    @property
    def loss_function(self):
        return self._loss_function

    def guess_correct(self, x_hat, x_orig):
        raise Exception("not implemented")

    @property
    def scaler(self):
        return self._scaler

    def make_scaler(self):
        # default scaler
        return sklearn.preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))

    def make_and_fit_scaler(self, data):
        self._scaler = self.make_scaler()
        self._scaler.fit(data)
        return self._scaler

class DatetimeField(AbstractField):

    def encode(self, entries, set_size, **kwargs):
        ret = []
        for entry in entries:
            entry = re.match('(.*)Z', entry).groups()[0] # python's datetime does not parse the final 'Z'
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
        x_hat_descaled = self.scaler.inverse_transform(x_hat)[:,:3]
        x_descaled = self.scaler.inverse_transform(x)[:,:3]
        correct_ones = np.linalg.norm(x_hat_descaled - x_descaled,axis=1) > 0.5
        correct_ratio = np.mean(correct_ones)
        return correct_ratio

    @property
    def n_features(self):
        return 9 # 9 is the cardinality of the timetuple

class CategoryField(AbstractField):
    def __init__(self, name, codelist_name, **kwargs):
        if 'output_activation_function' not in kwargs:
            kwargs['output_activation_function'] = torch.nn.Softmax()
        if 'loss_function' not in kwargs:
            kwargs['loss_function'] = utils.OneHotCrossEntropyLoss()
        super().__init__(name, **kwargs)
        self.codelist_name = codelist_name

    @property
    def codelist(self):
        ret = get_codelists()[self.codelist_name]
        return ret

    def encode(self, entries, set_size, **kwargs):
        ret = np.zeros((set_size,len(self.codelist)))
        for index_code, code in enumerate(entries):
            if code is None:
                raise Exception("code is None: this shouldn't happen")
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
        return 1 # it's just one

    def guess_correct(self, x_hat, x):
        # within 10%
        x_min = x*0.95
        x_max = x*1.05

        n_correct = 0.0
        for curr_x_hat, curr_x in zip(x_hat,x):
            n_correct += float(x_hat[0] < curr_x_hat[0] and x_hat[0] > curr_x[0])
        return n_correct/float(x.shape[0])

rels = RelsCollection([
    Rel("budget", [
        CategoryField(
            "value_currency",
            'Currency',
            #output_activation_function=torch.nn.Sigmoid(),
            #loss_function=torch.nn.MSELoss()
        ),
        CategoryField("type", 'BudgetType'),
        CategoryField("status", 'BudgetStatus'),
        DatetimeField("period_start_iso_date"),
        DatetimeField("period_end_iso_date"),
        NumericalField("value")
    ])
])
