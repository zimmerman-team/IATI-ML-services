import abc
import datetime
import re
from abc import abstractmethod
import numpy as np
import functools

import torch.nn

import persistency
import utils


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

    def scale(self, train_dataset, test_dataset, default_scaler, with_set_index=False):
        scalers = []
        train_sections = self.divide(train_dataset, with_set_index=with_set_index)
        test_sections = self.divide(test_dataset, with_set_index=with_set_index)
        train_sections_scaled = []
        test_sections_scaled = []
        for field, train_section, test_section in zip(self.fields, train_sections, test_sections):
            scaler = field.scaler or default_scaler()
            scaler.fit(train_section)
            scalers.append(scaler)
            train_section_scaled = scaler.transform(train_section)
            test_section_scaled = scaler.transform(test_section)
            train_sections_scaled.append(train_section_scaled)
            test_sections_scaled.append(test_section_scaled)
        train_dataset_scaled = self.glue(train_sections_scaled)
        test_dataset_scaled = self.glue(test_sections_scaled)
        return train_dataset_scaled, test_dataset_scaled, scalers

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
    def __init__(self, name):
        self.name = name

    @property
    def n_features(self):
        raise Exception("not implemented")

    @property
    def output_activation_function(self):
        # default is none specified, hence in this case please use the one specified
        # in the model configuration
        return None

    @property
    def loss_function(self):
        # default is none specified, hence in this case please use the one specified
        # in the model configuration
        return None

    @property
    def scaler(self):
        # default is none specified, hence in this case please use the one specified
        # in the model configuration
        return None

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

    @property
    def n_features(self):
        return 9 # 9 is the cardinality of the timetuple

class CategoryField(AbstractField):
    def __init__(self, name, codelist_name):
        self.name = name
        self.codelist_name = codelist_name

    def encode(self, entries, set_size, **kwargs):
        codelist = get_codelists()[self.codelist_name]

        ret = np.zeros((set_size,len(codelist)))
        for index_code, code in enumerate(entries):
            if code is None:
                raise Exception("code is None: this shouldn't happen")
            else:
                index_one = codelist.index(code)
                ret[index_code, index_one] = 1
        short_of = set_size - len(entries)
        if short_of > 0:
            for i in range(short_of):
                avg = 1.0 / float(len(codelist))
                ret[set_size - 1 - i, :] = avg

        ret = ret.tolist()
        return ret

    @property
    def n_features(self):
        return len(get_codelists()[self.codelist_name])

    @property
    def output_activation_function(self):
        return torch.nn.Softmax()

    @property
    def loss_function(self):
        return utils.OneHotCrossEntropyLoss()

    @property
    def scaler(self):
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

rels = RelsCollection([
    Rel("budget", [
        CategoryField("value_currency",'Currency'),
        CategoryField("type", 'BudgetType'),
        CategoryField("status", 'BudgetStatus'),
        DatetimeField("period_start_iso_date"),
        DatetimeField("period_end_iso_date"),
        NumericalField("value")
    ])
])
