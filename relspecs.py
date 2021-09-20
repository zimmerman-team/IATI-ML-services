import abc
import datetime
import re
from abc import abstractmethod
import numpy as np
import functools

import persistency

@functools.cache
def get_codelists():
    db = persistency.mongo_db()
    ret = {}
    for curr in db['codelists'].find({}):
        ret [curr['name']] = curr['codelist']
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

class Rel(object):
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields

    @property
    def n_fields(self):
        return len(self.fields)

    @property
    def n_features(self):
        return sum([curr.n_features for curr in self.fields])

    @property
    def fields_names(self):
        return [ f.name for f in self.fields ]

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
