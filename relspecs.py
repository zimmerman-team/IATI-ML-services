import abc
import datetime
import re
from abc import abstractmethod
import numpy as np
import functools
import sklearn.preprocessing
import torch.nn

import persistency
import utils
import config
import text_model

@functools.cache
def get_codelists():
    db = persistency.mongo_db()
    ret = {}
    for curr in db['codelists'].find({}):
        ret[curr['name']] = curr['codelist']
    return ret

class RelsCollection(utils.Collection):

    def __init__(self, rels_list):
        for rel in rels_list:
            if rel.name in self.names:
                raise Exception(f"rel {rel.name} already in this RelsCollection")
            self[rel.name] = rel
        super().__init__()

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

        #FIXME: justify why norm < 0.5
        correct_ones = np.linalg.norm(x_hat_descaled - x_descaled,axis=1) < 0.5
        correct_ratio = np.mean(correct_ones)
        return correct_ratio

    @property
    def n_features(self):
        return 9 # 9 is the cardinality of the timetuple

class CategoryField(AbstractField):
    def __init__(self, name, codelist_name, **kwargs):
        if 'output_activation_function' not in kwargs:
            kwargs['output_activation_function'] = torch.nn.Softmax(dim=1)

        prevent_constant_prediction = kwargs.pop('prevent_constant_prediction', None)

        super().__init__(name, **kwargs)
        self.codelist_name = codelist_name

        if 'loss_function' not in kwargs:
            if(prevent_constant_prediction):
                prevent_constant_prediction_idx = self.codelist.index(prevent_constant_prediction)
                siz = len(self.codelist)
                weight = np.ones(siz)/siz
                weight[prevent_constant_prediction_idx] /= 10
                print('foo',weight[prevent_constant_prediction_idx])
            else:
                weight = None
            kwargs['loss_function'] = utils.OneHotCrossEntropyLoss(weight=weight)


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
        for curr_x_hat, curr_x_min, curr_x_max in zip(x_hat,x_min,x_max):
            n_correct += float(curr_x_hat < curr_x_max and curr_x_hat > curr_x_min)
        return n_correct/float(x.shape[0])

class TextField(AbstractField):
    def encode(self, entries, set_size, **kwargs):
        ret = [text_model.instance().encode(x) for x in entries]
        short_of = set_size - len(entries)
        if short_of > 0:
            for i in range(short_of):
                ret.append(text_model.instance().empty_vector)
        return ret

    @property
    def n_features(self):
        return text_model.instance().n_features

    def guess_correct(self, x_hat, x):
        # within 10%
        x_min = x*0.95
        x_max = x*1.05

        n_correct = 0.0
        for curr_x_hat, curr_x_min, curr_x_max in zip(x_hat,x_min,x_max):
            datapoint_correct = True
            for x_hat_loc,x_min_loc,x_max_loc in zip(curr_x_hat,curr_x_min,curr_x_max):
                datapoint_correct = datapoint_correct and x_hat_loc < x_max_loc
                datapoint_correct = datapoint_correct and x_hat_loc > x_min_loc
            n_correct += float(datapoint_correct)
        return n_correct/float(x.shape[0])

rels = RelsCollection([
    Rel("budget", [
        CategoryField(
            "value_currency",
            'Currency',
            #output_activation_function=torch.nn.Sigmoid(),
            #loss_function=torch.nn.MSELoss(),
            prevent_constant_prediction='USD'
        ),
        CategoryField("type", 'BudgetType'),
        CategoryField("status", 'BudgetStatus'),
        DatetimeField("period_start_iso_date"),
        DatetimeField("period_end_iso_date"),
        NumericalField("value")
    ]),
    Rel("result",[
        CategoryField("type", 'ResultType'),
        TextField("title_narrative"),
        #TextField("description_narrative"),
        #aggregation_status
        #CategoryField("indicator_measure","IndicatorMeasure"),
        #indicator_ascending
        #indicator_aggregation_status
        #TextField("indicator_title_narrative"),
        #CategoryField("indicator_title_narrative_lang","Language"),
        #TextField("indicator_description_narrative"),
        #CategoryField("indicator_description_narrative_lang","Language"),
        #NumericalField("indicator_baseline_year"),
        #DatetimeField("indicator_baseline_iso_date"),
        #indicator_baseline_value

        # FIXME TODO WARNING: following fields may be presented multiple times for each
        # result instance. Hence their cardinality may be k*cardinality(result)
        # should I consider only the first for each result?
        # But then they are not grouped for each result but all put in the same list,
        # so that might be difficult.
        #DatetimeField("indicator_period_period_start_iso_date"),
        #DatetimeField("indicator_period_period_end_iso_date"),
        #NumericalField("indicator_period_target_value"),
        #NumericalField("indicator_period_actual_value")
    ])
])
