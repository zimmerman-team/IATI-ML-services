#!/bin/env python3
import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import datetime
import requests
import logging
import time
import json
import tempfile
import os
import re
import pymongo
import datetime
import time
import random
import numpy as np
from collections import defaultdict, OrderedDict
import sklearn.model_selection
import mlflow
import abc
import enum

import utils

logging.basicConfig(level=logging.DEBUG)

DATASTORE_ACTIVITY_URL="https://datastore.iati.cloud/api/v2/activity"
DATASTORE_CODELIST_URL="https://datastore.iati.cloud/api/codelists/{}/"
PAGE_SIZE=200
MAX_PAGES=200
MONGODB_CONN="mongodb://mongouser:XGkS1wDyb4922@localhost:27017/learning_sets"

class Rel(object):
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields

    @property
    def codelists_names(self):
        ret = []
        for field in self.fields:
            if type(field) is CategoryField:
                ret.append(field.codelist_name)
        return ret

def extract_codelists(_rels):
    ret = set()
    for rel in _rels:
        ret = ret.union(rel.codelists_names)
    return ret

class FieldType(enum.Enum): # FIXME: is this really necessary?
    DATETIME = enum.auto()
    CATEGORY = enum.auto()
    NUMERICAL = enum.auto()

class AbstractField(abc.ABC):
    def __init__(self, name):
        self.name = name

    @property
    def type_(self):
        raise Exception("not implemented in subclass")

class DatetimeField(AbstractField):
    @property
    def type_(self):
        return FieldType.DATETIME

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
                tmp = [0] * 9  # 9 is the cardinality of the timetuple
                ret.append(tmp)
        return ret

class CategoryField(AbstractField):
    def __init__(self, name, codelist_name):
        self.name = name
        self.codelist_name = codelist_name

    @property
    def type_(self):
        return FieldType.CATEGORY

    def encode(self, entries, set_size, **kwargs):
        codelist = kwargs['all_codelists'][self.codelist_name]

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

class NumericalField(AbstractField):
    @property
    def type_(self):
        return FieldType.NUMERICAL

    def encode(self, entries, set_size, **kwargs):
        ret = [float(x) for x in entries]
        short_of = set_size - len(entries)
        if short_of > 0:
            for i in range(short_of):
                ret.append(0.0)
        return ret

rels = [
    Rel("budget", [
        CategoryField("value_currency",'Currency'),
        CategoryField("type", 'BudgetType'),
        CategoryField("status", 'BudgetStatus'),
        DatetimeField("period_start_iso_date"),
        DatetimeField("period_end_iso_date"),
        NumericalField("value")
    ])
]

def write_tmp(data):
    filename = tempfile.mktemp()
    with open(filename, 'w+') as f:
        json.dump(data, f)
        f.flush()
    return filename

def read_tmp(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def send(ti, data):
    tmp_filename = write_tmp(data)
    ti.xcom_push(key='tmp_filename', value=tmp_filename)

def recv(ti,task_id):
    input_filename = ti.xcom_pull(key='tmp_filename', task_ids=task_id)
    assert input_filename is not None
    data = read_tmp(input_filename)
    return data

def clear_recv(ti, task_id):
    input_filename = ti.xcom_pull(key='tmp_filename', task_ids=task_id)
    os.unlink(input_filename)

def download(start, ti):
    params = {
        'q': "*:*",
        'start': start,
        'rows': PAGE_SIZE
    }
    logging.info(f"requesting {DATASTORE_ACTIVITY_URL} with {params}")
    response = requests.get(DATASTORE_ACTIVITY_URL,params=params)
    data = response.json()
    send(ti,data)

def parse(page,ti):
    rels_vals = defaultdict(lambda: defaultdict(lambda:dict()))
    data = recv(ti, f"download_{page}")
    for activity in data['response']['docs']:
        activity_id = activity['iati_identifier']
        for k,v in activity.items():
            for rel in rels:
                m = re.match(f'{rel.name}_(.*)',k)
                if m is not None:
                    rel_field = m.group(1)
                    rels_vals[rel.name][activity_id][rel_field] = v

    for rel, sets in rels_vals.items():
        for activity_id, fields in sets.items():
            assert len(set(map(lambda l: len(l),fields.values()))) == 1, f"all fields need to have same amount of values (rel:{rel}, activity_id:{activity_id}, fields:{fields}"
    send(ti,rels_vals)
    clear_recv(ti, f"download_{page}")

def persist(page, ti):
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    data = recv(ti, f'parse_{page}')

    for rel, sets in data.items():
        for activity_id, set_ in sets.items():
            db[rel].delete_one({'activity_id':activity_id}) # remove pre-existing set for this activity
            db[rel].insert_one({
                'activity_id': activity_id,
                'set_':set_
            })

    clear_recv(ti, f'parse_{page}')

def codelists(ti):
    all_codelists = {}
    for codelist in extract_codelists(rels):
        url = DATASTORE_CODELIST_URL.format(codelist)
        params = {'format':'json'}
        response = requests.get(url,params=params)
        data = response.json()
        l = []
        for curr in data:
            l.append(curr['code'])
        all_codelists[codelist] = l
    send(ti,all_codelists)

def get_set_size(set_):
    size = 0
    for field_name, values in set_.items():
        size = max(len(values),size)
    return size

def encode(ti):
    all_codelists = recv(ti, 'codelists')
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    for rel in rels:
        coll_in = db[rel.name]
        coll_out = db[rel.name + "_encoded"]
        for document in coll_in.find():
            document = dict(document) # copy
            set_ = document['set_']
            set_size = get_set_size(set_)
            for field in rel.fields:
                encodable = set_.get(field.name,[])
                tmp = field.encode(encodable, set_size, all_codelists=all_codelists)
                set_[field.name] = tmp

            del document['_id']
            lens = list(map(lambda field: len(set_[field.name]), rel.fields))
            if len(set(lens)) > 1:
                msg ="lens "+str(lens)+" for fields "+str(list(map(lambda curr:curr.name,rel.fields)))
                logging.info(msg)
                logging.info(document)
                raise Exception(msg)
            coll_out.delete_one({'activity_id':document['activity_id']}) # remove pre-existing set for this activity
            coll_out.insert_one(document)
    clear_recv(ti, 'codelists')

def arrayfy(ti):
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    for rel in rels:
        coll_in = db[rel.name+"_encoded"]
        coll_out = db[rel.name+"_arrayfied"]
        coll_out.delete_many({}) # empty the collection
        for set_index, document in enumerate(coll_in.find()):
            set_npas = []
            set_ = document['set_']
            keys = sorted(set_.keys())
            for k in keys: # we need to always have a same ordering of the fields!
                if len(set_[k]) > 0 and type(set_[k][0]) is list:
                    floats = list(map(lambda v: list(map(lambda x:float(x),v)), set_[k]))
                else: # not something that is dummified: simple numerical value field
                    floats = list(map(lambda x: [float(x)], set_[k]))
                field_npa = np.array(floats)
                set_npas.append(field_npa)
            if len(set(map(lambda curr: curr.shape[0],set_npas))) > 1:
                logging.info("keys:" + str(keys))
                logging.info("set_npas shapes:"+str([curr.shape for curr in set_npas]))
            set_npa = np.hstack(set_npas)
            set_npa_serialized = utils.serialize(set_npa)
            coll_out.insert_one({'set_index':set_index,'npa':set_npa_serialized})

def to_npa(ti):
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    for rel in rels:
        coll_in = db[rel.name+"_arrayfied"]
        coll_out = db['npas']
        rel_npas = []
        for document in coll_in.find():
            set_npa = utils.deserialize(document['npa'])
            set_index = document['set_index']
            set_index_col = np.ones((set_npa.shape[0], 1))*set_index
            rel_npas.append(np.hstack([set_index_col, set_npa]))
        rel_npa = np.vstack(rel_npas)
        coll_out.remove({'rel':rel.name})
        coll_out.insert_one({
            'rel':rel.name,
            'npa':utils.serialize(rel_npa)
        })

def to_tsets(ti):
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    for rel in rels:
        coll_in = db[rel.name+'_arrayfied']
        set_indices_results = coll_in.find({},{'set_index':1})
        set_indices = list(set(map( lambda document: document['set_index'],set_indices_results)))
        train_indices, test_indices = sklearn.model_selection.train_test_split(set_indices, train_size=0.75)
        coll_out = db['npas_tsets']
        train_npas = []
        test_npas = []
        for document in coll_in.find():
            set_npa = utils.deserialize(document['npa'])
            set_index = document['set_index']
            set_index_col = np.ones((set_npa.shape[0], 1))*set_index
            npa = np.hstack([set_index_col, set_npa])
            if set_index in train_indices:
                train_npas.append(npa)
            elif set_index in test_indices:
                test_npas.append(npa)

        train_npa = np.vstack(train_npas)
        test_npa = np.vstack(test_npas)
        coll_out.insert_one({
            'rel':rel.name,
            'train_npa':utils.serialize(train_npa),
            'test_npa': utils.serialize(test_npa)
        })

default_args = {
    'retries':5,
    'retry_delay':datetime.timedelta(minutes=1)
}

with DAG(
    'download_sets',
    description='Downloads sets data from IATI.cloud',
    tags=['download','sets'],
    default_args=default_args
) as dag:

    pages = list(range(MAX_PAGES))

    t_codelists = PythonOperator(
        task_id="codelists",
        python_callable=codelists,
        start_date=days_ago(2),
        op_kwargs={}
    )

    t_encode = PythonOperator(
        task_id="encode",
        python_callable=encode,
        start_date=days_ago(2),
        op_kwargs={}
    )

    t_arrayfy = PythonOperator(
        task_id="arrayfy",
        python_callable=arrayfy,
        start_date=days_ago(2),
        op_kwargs={}
    )

    t_to_npa= PythonOperator(
        task_id="to_npa",
        python_callable=to_npa,
        start_date=days_ago(2),
        op_kwargs={}
    )

    t_to_tsets= PythonOperator(
        task_id="to_tsets",
        python_callable=to_tsets,
        start_date=days_ago(2),
        op_kwargs={}
    )

    for page in pages:
        start = page*PAGE_SIZE
        t_download = PythonOperator(
            task_id=f"download_{page}",
            python_callable = download,
            start_date = days_ago(2),
            op_kwargs={'start':start}
        )

        t_parse = PythonOperator(
            task_id=f"parse_{page}",
            python_callable = parse,
            start_date = days_ago(2),
            op_kwargs={'page':page}
        )

        t_persist = PythonOperator(
            task_id=f"persist_{page}",
            python_callable = persist,
            start_date = days_ago(2),
            op_kwargs={'page':page}
        )
        t_download >> t_parse >> t_persist >> t_encode
    t_codelists >> t_encode >> t_arrayfy >> t_to_npa
    t_arrayfy >> t_to_tsets