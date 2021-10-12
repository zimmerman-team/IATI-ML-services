#!/bin/env python3
import functools

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
import logging
import re
import datetime
import numpy as np
from collections import defaultdict
import sklearn.model_selection
import os
import sys

# since airflow's DAG modules are imported elsewhere (likely ~/airflow)
# we have to explicitly add the path of this module to python's path
path = os.path.dirname(os.path.abspath(__file__))
sys.path = [path]+sys.path
import utils
import relspecs
import persistency
import large_mp

rels = relspecs.rels.downloadable
logging.basicConfig(level=logging.DEBUG)

DATASTORE_ACTIVITY_URL="https://datastore.iati.cloud/api/v2/activity"
DATASTORE_CODELIST_URL="https://datastore.iati.cloud/api/codelists/{}/"
PAGE_SIZE=1000
MAX_PAGES=1000

def extract_codelists(_rels):
    ret = set()
    for rel in _rels:
        ret = ret.union(rel.codelists_names)
    return ret

def download(start, ti):
    fl = ",".join(["iati_identifier"]+relspecs.rels.downloadable_prefixed_fields_names)
    params = {
        'q': "*:*",
        'fl': fl,
        'start': start,
        'rows': PAGE_SIZE
    }
    logging.info(f"requesting {DATASTORE_ACTIVITY_URL} with {params}")
    response = requests.get(DATASTORE_ACTIVITY_URL,params=params)
    logging.info(f"response.url:{response.url}")
    data = response.json()
    large_mp.send(ti,data)

def parse(page,ti):
    rels_vals = defaultdict(lambda: defaultdict(lambda:dict()))
    data = large_mp.recv(ti, f"download_{page}")
    for activity in data['response']['docs']:
        activity_id = activity['iati_identifier']
        #logging.info(f"processing activity {activity_id}")
        for k,v in activity.items():
            #logging.info(f"processing activity item {k}")
            for rel in rels:
                #logging.info(f"processing rel {rel.name}")
                m = re.match(f'{rel.name}_(.*)',k)
                if m is not None:
                    rel_field = m.group(1)
                    if rel_field in rel.fields_names:
                        #logging.info(f"considering field {rel_field}")
                        rels_vals[rel.name][activity_id][rel_field] = v

    for rel, sets in rels_vals.items():
        for activity_id, fields in sets.items():
            logging.info('fields.keys'+str(fields.keys()))
            lens = {}
            for k,v in fields.items():
                lens[k] = len(v)
            assert len(set(lens.values())) == 1, f"all fields need to have same amount of values (rel:{rel}, lens:{lens} activity_id:{activity_id}, fields:{fields}"
    large_mp.send(ti,rels_vals)
    large_mp.clear_recv(ti, f"download_{page}")

def persist(page, ti):
    db = persistency.mongo_db()
    data = large_mp.recv(ti, f'parse_{page}')

    for rel, sets in data.items():
        for activity_id, set_ in sets.items():
            db[rel].delete_one({'activity_id':activity_id}) # remove pre-existing set for this activity
            db[rel].insert_one({
                'activity_id': activity_id,
                'set_':set_
            })

    large_mp.clear_recv(ti, f'parse_{page}')

def codelists(ti):
    db = persistency.mongo_db()
    for codelist_name in extract_codelists(rels):
        url = DATASTORE_CODELIST_URL.format(codelist_name)
        params = {'format':'json'}
        response = requests.get(url,params=params)
        data = response.json()
        l = []
        for curr in data:
            l.append(curr['code'])
        db['codelists'].delete_many({'name':codelist_name})
        db['codelists'].insert({
            'name':codelist_name,
            'codelist':l
        })

def get_set_size(set_):
    size = 0
    for field_name, values in set_.items():
        size = max(len(values),size)
    return size

def encode(rel,ti):
    db = persistency.mongo_db()
    coll_in = db[rel.name]
    coll_out = db[rel.name + "_encoded"]
    for document in coll_in.find(no_cursor_timeout=True):
        document = dict(document) # copy
        set_ = document['set_']
        set_size = get_set_size(set_)
        for field in rel.fields:
            encodable = set_.get(field.name,[])
            tmp = field.encode(encodable, set_size)
            set_[field.name] = tmp

        del document['_id']
        lens = list(map(lambda field: len(set_[field.name]), rel.fields))
        if len(set(lens)) > 1:
            msg ="lens "+str(lens)+" for fields "+str(rel.fields_names)
            logging.info(msg)
            logging.info(document)
            raise Exception(msg)
        coll_out.delete_one({'activity_id':document['activity_id']}) # remove pre-existing set for this activity
        coll_out.insert_one(document)

def arrayfy(rel,ti):
    db = persistency.mongo_db()
    coll_in = db[rel.name+"_encoded"]
    coll_out = db[rel.name+"_arrayfied"]
    coll_out.delete_many({}) # empty the collection
    for set_index, document in enumerate(coll_in.find()):
        set_npas = []
        set_ = document['set_']
        keys = rel.fields_names
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

def to_npa(rel,ti):
    db = persistency.mongo_db()
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
        'creation_date': utils.strnow(),
        'npa':utils.serialize(rel_npa),
        'npa_rows': rel_npa.shape[0],
        'npa_cols': rel_npa.shape[1]
    })

def to_tsets(rel,ti):
    db = persistency.mongo_db()
    coll_in = db[rel.name+'_arrayfied']
    set_indices_results = coll_in.find({},{'set_index':1})
    set_indices = list(set(map( lambda document: document['set_index'],set_indices_results)))
    train_indices, test_indices = sklearn.model_selection.train_test_split(set_indices, train_size=0.90)
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
        'creation_time': utils.strnow(),
        'train_npa':utils.serialize(train_npa),
        'test_npa': utils.serialize(test_npa),
        'train_npa_rows':train_npa.shape[0],
        'train_npa_cols':train_npa.shape[1],
        'test_npa_rows':test_npa.shape[0],
        'test_npa_cols':test_npa.shape[1]
    })

default_args = {
    'retries':2,
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

    t_persist = {}
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

        t_persist[page] = PythonOperator(
            task_id=f"persist_{page}",
            python_callable = persist,
            start_date = days_ago(2),
            op_kwargs={'page':page}
        )
        t_download >> t_parse >> t_persist[page]

    for rel in rels:
        t_to_npa = PythonOperator(
            task_id=f"to_npa_{rel.name}",
            python_callable=to_npa,
            start_date=days_ago(2),
            op_kwargs={'rel':rel}
        )

        t_to_tsets = PythonOperator(
            task_id=f"to_tsets_{rel.name}",
            python_callable=to_tsets,
            start_date=days_ago(2),
            op_kwargs={'rel':rel}
        )
        t_encode = PythonOperator(
            task_id=f"encode_{rel.name}",
            python_callable=encode,
            start_date=days_ago(2),
            op_kwargs={'rel':rel}
        )

        t_arrayfy = PythonOperator(
            task_id=f"arrayfy_{rel.name}",
            python_callable=arrayfy,
            start_date=days_ago(2),
            op_kwargs={'rel':rel}
        )

        for page in pages:
            t_persist[page] >> t_encode
        t_codelists >> t_encode >> t_arrayfy
        t_arrayfy >> t_to_npa
        t_arrayfy >> t_to_tsets
