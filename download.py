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
from bson.binary import Binary
import pickle
from collections import defaultdict, OrderedDict
import sklearn.model_selection
import mlflow

logging.basicConfig(level=logging.DEBUG)

DATASTORE_ACTIVITY_URL="https://datastore.iati.cloud/api/v2/activity"
DATASTORE_CODELIST_URL="https://datastore.iati.cloud/api/codelists/{}/"
PAGE_SIZE=100
MAX_PAGES=500
MONGODB_CONN="mongodb://mongouser:XGkS1wDyb4922@localhost:27017/learning_sets"
chosen_codelists = ["Currency","BudgetType","BudgetStatus"] # FIXME: duplicated info from rels_to_codelists?
chosen_rels = ['budget']
rels_to_codelists = {
    'budget': {
        'value_currency': 'Currency',
        'type': 'BudgetType',
        'status': 'BudgetStatus'
    }
}

rels_datetimes = {
    'budget': (
        'period_start_iso_date',
        'period_end_iso_date'
    )
}

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
    logging.info(f"sending data:{data}")
    ti.xcom_push(key='tmp_filename', value=tmp_filename)

def recv(ti,task_id):
    input_filename = ti.xcom_pull(key='tmp_filename', task_ids=task_id)
    logging.info(f"input_filename:{input_filename}")
    assert input_filename is not None
    data = read_tmp(input_filename)
    logging.info(f"received data {data}")
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
    logging.info(f'task parse started! getting tmp_filename from page {page}..')
    logging.info('previous task:'+str(ti.previous_ti))
    rels_vals = defaultdict(lambda: defaultdict(lambda:dict()))
    data = recv(ti, f"download_{page}")
    for activity in data['response']['docs']:
        activity_id = activity['iati_identifier']
        for k,v in activity.items():
            for rel in chosen_rels:
                m = re.match(f'{rel}_(.*)',k)
                if m is not None:
                    rel_field = m.group(1)
                    rels_vals[rel][activity_id][rel_field] = v

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
        logging.info(f"rel:{rel},sets:{sets}")
        for activity_id, set_ in sets.items():
            db[rel].delete_one({'activity_id':activity_id}) # remove pre-existing set for this activity
            db[rel].insert_one({
                'activity_id': activity_id,
                'set_':set_
            })

    clear_recv(ti, f'parse_{page}')

def codelists(ti):
    all_codelists = {}
    for codelist in chosen_codelists:
        url = DATASTORE_CODELIST_URL.format(codelist)
        params = {'format':'json'}
        response = requests.get(url,params=params)
        data = response.json()
        l = []
        for curr in data:
            l.append(curr['code'])
        all_codelists[codelist] = l
    send(ti,all_codelists)

def dummify_codes(codes_entries,codelist):

    if not codes_entries:
        ret = np.zeros((0,len(codelist)))
    else:
        logging.info("dummify_codes codelist:"+str(codelist))
        ret = np.zeros((len(codes_entries),len(codelist)))
        for index_code, code in enumerate(codes_entries):
            if code is None:
                # missing information about active category: just assume a uniform probability distribution
                #ret[index_code,:] = 1.0/float(len(codelist))
                raise Exception("code is None: this shouldn't happen")
            else:
                index_one = codelist.index(code)
                ret[index_code, index_one] = 1
    ret = ret.tolist()
    return ret

def encode_datetime(entries):
    ret = []
    for entry in entries:
        entry = re.match('(.*)Z', entry).groups()[0] # python's datetime does not parse the final 'Z'
        dt = datetime.datetime.fromisoformat(entry)
        t = tuple(dt.timetuple())
        ret.append(t)
    return ret

def encode(ti):
    all_codelists = recv(ti, 'codelists')
    logging.info('all_codelists:'+str(all_codelists))
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    for rel in chosen_rels:
        coll_in = db[rel]
        coll_out = db[rel + "_encoded"]
        for document in coll_in.find():
            document = dict(document) # copy
            set_ = document['set_']
            for field_name, codelist_name in rels_to_codelists[rel].items():
                tmp = dummify_codes(set_.get(field_name), all_codelists[codelist_name])
                logging.info(f'setting {field_name} to {tmp}')
                set_[field_name] = tmp
            for field_name in rels_datetimes[rel]:
                if field_name in set_.keys():
                    tmp = encode_datetime(set_[field_name])
                else:
                    tmp = [0] * 9 # 9 is the cardinality of the timetuple
                set_[field_name] = tmp
            del document['_id']
            coll_out.delete_one({'activity_id':document['activity_id']}) # remove pre-existing set for this activity
            coll_out.insert_one(document)
    clear_recv(ti, 'codelists')

def serialize(npa):
    return Binary(pickle.dumps(npa, protocol=2))

def arrayfy(ti):
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    for rel in chosen_rels:
        coll_in = db[rel+"_encoded"]
        coll_out = db[rel+"_arrayfied"]
        coll_out.delete_many({}) # empty the collection
        for set_index, document in enumerate(coll_in.find()):
            set_npas = []
            set_ = document['set_']
            for k in sorted(set_.keys()): # we need to always have a same ordering of the fields!
                logging.info(f"set_[{k}]:"+str(set_[k]))
                if type(set_[k][0]) is list:
                    floats = list(map(lambda v: list(map(lambda x:float(x),v)), set_[k]))
                else: # not something that is dummified: simple numerical value field
                    floats = list(map(lambda x: [float(x)], set_[k]))
                field_npa = np.array(floats)
                set_npas.append(field_npa)
            for curr in set_npas:
                logging.info("curr.shape:"+str(curr.shape))
            set_npa = np.hstack(set_npas)
            set_npa_serialized = serialize(set_npa)
            coll_out.insert_one({'set_index':set_index,'npa':set_npa_serialized})

def to_npa(ti):
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    for rel in chosen_rels:
        coll_in = db[rel+"_arrayfied"]
        coll_out = db['npas']
        rel_npas = []
        for document in coll_in.find():
            set_npa = pickle.loads(document['npa'])
            set_index = document['set_index']
            set_index_col = np.ones((set_npa.shape[0], 1))*set_index
            rel_npas.append(np.hstack([set_index_col, set_npa]))
        rel_npa = np.vstack(rel_npas)
        coll_out.remove({'rel':rel})
        coll_out.insert_one({
            'rel':rel,
            'npa':serialize(rel_npa)
        })

def to_tsets(ti):
    client = pymongo.MongoClient(MONGODB_CONN)
    db = client['learning_sets']
    for rel in chosen_rels:
        coll_in = db[rel+'_arrayfied']
        set_indices_results = coll_in.find({},{'set_index':1})
        set_indices = list(set(map( lambda document: document['set_index'],set_indices_results)))
        train_indices, test_indices = sklearn.model_selection.train_test_split(set_indices, train_size=0.75)
        coll_out = db['npas_tsets']
        train_npas = []
        test_npas = []
        for document in coll_in.find():
            set_npa = pickle.loads(document['npa'])
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
            'rel':rel,
            'train_npa':serialize(train_npa),
            'test_npa': serialize(test_npa)
        })

default_args = {
    'retries':120,
    'retry_delay':datetime.timedelta(minutes=3)
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