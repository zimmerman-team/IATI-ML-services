#!/bin/env python3
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
import pymongo
import time

# since airflow's DAG modules are imported elsewhere (likely ~/airflow)
# we have to explicitly add the path of the parent directory to this module to python's path
path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path]+sys.path
from common import utils, relspecs, persistency, config
from preprocess import large_mp
rels = relspecs.rels.downloadable
specs = relspecs.specs.downloadable
logging.basicConfig(level=logging.DEBUG)

DATASTORE_ACTIVITY_URL = "https://datastore.iati.cloud/api/v2/activity"
DATASTORE_CODELIST_URL = "https://datastore.iati.cloud/api/codelists/{}/"

def extract_codelists(_specs):
    """
    :param _rels: iterable of relations (`relspects.Rel` objects)
    :return: set of codelist names used across the relations
    """
    ret = set()
    for spec in _specs:
        ret = ret.union(spec.codelists_names)
    return ret


def download(start, ti):
    """
    Airflow task: retrieve activities from back-end
    :param start: starting search result index of the page (`start` to `start+config.download_page_size`) being downloaded
    :param ti (str): task id
    :return: None
    """
    fl = ",".join(["iati_identifier"]+relspecs.rels.downloadable_prefixed_fields_names)
    params = {
        'q': "*:*",
        'fl': fl,
        'start': start,
        'rows': config.download_page_size
    }
    logging.info(f"requesting {DATASTORE_ACTIVITY_URL} with {params}")
    response = requests.get(DATASTORE_ACTIVITY_URL, params=params)
    logging.info(f"response.url:{response.url}")
    data = response.json()
    large_mp.send(ti, data)

def persist_activity_data(page, ti):
    db = persistency.mongo_db()
    coll_out = db['activity_data']

    # this large message is cleared in parse_sets_*,
    # which is subsequent to persist_activity_data_*
    data = large_mp.recv(ti, f"download_{page}")

    for activity in data['response']['docs']:
        activity_id = activity['iati_identifier']

        coll_out.insert_one({
            'activity_id': activity_id
        })

def parse_sets(page, ti):
    """
    Airflow task: parse downloaded page of activities
    :param page: index of the downloaded page to be parsed
    :param ti (str): task id
    :return: None
    """
    rels_vals = defaultdict(lambda: defaultdict(lambda: dict()))
    data = large_mp.recv(ti, f"download_{page}")
    for activity in data['response']['docs']:
        activity_id = activity['iati_identifier']
        # logging.info(f"processing activity {activity_id}")
        for rel in rels:
            rels_vals[rel.name][activity_id] = rel.extract_from_activity_data(activity)

    for rel, sets in rels_vals.items():
        remove = []
        for activity_id, fields in sets.items():
            logging.info('fields.keys'+str(fields.keys()))

            # now we check if all the fields in this set have the same
            # cardinality. They have to be because every item in the
            # set needs to have all the fields.

            lens = {}
            for k, v in fields.items():
                lens[k] = len(v)

            if len(set(lens.values())) != 1:

                logging.error(
                    "all fields need to have same amount of values"\
                    + f"(rel:{rel}, lens:{lens} activity_id:{activity_id}, fields:{fields}"
                )
                remove.append(activity_id)

        for activity_id in remove:
            # remove the invalid activity-set
            try:
                del rels_vals[rel][activity_id]
            except:
                pass # silently ignore the fact that activity_id is not in the data from that rel

    large_mp.send(ti, rels_vals)

def clear_activity_data(ti):
    db = persistency.mongo_db()
    coll = db['activity_data']

    # remove all data previously stored for this relation
    coll.delete_many({})
    coll.create_index([("activity_id", -1)])

def clear(rel, ti):
    db = persistency.mongo_db()
    coll = db[rel.name]

    # remove all data previously stored for this relation
    coll.delete_many({})
    coll.create_index([("activity_id", -1)])

def persist_sets(page, ti):
    """
    Airflow tasks: store previosly parsed page of activities in the mondodb
    :param page: index of the downloaded page to be stored
    :param ti (str): task id
    :return: None
    """
    db = persistency.mongo_db()
    data = large_mp.recv(ti, f'parse_sets_{page}')

    # FIXME: per-rel tasks
    for rel_name, sets in data.items():

        for activity_id, set_ in sets.items():

            # remove pre-existing set for this activity
            db[rel_name].delete_one({'activity_id': activity_id})

            db[rel_name].insert_one({
                'activity_id': activity_id,
                'set_': set_
            })

    large_mp.clear_recv(ti, f'parse_sets_{page}')
    large_mp.clear_recv(ti, f"download_{page}")

def codelists(ti):
    """
    Airflow task: download all codelists that are needed in order to encode category fields
    :param ti (str): task id
    :return: None
    """
    db = persistency.mongo_db()
    coll_out = db['codelists']
    coll_out.create_index([("name", -1)])
    for codelist_name in extract_codelists(specs):
        url = DATASTORE_CODELIST_URL.format(codelist_name)
        params = {'format': 'json'}
        response = requests.get(url, params=params)
        data = response.json()
        lst = []
        for curr in data:
            lst.append(curr['code'])
        coll_out.delete_many({'name': codelist_name})
        coll_out.insert({
            'name': codelist_name,
            'codelist': lst
        })


def get_set_size(set_):
    """
    Returns the maximum cardinality across multiple one2many relations sets
    :param set_: list of one2many relations items
    :return: the maximum cardinality across the one2many relations
    """
    size = 0
    for field_name, values in set_.items():
        size = max(len(values), size)
    return size


def encode(rel, ti):
    """
    Airflow task: encodes all fields into machine learning-friendly
        (numerical vectors)
    :param rel: the relations that needs encoding
    :param ti: task id
    :return: None
    """
    db = persistency.mongo_db()
    coll_in = db[rel.name]
    coll_out = db[rel.name + "_encoded"]

    # remove existing data in the collection
    coll_out.delete_many({})
    coll_out.create_index([("activity_id", -1)])

    for document in coll_in.find(no_cursor_timeout=True):
        document = dict(document)  # copy
        set_ = document['set_']
        set_size = get_set_size(set_)

        # how much time does each item require to be encoded
        start = time.time()

        for field in rel.fields:
            encodable = set_.get(field.name, [])
            tmp = field.encode(encodable, set_size)
            set_[field.name] = tmp

        end = time.time()
        encoding_time = end-start
        document['encoding_time'] = encoding_time
        del document['_id']
        lens = list(map(lambda fld: len(set_[fld.name]), rel.fields))
        if len(set(lens)) > 1:
            msg = "lens " + str(lens) + " for fields " + str(rel.fields_names)
            logging.info(msg)
            logging.info(document)
            raise Exception(msg)
        coll_out.delete_one({'activity_id': document['activity_id']})  # remove pre-existing set for this activity

        try:
            coll_out.insert_one(document)
        except pymongo.errors.DocumentTooLarge as e:
            logging.info(f"document[activity_id]: {document['activity_id']}")
            for field in rel.fields:
                logging.info(f"{field.name} len: {len(set_[field.name])}")
            raise Exception(f"cannot insert document into relation {rel.name} because {str(e)}")

def arrayfy(rel, ti):
    """
    Gets all the encoded fields and concatenates them into a dataset arrays
    indexed by set index
    :param rel: relation to be encoded
    :param ti: task id
    :return: None
    """
    db = persistency.mongo_db()
    coll_in = db[rel.name+"_encoded"]
    coll_out = db[rel.name+"_arrayfied"]
    coll_out.delete_many({})  # empty the collection
    coll_out.create_index([("activity_id", -1)])
    for set_index, document in enumerate(coll_in.find()):
        set_npas = []
        set_ = document['set_']
        keys = rel.fields_names
        for k in keys:  # we need to always have a same ordering of the fields!
            if len(set_[k]) > 0 and type(set_[k][0]) is list:
                floats = list(map(lambda v: list(map(lambda x: float(x), v)), set_[k]))
            else:  # not something that is dummified: simple numerical value field
                floats = list(map(lambda x: [float(x)], set_[k]))
            field_npa = np.array(floats)
            set_npas.append(field_npa)
        if len(set(map(lambda curr: curr.shape[0], set_npas))) > 1:
            logging.info("keys:" + str(keys))
            logging.info("set_npas shapes:"+str([curr.shape for curr in set_npas]))
        set_npa = np.hstack(set_npas)
        set_npa_serialized = utils.serialize(set_npa)
        coll_out.insert_one({'set_index': set_index, 'npa': set_npa_serialized})


def to_npa(rel, ti):
    """
    Airflow task: concatenates all set-indexed arrays into one numpy array dataset
    :param rel: relation whose data is going to be converted into a dataset
    :param ti: task id
    :return: None
    """
    db = persistency.mongo_db()
    coll_in = db[rel.name+"_arrayfied"]
    coll_out = db['npas']
    coll_out.create_index([("rel", -1)])
    coll_out.create_index([("creation_date", -1)])
    coll_out.create_index([("npa_file_id", -1)])
    rel_npas = []
    for document in coll_in.find():
        set_npa = utils.deserialize(document['npa'])
        set_index = document['set_index']
        set_index_col = np.ones((set_npa.shape[0], 1))*set_index
        rel_npas.append(np.hstack([set_index_col, set_npa]))
    rel_npa = np.vstack(rel_npas)
    coll_out.delete_many({'rel': rel.name})

    coll_out.insert_one({
        'rel': rel.name,
        'creation_date': utils.strnow_iso(),
        'npa_file_id': persistency.save_npa(f"{rel.name}", rel_npa),
        'npa_rows': rel_npa.shape[0],
        'npa_cols': rel_npa.shape[1]
    })


def to_tsets(rel, ti):
    """
    Concatenates all set-indexed arrays into split array datasets, one for training
    and the other for validation/test
    :param rel: relation whose training/validation/test sets are being created
    :param ti: task id
    :return: None
    """
    db = persistency.mongo_db()
    coll_in = db[rel.name+'_arrayfied']
    set_indices_results = coll_in.find({}, {'set_index': 1})
    set_indices = list(set(map(lambda document: document['set_index'], set_indices_results)))
    train_indices, test_indices = sklearn.model_selection.train_test_split(set_indices, train_size=0.90)
    coll_out = db['npas_tsets']
    coll_out.create_index([("rel", -1)])
    coll_out.create_index([("creation_date", -1)])
    coll_out.create_index([("train_npa_file_id", -1)])
    coll_out.create_index([("test_npa_file_id", -1)])
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

    coll_out.delete_many({'rel': rel.name})
    coll_out.insert_one({
        'rel': rel.name,
        'creation_time': utils.strnow_iso(),
        'train_npa_file_id': persistency.save_npa(f"{rel.name}_train", train_npa),
        'test_npa_file_id': persistency.save_npa(f"{rel.name}_test", test_npa),
        'train_npa_rows': train_npa.shape[0],
        'train_npa_cols': train_npa.shape[1],
        'test_npa_rows': test_npa.shape[0],
        'test_npa_cols': test_npa.shape[1]
    })


default_args = {
    'retries': 2,
    'retry_delay': datetime.timedelta(minutes=1),
    'schedule_interval': None
}

with DAG(
    'download_and_preprocess_sets',
    description='Downloads sets data from IATI.cloud',
    tags=['download', 'preprocess', 'sets'],
    default_args=default_args,
    schedule_interval=None,
    max_active_runs=1,
) as dag:

    pages = list(range(config.download_max_pages))

    t_codelists = PythonOperator(
        task_id="codelists",
        python_callable=codelists,
        start_date=days_ago(2),
        op_kwargs={}
    )

    t_clear_activity_data = PythonOperator(
        task_id=f"clear_activity_data",
        python_callable=clear_activity_data,
        start_date=days_ago(2)
    )
    t_clear_sets = {}
    for rel in rels:
        t_clear_sets[rel.name] = PythonOperator(
            task_id=f"clear_sets_{rel.name}",
            python_callable=clear,
            start_date=days_ago(2),
            op_kwargs={'rel': rel}
        )

    t_persist_sets = {}
    for page in pages:
        start = page*config.download_page_size
        t_download = PythonOperator(
            task_id=f"download_{page}",
            python_callable=download,
            start_date=days_ago(2),
            op_kwargs={'start': start}
        )

        t_persist_activity_data = PythonOperator(
            task_id=f"persist_activity_data_{page}",
            python_callable=persist_activity_data,
            start_date=days_ago(2),
            op_kwargs={'page': page}
        )

        t_parse_sets = PythonOperator(
            task_id=f"parse_sets_{page}",
            python_callable=parse_sets,
            start_date=days_ago(2),
            op_kwargs={'page': page}
        )

        t_persist_sets[page] = PythonOperator(
            task_id=f"persist_sets_{page}",
            python_callable=persist_sets,
            start_date=days_ago(2),
            op_kwargs={'page': page}
        )
        for rel in rels:
            t_clear_activity_data >> t_clear_sets[rel.name]
            t_clear_sets[rel.name] >> t_download
        t_download >> t_parse_sets >> t_persist_activity_data
        t_persist_activity_data >> t_persist_sets[page]

    for rel in rels:
        t_to_npa = PythonOperator(
            task_id=f"to_npa_{rel.name}",
            python_callable=to_npa,
            start_date=days_ago(2),
            op_kwargs={'rel': rel}
        )

        t_to_tsets = PythonOperator(
            task_id=f"to_tsets_{rel.name}",
            python_callable=to_tsets,
            start_date=days_ago(2),
            op_kwargs={'rel': rel}
        )
        t_encode = PythonOperator(
            task_id=f"encode_{rel.name}",
            python_callable=encode,
            start_date=days_ago(2),
            op_kwargs={'rel': rel}
        )

        t_arrayfy = PythonOperator(
            task_id=f"arrayfy_{rel.name}",
            python_callable=arrayfy,
            start_date=days_ago(2),
            op_kwargs={'rel': rel}
        )

        for page in pages:
            t_persist_sets[page] >> t_encode
        t_codelists >> t_encode >> t_arrayfy
        t_arrayfy >> t_to_npa
        t_arrayfy >> t_to_tsets

