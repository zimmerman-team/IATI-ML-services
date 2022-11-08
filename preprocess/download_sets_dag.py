#!/bin/env python3
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
import logging
import datetime
import numpy as np
from collections import defaultdict
import sklearn.model_selection
import pymongo
import time

# since airflow's DAG modules are imported elsewhere (likely ~/airflow)
# we have to explicitly add the path of the parent directory to this module to python's path

from common import utils, specs_config, dataset_persistency
from configurator import config
from large_mp import large_mp

rels = specs_config.rels.downloadable
specs = specs_config.specs.downloadable
logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO))

DATASTORE_ACTIVITY_URL = "https://datastore.iati.cloud/api/v2/activity"
DATASTORE_CODELIST_URL = "https://datastore.iati.cloud/api/codelists/{}/"


def extract_codelists(_specs):
    """
    :param _specs: iterable of relations (`relspects.Rel` objects)
    :return: set of codelist names used across the relations
    """
    ret = set()
    for spec in _specs:
        ret = ret.union(spec.codelists_names)
    return ret


def download(start, ti):
    """
    Airflow task: retrieve activities from back-end
    :param _start: starting search result index of the page (`start` to `start+config.download_page_size`)
                   being downloaded
    :param ti: task id (string)
    :return: None
    """
    fl = ",".join(["iati_identifier"] + specs_config.specs.downloadable_prefixed_fields_names)
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


def persist_activity_ids(page, ti):
    """
    Collects the activity ids from the downloaded page, and
    stores them in a dedicated collection (activity_ids) in
    the mongo db.
    :param page:
    :param ti:
    :return:
    """
    with dataset_persistency.MongoDB() as db:
        coll_out = db['activity_ids']

        # this large message is cleared in parse_*,
        # which is subsequent to persist_activity_data_*
        data = large_mp.recv(ti, f"download_{page}")

        for activity in data['response']['docs']:
            activity_id = activity['iati_identifier']

            coll_out.insert_one({
                'activity_id': activity_id,
            })


def parse(page, ti):
    """
    Airflow task: parse downloaded page of activities
    :param page: index of the downloaded page to be parsed
    :param ti: task id (string)
    :return: None
    """
    specs_vals = defaultdict(lambda: defaultdict(lambda: dict()))
    data = large_mp.recv(ti, f"download_{page}")
    for activity in data['response']['docs']:
        activity_id = activity['iati_identifier']
        # logging.info(f"processing activity {activity_id}")
        for spec in specs:
            specs_vals[spec.name][activity_id] = spec.extract_from_raw_data(activity)

    for spec_name, spec_data in specs_vals.items():

        if spec_name == "activity_without_rels":
            # no need to check for field cardinality
            # in single-cardinality fields of activity
            # logging.info(f"activity data:{spec_data}")
            continue

        remove = []
        for activity_id, fields in spec_data.items():
            # logging.info('fields.keys'+str(fields.keys()))

            # now we check if all the fields in this set have the same
            # cardinality. They have to be because every item in the
            # set needs to have all the fields.

            lens = {}
            for k, v in fields.items():
                lens[k] = len(v)

            if len(set(lens.values())) != 1:

                logging.error(
                    "all fields need to have same amount of values"
                    + f"(spec:{spec_name}, lens:{lens} activity_id:{activity_id}, fields:{fields}"
                )
                remove.append(activity_id)

        for activity_id in remove:
            # remove the invalid activity-set
            try:
                del specs_vals[spec_name][activity_id]
            except:
                pass  # silently ignore the fact that activity_id is not in the data from that rel

    large_mp.send(ti, specs_vals)


def clear_activity_ids(ti):
    """
    Removes all previously-collected activities' ids
    :param ti:
    :return:
    """
    with dataset_persistency.MongoDB() as db: # FIXME: to decorator?
        coll = db['activity_ids']

        # remove all data previously stored for this relation
        coll.delete_many({})
        coll.create_index([("activity_id", -1)])


def clear(spec, ti):
    """
    Delete contents of the preliminary persistent storage (to mongodb)
    of either the non-relation activity field values, or relation fields.
    :param spec: either Activity or Rel
    :param ti:
    :return:
    """
    with dataset_persistency.MongoDB() as db:
        coll = db[spec.name]

        # remove all data previously stored for this relation
        coll.delete_many({})
        coll.create_index([("activity_id", -1)])


def persist(page, spec_name, ti):
    """
    Airflow tasks: store previosly parsed page of activities in the mondodb
    :param page: index of the downloaded page to be stored
    :param ti: task id (string)
    :return: None
    """
    with dataset_persistency.MongoDB() as db:
        data = large_mp.recv(ti, f'parse_{page}')

        if spec_name in data.keys():
            spec_data = data[spec_name]
            for activity_id, activity_data in spec_data.items():
                logging.info(f"data for {spec_name}:{activity_id}:{activity_data}")

                # remove pre-existing set for this activity
                db[spec_name].delete_one({'activity_id': activity_id})

                db[spec_name].insert_one({
                    'activity_id': activity_id,
                    'data': activity_data
                })

def clear_page(page, ti):
    large_mp.clear_recv(ti, f'parse_{page}')
    large_mp.clear_recv(ti, f"download_{page}")

def codelists(ti):
    """
    Airflow task: download all codelists that are needed in order to encode category fields
    :param ti: task id (string)
    :return: None
    """
    with dataset_persistency.MongoDB() as db:
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
            coll_out.insert_one({
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


def encode(spec, ti):
    """
    Airflow task: encodes all fields into machine learning-friendly
        (numerical vectors)
    :param spec: the relations that needs encoding
    :param ti: task id
    :return: None
    """
    with dataset_persistency.MongoDB() as db:
        coll_in = db[spec.name]
        coll_out = db[spec.name + "_encoded"]

        # remove existing data in the collection
        coll_out.delete_many({})
        coll_out.create_index([("activity_id", -1)])

        for document in coll_in.find(no_cursor_timeout=True):
            document = dict(document)  # copy

            data = document['data']
            if type(spec) is specs_config.Activity:
                data_amount = 1
            else:
                data_amount = get_set_size(data)

            # how much time does each item require to be encoded
            _start = time.time()

            for field in spec.fields:
                encodable = data.get(field.name, [])

                if type(spec) is specs_config.Activity:
                    # if it's an activity, there is only one value
                    # per field
                    encodable = [encodable]

                tmp = field.encode(encodable, data_amount)
                data[field.name] = tmp

            end = time.time()
            encoding_time = end-_start
            document['encoding_time'] = encoding_time
            del document['_id']
            lens = list(map(lambda fld: len(data[fld.name]), spec.fields))
            if len(set(lens)) > 1:
                msg = "lens " + str(lens) + " for fields " + str(spec.fields_names)
                logging.info(msg)
                logging.info(document)
                raise Exception(msg)
            coll_out.delete_one({'activity_id': document['activity_id']})  # remove pre-existing set for this activity

            try:
                coll_out.insert_one(document)
            except pymongo.errors.DocumentTooLarge as e:
                logging.info(f"document[activity_id]: {document['activity_id']}")
                for field in spec.fields:
                    logging.info(f"{field.name} len: {len(data[field.name])}")
                raise Exception(f"cannot insert document into spec {spec.name} because {str(e)}")


def arrayfy(spec, ti):
    """
    Gets all the encoded fields and concatenates them into a dataset arrays
    indexed by set index
    :param spec: spec to be encoded
    :param ti: task id
    :return: None
    """
    with dataset_persistency.MongoDB() as db:
        coll_in = db[spec.name+"_encoded"]
        coll_out = db[spec.name+"_arrayfied"]
        coll_out.delete_many({})  # empty the collection
        coll_out.create_index([("activity_id", -1)])
        for set_index, document in enumerate(coll_in.find()):
            data = document['data']
            set_npa = utils.create_set_npa(spec, data)
            set_npa_serialized = utils.serialize(set_npa)
            # FIXME: rename `set_index` with something more generic to account for simple datapoint indexes
            coll_out.insert_one({
                'set_index': set_index,
                'npa': set_npa_serialized
            })


def to_npa(spec, ti):
    """
    Airflow task: concatenates all set-indexed arrays into one numpy array dataset
    :param spec: relation whose data is going to be converted into a dataset
    :param ti: task id
    :return: None
    """
    with dataset_persistency.MongoDB() as db:
        coll_in = db[spec.name + "_arrayfied"]
        coll_out = db['npas']
        coll_out.create_index([("spec", -1)])
        coll_out.create_index([("creation_date", -1)])
        coll_out.create_index([("npa_file_id", -1)])
        spec_npas = []

        cursor = coll_in.find()
        if spec.limit:
            cursor = cursor.limit(spec.limit)

        for document in cursor:
            set_npa = utils.deserialize(document['npa'])

            # put the set index as first column in the numpy array
            set_index = document['set_index']
            set_index_col = np.ones((set_npa.shape[0], 1))*set_index

            spec_npas.append(np.hstack([set_index_col, set_npa]))

        spec_npa = np.vstack(spec_npas)
        coll_out.delete_many({'spec': spec.name})

        coll_out.insert_one({
            'spec': spec.name,
            'creation_date': utils.strnow_iso(),
            'npa_file_id': dataset_persistency.save_npa(f"{spec.name}", spec_npa),
            'npa_rows': spec_npa.shape[0],
            'npa_cols': spec_npa.shape[1]
        })


def to_splits(spec, ti):
    """
    Concatenates all set-indexed arrays into split array datasets, one for training
    and the other for validation/test
    :param spec: relation whose training/validation/test sets are being created
    :param ti: task id
    :return: None
    """
    with dataset_persistency.MongoDB() as db:
        coll_in = db[spec.name + '_arrayfied']
        set_indices_results = coll_in.find({}, {'set_index': 1})
        set_indices = list(set(map(lambda document: document['set_index'], set_indices_results)))
        train_indices, test_indices = sklearn.model_selection.train_test_split(set_indices, train_size=config.train_fraction)
        train_indices = set(train_indices)
        test_indices = set(test_indices)
        coll_out = db['npas_splits']
        coll_out.create_index([("spec", -1)])
        coll_out.create_index([("creation_date", -1)])
        coll_out.create_index([("train_npa_file_id", -1)])
        coll_out.create_index([("test_npa_file_id", -1)])
        train_npas = []
        test_npas = []
        cursor = coll_in.find()
        if spec.limit:
            cursor = cursor.limit(spec.limit)
        for document in cursor:
            set_npa = utils.deserialize(document['npa'])
            set_index = document['set_index']
            set_index_col = np.ones((set_npa.shape[0], 1))*set_index

            # NOTE: there will be a set_index even for relspecs.Activity
            #       data, even if they are not sets!
            #       This set_index is actually just a progressive datapoint index within
            #       its mongodb collection
            npa = np.hstack([set_index_col, set_npa])

            if set_index in train_indices:
                train_npas.append(npa)
            elif set_index in test_indices:
                test_npas.append(npa)

        train_npa = np.vstack(train_npas)
        test_npa = np.vstack(test_npas)

        coll_out.delete_many({'spec': spec.name})
        coll_out.insert_one({
            'spec': spec.name,
            'creation_time': utils.strnow_iso(),
            'train_npa_file_id': dataset_persistency.save_npa(f"{spec.name}_train", train_npa),
            'test_npa_file_id': dataset_persistency.save_npa(f"{spec.name}_test", test_npa),
            'train_npa_rows': train_npa.shape[0],
            'train_npa_cols': train_npa.shape[1],
            'test_npa_rows': test_npa.shape[0],
            'test_npa_cols': test_npa.shape[1]
        })


default_args = {
    'retries': 5,
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
    concurrency=int(config.airflow_concurrency),
    max_active_tasks=int(config.airflow_concurrency)
) as dag:

    pages = list(range(config.download_max_pages))

    t_codelists = PythonOperator(
        task_id="codelists",
        python_callable=codelists,
        start_date=days_ago(2),
        op_kwargs={}
    )

    t_clear_activity_ids = PythonOperator(
        task_id=f"clear_activity_ids",
        python_callable=clear_activity_ids,
        start_date=days_ago(2)
    )

    t_clear = {}
    for _spec in specs:
        t_clear[_spec.name] = PythonOperator(
            task_id=f"clear_{_spec.name}",
            python_callable=clear,
            start_date=days_ago(2),
            op_kwargs={'spec': _spec}
        )

    t_persist = {}
    t_clear_page = {}
    for _page in pages:
        start = _page * config.download_page_size
        t_download = PythonOperator(
            task_id=f"download_{_page}",
            python_callable=download,
            start_date=days_ago(2),
            op_kwargs={'start': start}
        )

        t_persist_activity_ids = PythonOperator(
            task_id=f"persist_activity_ids_{_page}",
            python_callable=persist_activity_ids,
            start_date=days_ago(2),
            op_kwargs={'page': _page}
        )

        t_parse = PythonOperator(
            task_id=f"parse_{_page}",
            python_callable=parse,
            start_date=days_ago(2),
            op_kwargs={'page': _page}
        )

        t_clear_page[_page] = PythonOperator(
            task_id=f"clear_page_{_page}",
            python_callable=clear_page,
            start_date=days_ago(2),
            op_kwargs={'page': _page}
        )

        for _spec in specs:
            if _page not in t_persist.keys():
                t_persist[_page] = {}
            t_persist[_page][_spec.name] = PythonOperator(
                task_id=f"persist_{_page}_{_spec.name}",
                python_callable=persist,
                start_date=days_ago(2),
                op_kwargs={'page': _page,'spec_name': _spec.name}
            )
            t_clear_activity_ids >> t_clear[_spec.name]
            t_clear[_spec.name] >> t_download
            t_persist_activity_ids >> t_persist[_page][_spec.name]
        t_download >> t_parse >> t_persist_activity_ids

    for _spec in specs:
        t_encode = PythonOperator(
            task_id=f"encode_{_spec.name}",
            python_callable=encode,
            start_date=days_ago(2),
            op_kwargs={'spec': _spec}
        )

        t_to_npa = PythonOperator(
            task_id=f"to_npa_{_spec.name}",
            python_callable=to_npa,
            start_date=days_ago(2),
            op_kwargs={'spec': _spec},
            pool="npas_intensive",
            pool_slots=1
        )

        t_to_splits = PythonOperator(
            task_id=f"to_splits_{_spec.name}",
            python_callable=to_splits,
            start_date=days_ago(2),
            op_kwargs={'spec': _spec},
            pool="npas_intensive",
            pool_slots=1
        )

        t_arrayfy = PythonOperator(
            task_id=f"arrayfy_{_spec.name}",
            python_callable=arrayfy,
            start_date=days_ago(2),
            op_kwargs={'spec': _spec}
        )

        for _page in pages:
            t_persist[_page][_spec.name] >> t_encode
            t_persist[_page][_spec.name] >> t_clear_page[_page]
        t_codelists >> t_encode >> t_arrayfy
        t_arrayfy >> t_to_npa
        t_arrayfy >> t_to_splits

