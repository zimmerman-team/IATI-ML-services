import datetime
import sys
import os
import collections
import logging
import sklearn

# since airflow's DAG modules are imported elsewhere (likely ~/airflow)
# we have to explicitly add the path of the parent directory to this module to python's path

from common import relspecs, dataset_persistency, utils, config
from preprocess import vectorize_activity, download_sets_dag

default_args = {
    'retries': 2,
    'retry_delay': datetime.timedelta(minutes=1),
    'schedule_interval': None
}

rels = relspecs.rels.downloadable

activity_vectorizer = vectorize_activity.ActivityVectorizer(config.model_modulename_rels)


def clear(ti):
    """
    removes the previous yields of this DAG
    :param ti:
    :return:
    """
    db = dataset_persistency.mongo_db(caching=False)
    db['activity_encoded_sets'].delete_many({})
    db['activity_encoded_sets'].create_index([("activity_id", -1)])
    db['activity_vectors'].delete_many({})
    db['activity_vectors'].create_index([("activity_id", -1)])


def collect(ti):
    """
    gets all the encoded sets and groups them by activity_id,
    stores all this in an ad-hoc collection
    :param ti:
    :return:
    """
    db = dataset_persistency.mongo_db()

    # open all necessary collections
    coll_sets = {}

    # this data was previously downloaded by a different airflow DAG and task
    # (download_sets_dag.py/persist)
    coll_activity_ids = db['activity_ids']
    coll_out = db['activity_encoded_sets']

    # open all collections for every rel
    for rel in rels:
        coll_sets[rel.name] = db[rel.name + "_encoded"]

    activity_ids_docs = coll_activity_ids.find({}, {'activity_id': 1})
    for activity_ids_doc in activity_ids_docs:
        encoded_sets = collections.OrderedDict()
        activity_id = activity_ids_doc['activity_id']
        for rel in rels:
            # get the data from a specific rel
            doc = coll_sets[rel.name].find_one({'activity_id': activity_id})
            if doc is not None:
                encoded_sets[rel.name] = doc['data']
            else:
                # eventually this will have to cause the use of an empty set numpy array
                encoded_sets[rel.name] = None
        new_document = {
            'activity_id': activity_id,
            'encoded_sets': encoded_sets
        }
        coll_out.insert_one(new_document)


def vectorize(ti):
    """
    takes all relation sets belonging to an activity and generates
    their latent code using the respective previously-trained Deep Set AutoEncoder models
    (this aspect is actually implemented in ActivityVectorizer).
    Also, these codes are being stored in an ad-hoc collection.
    :param ti:
    :return:
    """
    global activity_vectorizer
    db = dataset_persistency.mongo_db()
    coll_in_sets = db['activity_encoded_sets']
    coll_in_activity_without_rels = db['activity_encoded']
    coll_out = db['activity_with_rels_arrayfied']
    for datapoint_index, activity_sets_document in enumerate(coll_in_sets.find({})):
        activity_sets = activity_sets_document['encoded_sets']
        activity_without_rels_fields = coll_in_activity_without_rels.find_one({
            'activity_id':activity_sets_document['activity_id']
        })
        if activity_without_rels_fields is None:
            # no fixed-length fields data for this activity. Skip it.
            continue

        activity_without_rels_fields_npa = utils.create_set_npa(
            relspecs.activity_without_rels,
            activity_without_rels_fields['data'] # FIXME: rename in 'npa' as my convention?
        )
        activity_vector = activity_vectorizer.process(activity_sets, activity_without_rels_fields_npa)
        #logging.debug(f"activity_vector {activity_vector}")
        activity_vector_serialized = utils.serialize(activity_vector)
        new_document = {
            'activity_id': activity_sets_document['activity_id'],
            'npa': activity_vector_serialized,
            'set_index': datapoint_index #FIXME: rename this backward-compatibility `set_index`
        }
        coll_out.insert_one(new_document)


def setup_dag():
    # putting airflow imports inside this function
    # otherwise they interefere with alternative usages of the logging module
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago

    global activity_vectorizer

    with DAG(
            'vectorize_activities',
            description='Vectorize activities',
            tags=['vectorize', 'preprocess', 'activities'],
            default_args=default_args,
            schedule_interval=None
    ) as dag:

        t_clear = PythonOperator(
            task_id="clear",
            python_callable=clear,
            start_date=days_ago(2)
        )

        t_collect = PythonOperator(
            task_id="collect",
            python_callable=collect,
            start_date=days_ago(2)
        )

        t_vectorize = PythonOperator(
            task_id="vectorize",
            python_callable=vectorize,
            start_date=days_ago(2),
            pool="npas_intensive",
            pool_slots=1
        )

        t_to_tsets = PythonOperator(
            task_id="to_tsets",
            python_callable=download_sets_dag.to_tsets, # FIXME: move to_tsets to something more common?
            start_date=days_ago(2),
            op_kwargs={'spec': relspecs.activity_with_rels(activity_vectorizer.rel_latent_dim) },
            pool="npas_intensive",
            pool_slots=1
        )
        t_clear >> t_collect >> t_vectorize >> t_to_tsets

    thismodule = sys.modules[__name__]
    setattr(thismodule, "dag", dag)

def test():
    utils.setup_main()
    vectorize(None)

if __name__ == "__main__":
    test()
else:
    setup_dag()