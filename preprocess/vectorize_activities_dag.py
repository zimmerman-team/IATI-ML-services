from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import datetime
import sys
import os
import collections

# since airflow's DAG modules are imported elsewhere (likely ~/airflow)
# we have to explicitly add the path of the parent directory to this module to python's path
path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path]+sys.path

from common import relspecs, persistency, utils
from preprocess import vectorize_activity

default_args = {
    'retries': 2,
    'retry_delay': datetime.timedelta(minutes=1),
    'schedule_interval': None
}

rels = relspecs.rels.downloadable

def clear(ti):
    """
    removes the previous yields of this DAG
    :param ti:
    :return:
    """
    db = persistency.mongo_db()
    db['activity_data_encoded'].remove()
    db['activity_vectors'].remove()

def collect(ti):
    """
    gets all the encoded sets and groups them by activity_id,
    stores all this in an ad-hoc collection
    :param ti:
    :return:
    """
    db = persistency.mongo_db()

    # open all necessary collections
    coll_sets = {}
    coll_activity = db['activity_data']
    coll_out = db['activity_data_encoded']
    for rel in relspecs:
        coll_sets[rel.name] = db[rel.name + "_encoded"]

    activity_docs = coll_activity.find({}, {'activity_id':1})
    activity_sets = collections.OrderedDict()
    for activity_doc in activity_docs:
        encoded_sets = collections.OrderedDict()
        activity_id = activity_doc['activity_id']
        for rel in relspecs:
            encoded_sets[rel.name] = coll_sets[rel.name].find({'activity_id': activity_id})
        new_document = {
            'activity_id':activity_id,
            'encoded_sets':encoded_sets
        }
        coll_out.insert_one(new_document)

def vectorize(ti):
    """
    takes all relation sets belonging to an activity and generates
    their latent code using the respective previously-trained Set AutoEncoder models
    (this aspect is actually implemented in ActivityVectorizer).
    Also, these codes are being stored in an ad-hoc collection.
    :param ti:
    :return:
    """
    db = persistency.mongo_db()
    coll_in = db['activity_data_encoded']
    coll_out = db['activity_vectors']
    activity_vectorizer = vectorize_activity.ActivityVectorizer()
    for input_document in coll_in.find({}):
        activity_vector = activity_vectorizer.process(input_document)
        activity_vector_serialized = utils.serialize(activity_vector)
        new_document = {
            'activity_id': input_document['activity_id'],
            'activity_vector': activity_vector_serialized
        }
        coll_out.insert_one(new_document)

with DAG(
        'vectorize_activies',
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

    )

    t_clear >> t_collect >> t_vectorize
