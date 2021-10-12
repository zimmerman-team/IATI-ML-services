from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import datetime

from common import utils
from common import relspecs
from common import persistency
from preprocess import large_mp

default_args = {
    'retries':2,
    'retry_delay':datetime.timedelta(minutes=1)
}

def test_task(ti):
    return True

with DAG(
        'aaa_test_dag_2__',
        description='Test Dag',
        tags=['test','dag'],
        default_args=default_args
) as dag:
    t_test = PythonOperator(
        task_id='test_task',
        python_callable=test_task,
        start_date=days_ago(2),
        op_kwargs={}
    )