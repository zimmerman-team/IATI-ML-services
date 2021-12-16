from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import datetime

default_args = {
    'retries':2,
    'retry_delay':datetime.timedelta(minutes=1)
}

def test_task(ti):
    """
    Just a test task that does nothing.
    :param ti:
    :return:
    """
    return True

with DAG(
        'aaa_test_dag__',
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