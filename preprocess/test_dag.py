from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import datetime
import tqdm
import time

default_args = {
    'retries': 2,
    'retry_delay': datetime.timedelta(minutes=1)
}


def test_task(ti):
    """
    Just a test task that does nothing.
    :param ti:
    :return:
    """
    pass

def tqdm_sleep_task(ti):
    for i in tqdm.tqdm(range(3600)):
        time.sleep(1)

def tqdm_sleep_print_task(ti):
    for i in tqdm.tqdm(range(3600)):
        time.sleep(1)
        print(i)

with DAG(
        'aaa_test_dag__',
        description='Test Dag',
        tags=['test', 'dag'],
        default_args=default_args
) as dag:
    t_test = PythonOperator(
        task_id='test_task',
        python_callable=test_task,
        start_date=days_ago(2),
        op_kwargs={}
    )

    t_tqdm_sleep = PythonOperator(
        task_id='tqdm_sleep_task',
        python_callable=tqdm_sleep_task,
        start_date=days_ago(2),
        op_kwargs={}
    )

    t_tqdm_sleep_print = PythonOperator(
        task_id='tqdm_sleep_print_task',
        python_callable=tqdm_sleep_print_task,
        start_date=days_ago(2),
        op_kwargs={}
    )
