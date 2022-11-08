from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import datetime
import tqdm
import time
import logging
import os
import sys

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [path]+sys.path

from configurator import config

default_args = {
    'retries': 2,
    'retry_delay': datetime.timedelta(minutes=1)
}

logging_level = getattr(logging, config.log_level, logging.INFO)

logging.basicConfig(
    level=logging_level
)

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

def logging_info(ti):
    global logging_level
    print('log_level',logging.getLevelName(logging_level))
    logging.info("if you see this message then logging.info messages are within the logging level")
    print("end of task.")


def logging_debug(ti):
    global logging_level
    print('log_level',logging.getLevelName(logging_level))
    logging.debug("if you see this message then logging.debug messages are within the logging level")
    print("end of task.")

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

    t_logging_info = PythonOperator(
        task_id='logging_info',
        python_callable=logging_info,
        start_date=days_ago(2),
        op_kwargs={}
    )

    t_logging_debug = PythonOperator(
        task_id='logging_debug',
        python_callable=logging_debug,
        start_date=days_ago(2),
        op_kwargs={}
    )
