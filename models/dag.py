from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils import timezone
from airflow.utils.dates import days_ago
import datetime
import os
import sys

project_root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [project_root_path]+sys.path

import models.dspn_autoencoder

from common import relspecs, config
from models import run

def in_days(n):
    """
    Get a datetime object representing `n` days ago. By default the time is
    set to midnight.
    """
    today = timezone.utcnow()
    return today + datetime.timedelta(days=n)


def train_model(_rel, config_name, ti):
    """
    task function to train a Deep Set Prediction Network
    on a specific relation.
    :param _rel:
    :param ti:
    :return:
    """
    dynamic_config = {'rel_name': _rel.name}
    run.run(
        models.dspn_autoencoder.DSPNAE,
        config_name,
        dynamic_config=dynamic_config
    )

project_root_dir = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..'  # parent directory of models/
))
os.chdir(project_root_dir)

default_args = {
    'retries': 2,
    'retry_delay': datetime.timedelta(minutes=5),
    'schedule_interval': None
}
def make_dag(dag_name, config_name):
    with DAG(
            dag_name,
            description='trains DSPN models',
            tags=['train', 'dspn', 'sets', 'models'],
            default_args=default_args,
            schedule_interval=None,
            concurrency=config.models_dag_training_tasks_concurrency,  # maximum two models being trained at the same time
            max_active_runs=1,
            max_active_tasks=config.models_dag_training_tasks_concurrency
    ) as dag:
        days_interval = config.models_dag_days_interval
        for rel_i, rel in enumerate(relspecs.rels):

            # this time the tasks are shell commands
            train_cmd = f"cd {project_root_dir}; python3 models/dspn_autoencoder.py"\
                        + f" {config_name} --rel_name={rel.name}"

            t_train_model = BashOperator(
                task_id=f"train_dspn_model_{rel.name}",
                depends_on_past=False,
                bash_command=train_cmd,
                start_date=days_ago(2),
                dag=dag
            )

        # PythonOperator version:
        # t_train_model = PythonOperator(
        #    task_id=f"train_dsp_model_{rel.name}",
        #    python_callable=train_model,
        #    start_date=in_days((rel_i-1)*days_interval),
        #    op_kwargs={'rel':rel,'config_name':config_name}
        # )
    return dag

dag_train_models = make_dag('train_models_dag',config.models_dag_config_name)
dag_train_models_test = make_dag('train_models_dag_test',config.models_dag_config_name_test)
