from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils import timezone
from airflow.utils.dates import days_ago
import datetime
import os
import sys

project_root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+"/..")
sys.path = [project_root_path]+sys.path

import models
from models import model_class_loader
import models.dspn_autoencoder

from common import relspecs, config

def in_days(n):
    """
    Get a datetime object representing `n` days ago. By default the time is
    set to midnight.
    """
    today = timezone.utcnow()
    return today + datetime.timedelta(days=n)


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

def make_dag_rels(dag_name, config_name, modelname):
    """
    models that train on rels
    :param dag_name: name of the dag
    :param config_name: name of the configuration file (without path and extension)
    :param modelname: name of the model (module containing it, actually)
    :return:
    """
    with DAG(
            dag_name,
            description=f'trains {modelname} models to be trained on relation sets',
            tags=['train', 'rels', 'sets', 'models'],
            default_args=default_args,
            schedule_interval=None,
            concurrency=config.models_dag_training_tasks_concurrency,  # maximum two models being trained at the same time
            max_active_runs=1,
            max_active_tasks=config.models_dag_training_tasks_concurrency
    ) as dag:

        for rel_i, rel in enumerate(relspecs.rels):

            # this time the tasks are shell commands
            train_cmd = f"cd {project_root_dir}; python3 models/{modelname}.py"\
                        + f" {config_name} --rel_name={rel.name}"

            t_train_model = BashOperator(
                task_id=f"train_{modelname}_model_{rel.name}",
                depends_on_past=False,
                bash_command=train_cmd,
                start_date=days_ago(2),
                dag=dag
            )

    return dag

def make_dag_activity(dag_name, config_name, modelname):
    """
    models that train on rels
    :param dag_name: name of the dag
    :param config_name: name of the configuration file (without path and extension)
    :param modelname: name of the model (module containing it, actually)
    :return:
    """
    with DAG(
            dag_name,
            description=f'trains {modelname} models to be trained on relation sets',
            tags=['train', 'activity', 'sets', 'models'],
            default_args=default_args,
            schedule_interval=None,
            concurrency=config.models_dag_training_tasks_concurrency,  # maximum two models being trained at the same time
            max_active_runs=1,
            max_active_tasks=config.models_dag_training_tasks_concurrency
    ) as dag:


        # this time the tasks are shell commands
        train_cmd = f"cd {project_root_dir}; python3 models/{modelname}.py" \
                    + f" {config_name}"

        t_train_model = BashOperator(
            task_id=f"train_{modelname}_model_activity",
            depends_on_past=False,
            bash_command=train_cmd,
            start_date=days_ago(2),
            dag=dag
        )

    return dag

def populate_module_with_dags():
    thismodule = sys.modules[__name__]
    for modelname in model_class_loader.all_modelnames():
        args = [f"train_models_dag_{modelname}",config.models_dag_config_name, modelname]
        if model_class_loader.does_model_train_on(modelname, 'rels'):
            train_models_dag = make_dag_rels(*args)
        if model_class_loader.does_model_train_on(modelname, 'activity'):
            train_models_dag = make_dag_activity(*args)
        setattr(thismodule, "train_models_dag_"+modelname, train_models_dag)

populate_module_with_dags()