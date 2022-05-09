""" add additional DAGs folders """
import os
from airflow.models import DagBag
import logging
import tempfile

log_filename = tempfile.mktemp(prefix="add_dag_bags_",suffix=".log")
log_file = open(log_filename, 'w+')

def log(msg):
    logging.info(msg)
    log_file.write(f"{msg}\n")
    log_file.flush()

dags_dirs = [
    '/learning_sets/dags/',
    '/learning_sets/preprocess/',
    ]

for d in dags_dirs:
    log(f"creating DagBag with path {d}")
    expanded_d = os.path.expanduser(d)
    log(f"expanded path d={expanded_d}")
    dag_bag = DagBag(
        dag_folder=expanded_d,
        include_examples=False,
        safe_mode=False
    )
    log(f"dag_bag {dag_bag}")
    log(f"dag_bag.size() = {dag_bag.size()}")
    if(dag_bag):
        for dag_id, dag in dag_bag.dags.items():
            log(f"adding dag_bag item to globals: {dag_id} = {dag}")
            globals()[dag_id] = dag

