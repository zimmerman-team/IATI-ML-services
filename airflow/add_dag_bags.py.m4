""" add additional DAGs folders """
import os
from airflow.models import DagBag
dags_dirs = ['LEARNING_SETS_DIR/preprocess/']

for d in dags_dirs:
    print(f"creating DagBag with path {d}")
    dag_bag = DagBag(os.path.expanduser(d))
    print(f"dag_bag {dag_bag}")
    if(dag_bag):
        for dag_id, dag in dag_bag.dags.items():
            globals()[dag_id] = dag

