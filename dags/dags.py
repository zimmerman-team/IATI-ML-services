# line to be included to be detected as a DAG-containing file
from airflow import DAG

# this module is loaded directly by add_dag_bags.py's DagBag
# hence needs to adjust the path to the parent folder
import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.dirname(
            os.path.abspath(__file__)
        )+"/.."
    )
)

import preprocess
from preprocess.download_sets_dag import dag as download_sets_dag_obj
import models
from models.dag import *
