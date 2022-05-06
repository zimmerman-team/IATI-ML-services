# line to be included to be detected as a DAG-containing file

# this module is loaded directly by add_dag_bags.py's DagBag
# hence needs to adjust the path to the parent folder
import sys
import os
import logging
import tempfile
sys.path.append(
    os.path.abspath(
        os.path.dirname(
            os.path.abspath(__file__)
        )+"/.."
    )
)


log_filename = tempfile.mktemp(prefix="dags_py_",suffix=".log")
log_file = open(log_filename, 'w+')

def log(msg):
    logging.info(msg)
    log_file.write(f"{msg}\n")
    log_file.flush()

from preprocess.download_sets_dag import dag as download_sets_dag_obj
log(f"download_sets_dag_obj {download_sets_dag_obj}")
