#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
mkdir -pv $HOME/airflow
mkdir -pv $HOME/airflow/dags

# will macroexpand HOME with the user's home directory
m4 -DHOME=$HOME airflow.cfg.m4 > $HOME/airflow/airflow.cfg

# will set the learning_sets dir (extracted from this script's path) to the module that will add learning_sets' dag
m4 -DLEARNING_SETS_DIR=$LEARNING_SETS_DIR add_dag_bags.py.m4 > $HOME/airflow/dags/add_dag_bags.py
