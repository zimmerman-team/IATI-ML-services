#!/bin/bash

# exit when any command fails
#set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

# FIXME: automate conf item extraction?
AIRFLOW_PG_PASSWORD="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh airflow_pg_password)"
AIRFLOW_USER="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh airflow_user)"
AIRFLOW_PASSWORD="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh airflow_password)"
AIRFLOW_EMAIL="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh airflow_email)"
AIRFLOW_CONCURRENCY="$(bash $LEARNING_SETS_DIR/config/get_conf_item.sh airflow_concurrency)"

# exit 0 here prevents a failing killall to interrupt the script
( killall airflow ; exit 0 )

m4 -DAIRFLOW_PG_PASSWORD=$AIRFLOW_PG_PASSWORD psql_commands.m4 |
  sudo su postgres -c "/usr/bin/psql"

mkdir -pv $HOME/airflow
mkdir -pv $HOME/airflow/dags

# will macroexpand HOME with the user's home directory
m4 -DHOME=$HOME \
   -DAIRFLOW_PG_PASSWORD=$AIRFLOW_PG_PASSWORD \
   -DAIRFLOW_CONCURRENCY=$AIRFLOW_CONCURRENCY \
   airflow.cfg.m4 > $HOME/airflow/airflow.cfg

# will set the learning_sets dir (extracted from this script's path) to the module that will add learning_sets' dag
m4 -DLEARNING_SETS_DIR=$LEARNING_SETS_DIR \
   add_dag_bags.py.m4 > $HOME/airflow/dags/add_dag_bags.py

airflow db init

airflow users create -u $AIRFLOW_USER -e "$AIRFLOW_EMAIL" -r Admin -f $AIRFLOW_USER -l X -p "$AIRFLOW_PASSWORD"

airflow pools set npas_intensive 1 "creation of large npas may require a lot of resources"

pip3 install 'apache-airflow[statsd]'
