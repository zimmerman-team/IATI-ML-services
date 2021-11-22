#!/bin/bash

# some tasks may fail, but that doesn't mean that the entire dataset download and creation
# process also has to fail. This script marks some failed tasks as successful so dependent
# tasks can continue being run even with slightly less data

L=$(airflow dags list-runs -d download_and_preprocess_sets -o plain |
  grep download_and_preprocess_sets |
  grep failed |
  head -n 1)
IFS=" " read DAG_ID RUN_ID STATE EXECUTION_DATE START_TIMESTAMP END_TIMESTAMP <<< $L
START_DATE=${START_TIMESTAMP/T*/}
echo DAG_ID:$DAG_ID
echo RUN_ID:$RUN_ID
echo STATE:$STATE
echo EXECUTION_DATE:$EXECUTION_DATE
echo START_TIMESTAMP:$START_TIMESTAMP
echo START_DATE:$START_DATE
echo END_TIMESTAMP:$END_TIMESTAMP

if [ -z "$END_TIMESTAMP" ]; then
    echo "this script is for runs that are not currently running. Is this currently running?"
else
    airflow tasks clear -y -f -d -s $START_DATE -e $END_TIMESTAMP download_and_preprocess_sets
    echo "done."
fi

