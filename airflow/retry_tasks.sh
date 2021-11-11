#!/bin/bash

# some tasks may fail, but that doesn't mean that the entire dataset download and creation
# process also has to fail. This script marks some failed tasks as successful so dependent
# tasks can continue being run even with slightly less data

L=$(airflow dags list-runs -d download_and_preprocess_sets -o plain |
  grep download_and_preprocess_sets |
  head -n 1)
IFS=" " read x1 RUN_ID x2 EXECUTION_DATE START_TIMESTAMP END_TIMESTAMP <<< $L
START_DATE=${START_TIMESTAMP/T*/}
echo RUN_ID:$RUN_ID
echo EXECUTION_DATE:$EXECUTION_DATE
echo START_DATE:$START_DATE
echo END_TIMESTAMP:$END_TIMESTAMP

airflow tasks clear -y -f -d -s $START_DATE -e $END_TIMESTAMP download_and_preprocess_sets
