#!/bin/bash

# some tasks may fail, but that doesn't mean that the entire dataset download and creation
# process also has to fail. This script marks some failed tasks as successful so dependent
# tasks can continue being run even with slightly less data

RUN_ID=$(airflow dags list-runs --state running -d download_and_preprocess_sets -o plain |
  grep download_and_preprocess_sets |
  head -n 1 |
  awk '{print $2}')

echo RUN_ID:$RUN_ID