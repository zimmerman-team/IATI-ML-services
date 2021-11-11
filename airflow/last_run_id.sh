#!/bin/bash

RUN_ID=$(airflow dags list-runs --state running -d download_and_preprocess_sets -o plain |
  grep download_and_preprocess_sets |
  head -n 1 |
  awk '{print $2}')

echo $RUN_ID

