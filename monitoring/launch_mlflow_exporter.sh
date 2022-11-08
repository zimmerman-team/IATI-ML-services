#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

TS=$(date +%Y%m%d_%H%M%S)

screen -X -S mlflow_exporter quit
sleep 1

cd ${LEARNING_SETS_DIR}

screen -L -Logfile ${LEARNING_SETS_DIR}/logs/mlflow_exporter_${TS}.log -S mlflow_exporter -d -m python3 ${LEARNING_SETS_DIR}/monitoring/mlflow_exporter.py
