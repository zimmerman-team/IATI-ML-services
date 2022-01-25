#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

cd $LEARNING_SETS_DIR

TS=$(date +%Y%m%d_%H%M%S)

screen -X -S mlflow_ui quit
killall mlflow

screen -L -Logfile $LEARNING_SETS_DIR/logs/mlflow_ui_${TS}.log -S mlflow_ui -d -m mlflow ui
