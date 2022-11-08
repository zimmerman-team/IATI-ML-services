#!/bin/bash


SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR
cd $LEARNING_SETS_DIR

EXPOSE_PROMETHEUS_DIR=$LEARNING_SETS_DIR/mlflow_prometheus

TS=$(date +%Y%m%d_%H%M%S)

screen -X -S mlflow quit
killall mlflow

screen -L -Logfile $LEARNING_SETS_DIR/logs/mlflow_${TS}.log -S mlflow -d -m mlflow server --expose-prometheus $EXPOSE_PROMETHEUS_DIR
