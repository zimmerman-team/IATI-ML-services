#!/bin/bash
#FIXME: duplicated code with launch_mlflow_ui.sh

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

cd $LEARNING_SETS_DIR

TS=$(date +%Y%m%d_%H%M%S)

screen -X -S airflow_webserver quit

sleep 1

ps aux | grep airflow | grep webserver | grep -v launch | awk '{print $2}' | xargs kill

sleep 1

screen -L -Logfile logs/airflow_webserver_${TS}.log -S airflow_webserver -d -m airflow webserver -H 127.0.0.1

echo "done."