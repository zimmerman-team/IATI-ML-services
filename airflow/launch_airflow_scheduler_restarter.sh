#!/bin/bash
#FIXME: duplicated code with launch_mlflow_ui.sh

# WARNING: this is a script that should counteract the unwanted Airflow's deadlock bug 
#          it's not desirable to have something that crashes constantly

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

cd $LEARNING_SETS_DIR

TS=$(date +%Y%m%d_%H%M%S)

screen -X -S airflow_scheduler_restarter quit

screen -L -Logfile $LEARNING_SETS_DIR/logs/airflow_scheduler_restarter_${TS}.log -S airflow_scheduler_restarter -d -m bash -c "while true; do airflow scheduler ; done"

