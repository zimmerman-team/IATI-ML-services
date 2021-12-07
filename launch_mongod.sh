#!/bin/bash
#FIXME: duplicated code with launch_mlflow_ui.sh

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

cd $LEARNING_SETS_DIR

TS=$(date +%Y%m%d_%H%M%S)

screen -X -S mongod quit

sleep 1

screen -L -Logfile logs/mongod_${TS}.log -S mongod -d -m mongod --config /etc/mongod.conf --setParameter maxSessions=100000000

echo "done."

