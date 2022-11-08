#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

cd $SCRIPT_DIR

screen -X -S jupyter quit
killall jupyter-notebook

jupyter nbextension enable --py widgetsnbextension

screen -L -Logfile $LEARNING_SETS_DIR/logs/jupyter_${TS}.log -S jupyter -d -m jupyter-notebook --allow-root .
