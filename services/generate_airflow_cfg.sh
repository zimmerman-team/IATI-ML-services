#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/../..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

D_OPTIONS="$(bash $LEARNING_SETS_DIR/config/get_conf_d_options.sh)"

# will macroexpand HOME with the user's home directory
m4 -P $D_OPTIONS $SCRIPT_DIR/airflow.cfg.m4 > $SCRIPT_DIR/airflow.cfg
