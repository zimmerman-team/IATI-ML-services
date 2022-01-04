#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

screen -L -Logfile $LEARNING_SETS_DIR/statsd_exporter_${TS}.log -S statsd_exporter -d -m $LEARNING_SETS_DIR/monitoring/statsd_exporter/statsd_exporter
