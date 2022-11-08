#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

TS=$(date +%Y%m%d_%H%M%S)

screen -X -S statsd quit

screen -L -Logfile $LEARNING_SETS_DIR/logs/statsd_${TS}.log -S statsd -d -m node statsd/stats.js statsdConfig.js
