#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

TS=$(date +%Y%m%d_%H%M%S)

screen -X -S statsd_exporter quit
sleep 1
killall statsd_exporter

cd ${LEARNING_SETS_DIR}/monitoring
screen -L -Logfile ${LEARNING_SETS_DIR}/logs/statsd_exporter_${TS}.log -S statsd_exporter -d -m ${LEARNING_SETS_DIR}/monitoring/statsd_exporter/statsd_exporter --statsd.mapping-config=${LEARNING_SETS_DIR}/monitoring/statsd_exporter_mapping.conf --web.listen-address="127.0.0.1:9102" --statsd.listen-udp="127.0.0.1:9125" --statsd.listen-tcp="127.0.0.1:9125"
