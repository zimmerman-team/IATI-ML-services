#!/bin/bash

MODEL=$1
CONF=$2

if [ -z "$MODEL" ]; then
  echo "argument 1 (model) was not specified"
fi

if [ -z "$CONF" ]; then
  echo "argument 2 (conf) was not specified"
fi

screen -S $CONF -d -m python3 models/${MODEL}.py $CONF
