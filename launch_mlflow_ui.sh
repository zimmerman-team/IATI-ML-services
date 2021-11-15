#!/bin/bash

TS=$(date +%Y%m%d_%H%M%S)

screen -L -Logfile logs/mlflow_ui_${TS}.log -S mlflow_ui -d -m mlflow ui
