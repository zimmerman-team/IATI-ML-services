#!/bin/bash

# FIXME: screen quits have been added to the launch scripts
ssh root@ml.nyuki.io "screen -X -S airflow_webserver quit ; screen -X -S airflow_scheduler quit; cd ~/learning_sets/airflow/ ; sleep 1 ; bash launch_airflow.sh; screen -ls; echo done"

ssh -L 8081:127.0.0.1:8080 root@ml.nyuki.io -N -f
