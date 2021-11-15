#!/bin/bash

ssh root@ml.nyuki.io "killall mlflow; cd ~/learning_sets/ ; bash launch_mlflow_ui.sh; screen -ls; echo done"

ssh -L 5001:127.0.0.1:5000 root@ml.nyuki.io -N -f

