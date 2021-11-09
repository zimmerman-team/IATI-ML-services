#!/bin/bash

rsync --exclude '*.pth' --exclude '*.png' -ruv root@ml.nyuki.io:~/learning_sets/mlruns/ ./mlruns/
