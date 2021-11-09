#!/bin/bash

rsync --exclude '*.pth' --exclude '*.png' -ruv root@ml.nyuki.io:~/learning_sets/mlruns/ ./mlruns/

find ./ -iname '*.yaml' -type f -exec sed -i 's/\/root\//\/home\/frablum\/PycharmProjects\/learning_sets\//g' {} \;
