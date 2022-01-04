#!/bin/bash

# from https://github.com/statsd/statsd

sudo apt install nodejs

git clone https://github.com/statsd/statsd

TS=$(date +%Y%m%d_%H%M%S)

bash launch_statsd.sh

