#!/bin/bash

# some tasks may fail, but that doesn't mean that the entire dataset download and creation
# process also has to fail. This script marks some failed tasks as successful so dependent
# tasks can continue being run even with slightly less data

RUN_ID=$(bash last_run_id.sh)
echo RUN_ID:$RUN_ID