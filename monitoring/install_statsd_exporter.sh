#!/bin/bash

sudo apt install golang

git clone https://github.com/prometheus/statsd_exporter

pushd statsd_exporter
make
go test
popd