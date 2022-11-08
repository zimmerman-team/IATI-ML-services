#!/bin/bash

sudo apt install golang
rm -rvf statsd_exporter
git clone https://github.com/prometheus/statsd_exporter

pushd statsd_exporter
make
go test
popd