#!/bin/bash

sudo systemctl daemon-reload
sudo systemctl stop grafana-server
sudo systemctl start grafana-server
sudo systemctl status grafana-server