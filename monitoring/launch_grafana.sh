#!/bin/bash

sudo systemctl daemon-reload
sudo systemctl start grafana-server
sudo systemctl status grafana-server