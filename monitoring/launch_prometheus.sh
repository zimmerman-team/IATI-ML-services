#!/bin/bash

sudo systemctl daemon-reload
sudo systemctl stop prometheus
sudo systemctl start prometheus

sudo systemctl status prometheus
