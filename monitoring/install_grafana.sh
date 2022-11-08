#!/bin/bash

sudo apt-get install -y apt-transport-https
sudo apt-get install -y software-properties-common wget
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -

echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list

sudo apt-get update
sudo apt-get -y install grafana crudini etckeeper

sudo crudini --set /etc/grafana/grafana.ini server http_port 3000

# limit service access to localhost
sudo crudini --set /etc/grafana/grafana.ini server http_addr 127.0.0.1

sudo systemctl enable grafana-server.service
sudo service grafana-server stop
sleep 1
sudo service grafana-server start
