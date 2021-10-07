#!/bin/bash

for CURR in $(ls config | sed 's/\..*//') ; do
    screen -S $CURR -d -m python3 simple_autoencoder.py $CURR
done
