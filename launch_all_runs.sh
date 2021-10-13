#!/bin/bash

for CURR in $(ls config | sed 's/\..*//') ; do
    screen -S $CURR -d -m python3 models/item_autoencoder.py $CURR
done
