#!/bin/bash

find . | grep model.pth\$ | while read CURR; do
    du -sm $CURR
    time xz -9e $CURR
done

