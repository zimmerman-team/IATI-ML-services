#!/bin/bash

find . | grep model.pth\$ | while read CURR; do
    echo $CURR
    xz -9e $CURR
done

