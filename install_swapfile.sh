#!/bin/bash
FILENAME=/swapfile1
apt install util-linux
fallocate -l 64G $FILENAME
chmod 600 $FILENAME
mkswap $FILENAME
swapon $FILENAME
swapon -show
free -h
cp /etc/fstab /etc/fstab.bck
grep -q "$FILENAME" /etc/fstab || printf "$FILENAME swap swap defaults 0 0\n" >> /etc/fstab