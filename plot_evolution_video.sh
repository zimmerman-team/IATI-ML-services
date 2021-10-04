#!/bin/bash

PREFIX=$1
if test -z "$PREFIX"; then
    echo "first argument has to be a prefix"
    exit
fi

ls $PREFIX* |
    while read BEFORE; do
        AFTER=$(echo $BEFORE | sed "s/^\([^0-9]*\)\([0-9]*\)\(.*\)$/\1\2.png/g"); 
        mv $BEFORE $AFTER; 
    done

ffmpeg \
    -r:v 10 \
    -i "$PREFIX_%04d.png" \
    -codec:v libx264 \
    -preset veryslow \
    -pix_fmt yuv420p \
    -crf 28 \
    -an "$PREFIX.mp4"
