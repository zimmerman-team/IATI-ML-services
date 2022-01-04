#!/bin/bash

ssh -L 8081:127.0.0.1:8080 root@ml.nyuki.io -N -f
ssh -L 9091:127.0.0.1:9090 root@ml.nyuki.io -N -f
ssh -L 19125:127.0.0.1:9125 root@ml.nyuki.io -N -f
ssh -L 19102:127.0.0.1:9102 root@ml.nyuki.io -N -f
ssh -L 3001:127.0.0.1:3000 root@ml.nyuki.io -N -f
