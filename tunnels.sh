#!/bin/bash

VM_URI=$(bash config/vm_uri.sh)

ssh -L 8081:127.0.0.1:8080 $VM_URI -N -f
ssh -L 9091:127.0.0.1:9090 $VM_URI -N -f
ssh -L 19125:127.0.0.1:9125 $VM_URI -N -f
ssh -L 19102:127.0.0.1:9102 $VM_URI -N -f
ssh -L 3001:127.0.0.1:3000 $VM_URI -N -f
ssh -L 18888:127.0.0.1:8888 $VM_URI -N -f
