#!/bin/bash

# we use an alternative local ip address to represent the remote machine
TUNNEL_IP=127.0.2.1

# also we add an alternative hostname instead of "localhost" and we call this "tunnel"
TUNNEL_HOSTNAME=tunnel

# remote ip address where the services are actually listening
REMOTE_HOST=127.0.3.1

# lists of ports on the remote machine that are going to be mapped locally
PORTS="8080 9090 9125 9102 3000 5000 5500 8888 18080 3100"

VM_URI=$(bash config/vm_uri.sh)

sudo cp /etc/hosts /etc/hosts.bck

# adding the tunnel hostname and ip address to /etc/hosts only if not already there
sudo grep -q "$TUNNEL_HOSTNAME" /etc/hosts || printf "$TUNNEL_IP $TUNNEL_HOSTNAME\n" | sudo tee -a /etc/hosts

for CURR in $PORTS ; do
  echo "current port: $CURR . Can be accessed via ${TUNNEL_HOSTNAME}:${CURR}"
  ssh -L ${TUNNEL_HOSTNAME}:${CURR}:${REMOTE_HOST}:${CURR} $VM_URI -N -f
done
