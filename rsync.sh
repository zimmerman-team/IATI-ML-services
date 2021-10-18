#!/bin/bash
VM_URI=$(bash vm_uri.sh)
bash mongo_dump.sh
rsync -ruv --exclude mlruns .  $VM_URI:~/learning_sets/
