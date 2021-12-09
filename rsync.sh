#!/bin/bash
VM_URI=$(bash config/vm_uri.sh)
#bash mongo_dump.sh
rsync -ruv --exclude 'mlruns*' --exclude logs --exclude corpora .  $VM_URI:~/learning_sets/
