#!/bin/bash
VM_URI=$(bash config/vm_uri.sh)
#bash mongo_dump.sh
rsync -ruv --exclude .git --exclude 'mlruns*' --exclude logs --exclude corpora --exclude db_dumps .  $VM_URI:~/learning_sets/
