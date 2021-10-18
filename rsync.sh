#!/bin/bash

MONGO_URI=$(bash mongo_uri.sh)
VM_URI=$(bash vm_uri.sh)
mongoexport --collection npas_tsets --out db_dumps/exported_npas_tsets.json $MONGO_URI
rsync -ruv --exclude mlruns .  $VM_URI:~/learning_sets/
