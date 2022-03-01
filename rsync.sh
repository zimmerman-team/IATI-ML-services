#!/bin/bash
VM_URI=$(bash config/vm_uri.sh)
#bash mongo_dump.sh
rsync -ruv --exclude '*.pyc' --exclude .git --exclude 'mlruns*' --exclude logs --exclude corpora --exclude db_dumps --exclude 'prometheus*.tar.gz.*' --exclude 'mlflow_prometheus' --exclude 'trained_models' --exclude 'trained_models.old' .  $VM_URI:~/learning_sets/

