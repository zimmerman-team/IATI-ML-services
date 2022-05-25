#!/bin/bash
VM_URI=$(bash config/vm_uri.sh)
#bash mongo_dump.sh
rsync -ruv --exclude services/airflow_scheduler/code --exclude mongo/db --exclude postgresql_for_airflow/volume/data --exclude '*.pyc' --exclude .git --exclude 'mlruns*' --exclude logs --exclude corpora --exclude db_dumps --exclude 'prometheus*.tar.gz.*' --exclude 'mlflow_prometheus' --exclude 'trained_models' --exclude 'trained_models.old' .  $VM_URI:~/learning_sets/

