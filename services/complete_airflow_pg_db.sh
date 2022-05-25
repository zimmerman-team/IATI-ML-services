#!/bin/bash

# FIXME: script may buggy and work only for the first command

. ./.env
cat complete_airflow_pg_db.commands | while read ENTRYPOINT ; do
  echo "running ENTRYPOINT:$ENTRYPOINT"
  COMPOSE_PROJECT_NAME=learning_sets docker-compose run --rm --entrypoint "$ENTRYPOINT" airflow_scheduler
done
echo "all done"