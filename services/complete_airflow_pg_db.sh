#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo SCRIPT_DIR: $SCRIPT_DIR
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
echo LEARNING_SETS_DIR: $LEARNING_SETS_DIR

D_OPTIONS="$(bash $LEARNING_SETS_DIR/configurator/get_conf_d_options.sh)"

# FIXME: script may buggy and work only for the first command

. ./.env
m4 -P $D_OPTIONS complete_airflow_pg_db.commands | while read ENTRYPOINT ; do
  echo "running ENTRYPOINT:$ENTRYPOINT"
  COMPOSE_PROJECT_NAME=learning_sets docker-compose run --rm --entrypoint "$ENTRYPOINT" airflow_scheduler
done
echo "all done"