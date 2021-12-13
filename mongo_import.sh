#!/bin/bash

MONGO_URI=$(bash mongo_uri.sh)
for COLLECTION in npas_tsets fs.chunks fs.files ; do
  FILENAME="db_dumps/${COLLECTION}.json"
  echo "Importing collection $COLLECTION from $FILENAME"
  mongoimport -d learning_sets \
    -c $COLLECTION \
    $MONGO_URI \
    $FILENAME
done
