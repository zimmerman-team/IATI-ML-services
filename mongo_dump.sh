#!/bin/bash
MONGO_URI=$(bash mongo_uri.sh)
for COLLECTION in npas_tsets fs.chunks fs.files ; do
  FILENAME="db_dumps/${COLLECTION}.json"
  echo "Exporting collection $COLLECTION to $FILENAME"
  mongoexport --collection $COLLECTION --out $FILENAME $MONGO_URI
done
