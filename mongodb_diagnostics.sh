#!/bin/bash
echo "db.serverStatus().connections
db.serverStatus().logicalSessionRecordCache" \
| mongosh --quiet

