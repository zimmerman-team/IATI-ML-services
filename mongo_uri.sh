#!/bin/bash

python3 -c "from common import config ; print(config.mongo_uri())" | tail -n1
