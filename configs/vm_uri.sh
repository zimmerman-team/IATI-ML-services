#!/bin/bash

python3 -c "from common import config ; print(config.vm_uri())" | tail -n1
