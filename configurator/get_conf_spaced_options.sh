#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LEARNING_SETS_DIR="$(readlink -f $SCRIPT_DIR/..)"
cd $LEARNING_SETS_DIR

python3 -c "from common import config ; print(config.spaced_options())"
