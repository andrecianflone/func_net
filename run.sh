#!/bin/bash
# Usage:
# ./run.sh case [debug]
#
# Eg:
# $ ./run.sh 1       # Run baseline
# $ ./run.sh 2 debug # Debug mode

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED} $1${CLEAR}\n";
  fi
  exit 1
}

# Debug or not
BASE='python main.py'
if [ -n "$2" ]; then
    if [[ $2 == *"debug"* ]]; then
        BASE='python -m pudb main.py'
    fi
fi

case "$1" in
        1)
            echo "Baseline"
            eval $BASE \
                --embed_dim 32 \
                --num_epochs 2
            ;;

        2)
            echo "Non-mon"
            eval $BASE \
                --nlayers 1 \
                --epoch 1000
            ;;

        *)
            usage "You need to call $0 with an int option"
            exit 1
esac

