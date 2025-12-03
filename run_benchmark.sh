#!/usr/bin/env bash

if [[ -z "$1" ]]; then
    echo "Please provide a number to indicate how many parallel processes you want to start"
fi

NUM_PROCESSES=$1
instances=$(echo {1..200})

echo $instance | xargs -n 1 -P $NUM_PROCESSES ./run_single.sh