#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <last_argument>"
    exit 1
fi

ARG=$1

sbatch -p swl1 seq_mnist_hydra.sh "$ARG"
sbatch -p stat seq_mnist_hydra.sh "$ARG"
sbatch -p owners --array=1-4 seq_mnist_hydra.sh "$ARG"