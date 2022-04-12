#!/usr/bin/env bash
set -euo pipefail

KEY=series7
RUN_NUMBER=10
CONFIGS="tabnet kaggle-baseline kaggle-house-prices"

config=$1

if [ "$config" != "tabnet" ] && [ "$config" != "kaggle-baseline" ] && [ "$config" != "kaggle-house-prices" ]; then
    echo "Invalid config $config"
    exit 1
fi

echo "Running with config $config"
for run in $(seq 1 $RUN_NUMBER); do
    python main.py -c japan_sales_reshaped \
        --predefined $config \
        -sp 2017-03-01 \
        --cbr-column cbr-ea-m-10-clean-$run \
        --write-to-mongo dnn-$KEY-$config-clean-random-val-split-cbr-ea-$run \
        --tensorflow-random-seed $run &
done
