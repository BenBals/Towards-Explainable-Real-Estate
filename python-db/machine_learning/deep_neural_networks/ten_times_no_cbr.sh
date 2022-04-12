#!/usr/bin/env bash
set -euo pipefail

KEY=series5
RUN_NUMBER=10
CONFIGS="tabnet kaggle-baseline kaggle-house-prices"

for config in ${CONFIGS}; do
    for run in $(seq 1 $RUN_NUMBER); do
        python main.py -c japan_sales_reshaped \
            --predefined $config \
            -sp 2017-03-01 \
            --write-to-mongo dnn-$KEY-$config-clean-random-val-split-no-cbr-$run \
            --tensorflow-random-seed $run
    done
done


for config in ${CONFIGS}; do
    for run in $(seq 1 $RUN_NUMBER); do
        python main.py -c japan_sales_reshaped \
            --predefined $config \
            -sp 2017-03-01 \
            --unclean-japan \
            --write-to-mongo dnn-$KEY-$config-unclean-random-val-split-no-cbr-$run \
            --tensorflow-random-seed $run
    done
done
