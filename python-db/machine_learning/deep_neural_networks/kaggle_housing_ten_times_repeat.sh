#!/usr/bin/env bash
set -euo pipefail

KEY=series7
RUN_NUMBER=10
config="kaggle-house-prices"

for run in $(seq 1 $RUN_NUMBER); do
    python main.py -c japan_sales_reshaped \
        --predefined $config \
        -sp 2017-03-01 \
        --cbr-column dully-split201703-cleanv3 \
        --write-to-mongo dnn-$KEY-$config-clean-random-val-split-cbr-lbs-$run \
        --tensorflow-random-seed $run &
done

wait

for run in $(seq 1 $RUN_NUMBER); do
    python main.py -c japan_sales_reshaped \
        --predefined $config \
        -sp 2017-03-01 \
        --unclean-japan \
        --cbr-column dully-split201703-uncleanv3 \
        --write-to-mongo dnn-$KEY-$config-unclean-random-val-split-cbr-lbs-$run \
        --tensorflow-random-seed $run &
done
