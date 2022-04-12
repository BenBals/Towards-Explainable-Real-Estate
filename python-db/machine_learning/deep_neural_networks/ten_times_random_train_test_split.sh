#!/usr/bin/env bash
set -euo pipefail

KEY=series3
RUN_NUMBER=10
CONFIGS="tabnet kaggle-baseline kaggle-house-prices"

for config in ${CONFIGS}; do
    for run in $(seq 1 $RUN_NUMBER); do
        python main.py -c japan_sales_reshaped \
            --predefined $config \
            --sort-by-hash $run \
            --cbr-column dully-split201703-cleanv3 \
            --write-to-mongo dnn-$KEY-$config-clean-random-split-cbr-lbs-$run \
            --tensorflow-random-seed $run

        python main.py -c japan_sales_reshaped \
            --predefined $config \
            --sort-by-hash $run \
            --unclean-japan \
            --cbr-column dully-split201703-uncleanv3 \
            --write-to-mongo dnn-$KEY-$config-unclean-random-split-cbr-lbs-$run \
            --tensorflow-random-seed $run
    done
done
