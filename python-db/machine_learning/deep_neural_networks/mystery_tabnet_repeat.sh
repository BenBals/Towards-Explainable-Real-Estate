#!/usr/bin/env bash
set -euo pipefail

KEY=series5
RUN_NUMBER=10
CONFIGS="tabnet"

for config in ${CONFIGS}; do
    for run in $(seq 1 $RUN_NUMBER); do
        python main.py -c japan_sales_reshaped \
            --predefined $config \
            -sp 2017-03-01 \
            --cbr-column dully-split201703-cleanv3 \
            --write-to-mongo dnn-$KEY-$config-clean-random-val-split-cbr-lbs-$run \
            --tensorflow-random-seed $run

        python main.py -c japan_sales_reshaped \
            --predefined $config \
            -sp 2017-03-01 \
            --unclean-japan \
            --cbr-column dully-split201703-uncleanv3 \
            --write-to-mongo dnn-$KEY-$config-unclean-random-val-split-cbr-lbs-$run \
            --tensorflow-random-seed $run
    done
done
