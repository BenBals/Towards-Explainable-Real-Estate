#!/usr/bin/env bash
set -euo pipefail

script --flush -c "python3 main.py -c japan_sales_reshaped --japan -sp 2017-03-01 --predefined tabnet --ea-ensemble" dnn-ea-ensemble-tabnet-$(date -Ins).log
script --flush -c "python3 main.py -c japan_sales_reshaped --japan -sp 2017-03-01 --predefined kaggle-house-prices --ea-ensemble" dnn-ea-housing-ensemble-$(date -Ins).log
script --flush -c "python3 main.py -c japan_sales_reshaped --japan -sp 2017-03-01 --predefined kaggle-baseline --ea-ensemble" dnn-ea-ensemble-baseline-$(date -Ins).log
script --flush -c "python3 main.py -c japan_sales_reshaped --japan -sp 2017-03-01 --network-from-pickle ../evo_dnn/individuals-2021-08-28T21_56_05.pkl --ea-ensemble" dnn-ea-ensemble-dnn-evo-$(date -Ins).log
