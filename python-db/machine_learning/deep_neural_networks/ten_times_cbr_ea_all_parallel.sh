#!/usr/bin/env bash
set -euo pipefail

sh ./ten_times_cbr_ea.sh tabnet
sh ./ten_times_cbr_ea.sh kaggle-baseline
sh ./ten_times_cbr_ea.sh kaggle-house-prices

wait
echo "Finished all configs"
