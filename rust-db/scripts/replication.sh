#!/usr/bin/env bash
set -euo pipefail

# CBR+EA m<=infty clean
for run in {1..10}; do
    cargo run -p executables --bin evo --release -- --config ../config/evo/ba-default.dhall --collection japan_sales_reshaped --unclean --query ../queries/japan-no-duplicate-bukkens.json --split-at-date 2017-03-01 --write-to-mongo cbr-ea-m-infty-clean-$run
done

# CBR+EA m<=infty unclean
for run in {1..10}; do
    cargo run -p executables --bin evo --release -- --config ../config/evo/ba-default.dhall --collection japan_sales_reshaped --unclean --query ../queries/japan-cleanv3-no-marktwert-filter.json --split-at-date 2017-03-01 --write-to-mongo cbr-ea-m-infty-unclean-$run
done

# CBR+EA m<=10 clean
for run in {1..10}; do
    cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m10.dhall --collection japan_sales_reshaped --unclean --query ../queries/japan-no-duplicate-bukkens.json --split-at-date 2017-03-01 --write-to-mongo cbr-ea-m-10-clean-$run
done

# CBR+EA m<=10 unclean
for run in {1..10}; do
    cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m10.dhall --collection japan_sales_reshaped --unclean --query ../queries/japan-cleanv3-no-marktwert-filter.json --split-at-date 2017-03-01 --write-to-mongo cbr-ea-m-10-unclean-$run
done
