#!/usr/bin/env bash
set -euo pipefail

cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m10.dhall --collection japan_sales_reshaped --unclean --query ../queries/japan-cleanv3-no-marktwert-filter.json --split-at-date 2017-03-01
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json --config ../config/dully/unweighted-l2-m20.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-default.dhall --collection japan_sales_reshaped --unclean --query ../queries/japan-cleanv3-no-marktwert-filter.json --split-at-date 2017-03-01

cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json --config ../config/dully/cosine-m1.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json --config ../config/dully/cosine-m5.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json --config ../config/dully/cosine-m10.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json --config ../config/dully/cosine-m20.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json --config ../config/dully/cosine-m50.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json --config ../config/dully/unweighted-l2-m1.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json --config ../config/dully/unweighted-l2-m5.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json --config ../config/dully/unweighted-l2-m10.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-cleanv3-no-marktwert-filter.json --config ../config/dully/unweighted-l2-m50.dhall
