#!/usr/bin/env bash
set -euo pipefail

cargo run -p executables --bin dully --release -- --split-at-date 2020-01-01 --cleaning testing_only --median
cargo run -p executables --bin dully --release -- --split-at-date 2020-01-01 --cleaning testing_only --config ../config/dully/expert.dhall --median
cargo run -p executables --bin dully --release -- --split-at-date 2020-01-01 --cleaning testing_only --config ../config/dully/unweighted-l2-m20.dhall --median
cargo run -p executables --bin dully --release -- --split-at-date 2020-01-01 --cleaning testing_only --config ../config/dully/germany.dhall --median
cargo run -p executables --bin dully --release -- --split-at-date 2020-01-01 --cleaning testing_only --config ../config/dully/germany-unclean.dhall --median

cargo run -p executables --bin dully --release -- --split-at-date 2020-01-01 --cleaning testing_only -q ../queries/munich.json --median
cargo run -p executables --bin dully --release -- --split-at-date 2020-01-01 --cleaning testing_only --config ../config/dully/expert.dhall -q ../queries/munich.json --median
cargo run -p executables --bin dully --release -- --split-at-date 2020-01-01 --cleaning testing_only --config ../config/dully/unweighted-l2-m20.dhall -q ../queries/munich.json --median
cargo run -p executables --bin dully --release -- --split-at-date 2020-01-01 --cleaning testing_only --config ../config/dully/munich.dhall -q ../queries/munich.json --median
cargo run -p executables --bin dully --release -- --split-at-date 2020-01-01 --cleaning testing_only --config ../config/dully/munich-unclean.dhall -q ../queries/munich.json --median
