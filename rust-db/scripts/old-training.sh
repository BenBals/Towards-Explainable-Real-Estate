#!/usr/bin/env bash
set -euo pipefail

cargo run --release --bin dully -p executables -- --config ../config/dully/germany.dhall --split-at-date 2020-01-01 --training-before 2019-07-01
cargo run --release --bin dully -p executables -- --config ../config/dully/germany.dhall --split-at-date 2020-01-01 --training-before 2019-01-01
cargo run --release --bin dully -p executables -- --config ../config/dully/germany.dhall --split-at-date 2020-01-01 --training-before 2018-07-01
cargo run --release --bin dully -p executables -- --config ../config/dully/germany.dhall --split-at-date 2020-01-01 --training-before 2018-01-01

cargo run --release --bin dully -p executables -- --config ../config/dully/munich.dhall --split-at-date 2020-01-01 --training-before 2019-07-01 -q ../queries/munich.json
cargo run --release --bin dully -p executables -- --config ../config/dully/munich.dhall --split-at-date 2020-01-01 --training-before 2019-01-01 -q ../queries/munich.json
cargo run --release --bin dully -p executables -- --config ../config/dully/munich.dhall --split-at-date 2020-01-01 --training-before 2018-07-01 -q ../queries/munich.json
cargo run --release --bin dully -p executables -- --config ../config/dully/munich.dhall --split-at-date 2020-01-01 --training-before 2018-01-01 -q ../queries/munich.json
