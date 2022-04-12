#!/usr/bin/env bash
set -euo pipefail

for run in {1..10}; do
    cargo run -p executables --bin evo --release -- --config ../config/evo/ba-default.dhall -q ../queries/munich.json
done

for run in {1..10}; do
    cargo run -p executables --bin evo --release -- --config ../config/evo/ba-default.dhall
done
