#!/usr/bin/env bash
set -euo pipefail

cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/enable-regiotyp.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/enable_grundstuecksgroesse.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/enable_wohnflaeche.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/enable_zustand.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/enable_all.dhall -q ../queries/munich.json

cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/enable-regiotyp.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/enable_grundstuecksgroesse.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/enable_wohnflaeche.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/enable_zustand.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/enable_all.dhall
