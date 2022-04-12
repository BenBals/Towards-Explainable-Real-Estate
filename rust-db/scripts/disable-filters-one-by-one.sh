#!/usr/bin/env bash
set -euo pipefail

cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable-regiotyp.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_ausstattungsnote.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_baujahr.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_grundstuecksgroesse.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_objektunterart.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_wertermittlungsstichtag.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_wohnflaeche.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_zustand.dhall -q ../queries/munich.json

cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable-regiotyp.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_ausstattungsnote.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_baujahr.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_grundstuecksgroesse.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_objektunterart.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_wertermittlungsstichtag.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_wohnflaeche.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/disable-filters/disable_zustand.dhall
