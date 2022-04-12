set -euo pipefail

cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m1.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m5.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m10.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m1.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m5.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m10.dhall
