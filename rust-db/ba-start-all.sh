set -e

cargo run -p executables --bin evo --release -- --config ../config/evo/ba-default.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m1.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m5.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m10.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-no-filters.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-fix-exponent.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-no-filters-fix-exponent.dhall -q ../queries/munich.json
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-default.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m1.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m5.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m10.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-no-filters.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-fix-exponent.dhall
cargo run -p executables --bin evo --release -- --config ../config/evo/ba-no-filters-fix-exponent.dhall
