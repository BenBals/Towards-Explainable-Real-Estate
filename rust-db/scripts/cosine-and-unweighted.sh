set -euo pipefail

cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 -q ../queries/munich.json --config ../config/dully/cosine-m1.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 -q ../queries/munich.json --config ../config/dully/cosine-m5.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 -q ../queries/munich.json --config ../config/dully/cosine-m10.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 -q ../queries/munich.json --config ../config/dully/cosine-m20.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 -q ../queries/munich.json --config ../config/dully/cosine-m50.dhall

cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 -q ../queries/munich.json --config ../config/dully/unweighted-l2-m1.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 -q ../queries/munich.json --config ../config/dully/unweighted-l2-m5.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 -q ../queries/munich.json --config ../config/dully/unweighted-l2-m10.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 -q ../queries/munich.json --config ../config/dully/unweighted-l2-m20.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 -q ../queries/munich.json --config ../config/dully/unweighted-l2-m50.dhall


cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 --config ../config/dully/cosine-m1.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 --config ../config/dully/cosine-m5.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 --config ../config/dully/cosine-m10.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 --config ../config/dully/cosine-m20.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 --config ../config/dully/cosine-m50.dhall

cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 --config ../config/dully/unweighted-l2-m1.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 --config ../config/dully/unweighted-l2-m5.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 --config ../config/dully/unweighted-l2-m10.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 --config ../config/dully/unweighted-l2-m20.dhall
cargo run --release --bin dully -p executables -- --split-at-date 2020-01-01 --config ../config/dully/unweighted-l2-m50.dhall
