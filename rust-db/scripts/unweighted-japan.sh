cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-no-duplicate-bukkens.json --config ../config/dully/cosine-m1.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-no-duplicate-bukkens.json --config ../config/dully/cosine-m5.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-no-duplicate-bukkens.json --config ../config/dully/cosine-m10.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-no-duplicate-bukkens.json --config ../config/dully/cosine-m20.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-no-duplicate-bukkens.json --config ../config/dully/cosine-m50.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-no-duplicate-bukkens.json --config ../config/dully/unweighted-l2-m1.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-no-duplicate-bukkens.json --config ../config/dully/unweighted-l2-m5.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-no-duplicate-bukkens.json --config ../config/dully/unweighted-l2-m10.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-no-duplicate-bukkens.json --config ../config/dully/unweighted-l2-m20.dhall
cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --split-at-date 2017-03-01 --cleaning none -q ../queries/japan-no-duplicate-bukkens.json --config ../config/dully/unweighted-l2-m50.dhall