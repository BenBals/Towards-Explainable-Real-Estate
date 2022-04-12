#!/usr/bin/env bash
set -euo pipefail

#!/usr/bin/env bash
set -euo pipefail

# CBR+EA m<=infty clean
for run in {1..10}; do
    cargo run -p executables --bin evo --release -- --config ../config/evo/ba-default.dhall --collection japan_sales_reshaped --unclean --split-by-hash-key $run --query ../queries/japan-no-duplicate-bukkens.json --write-to-mongo cbr-ea-m-infty-random-train-test-split--clean-$run
done

# CBR+EA m<=infty unclean
for run in {1..10}; do
    cargo run -p executables --bin evo --release -- --config ../config/evo/ba-default.dhall --collection japan_sales_reshaped --unclean --split-by-hash-key $run --query ../queries/japan-cleanv3-no-marktwert-filter.json --write-to-mongo cbr-ea-m-infty-random-train-test-split--unclean-$run
done

# CBR+EA m<=10 clean
for run in {1..10}; do
    cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m10.dhall --collection japan_sales_reshaped --unclean --split-by-hash-key $run --query ../queries/japan-no-duplicate-bukkens.json --write-to-mongo cbr-ea-m-10-random-train-test-split--clean-$run
done

# CBR+EA m<=10 unclean
for run in {1..10}; do
    cargo run -p executables --bin evo --release -- --config ../config/evo/ba-m10.dhall --collection japan_sales_reshaped --unclean --split-by-hash-key $run --query ../queries/japan-cleanv3-no-marktwert-filter.json --write-to-mongo cbr-ea-m-10-random-train-test-split--unclean-$run
done

# CBR+LBS clean
for run in {1..10}; do
    cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --cleaning none --split-by-hash-key $run --query ../queries/japan-no-duplicate-bukkens.json --write-to-mongo cbr-lbs-random-train-test-split-clean-$run
done


# CBR+LBS unclean
for run in {1..10}; do
    cargo run -p executables --bin dully --release -- --collection japan_sales_reshaped --cleaning none --split-by-hash-key $run --query ../queries/japan-cleanv3-no-marktwert-filter.json --write-to-mongo cbr-lbs-random-train-test-split-unclean-$run
done
