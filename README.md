# Towards Explainable Real Estate Valuation via Evolutionary Algorithms - Code

See https://arxiv.org/abs/2110.05116 for the paper this code belongs to.

We know this is scientific that the is, at best, doubiously documented. We are happy to help, contact us at [ben.bals@student.hpi.de](mailto:ben.bals@student.hpi.de).

## What is this?
This repo contains scientific code to evaluate the algorithms presentend in the paper experimentally.

## The two parts
The paper compares our CBR+EA approach to more traditional DNNs. The folder `rust-db` contains code to execute the CBR+EA algorithm and `python-db`contains code to execute the comparision DNNs.

## The Database
The algorithms work with training and testing data which is loaded from a Mongo DB. You have to setup a  MongoDB which contains your dataset in a collection in the following schema. Then place the connection string into the environment variable `MONGO_HOST` in the following format

```bash
export MONGO_HOST="export MONGO_HOST="mongodb://[user]:[encoded pw]@[hostname]:[port]/[database_name]"
```

If your  dataset  has a different schema, you mainly need to adjust `python-db/data/data.py` and `rust-db/common/src/immo.rs` to match it. In Rust, the compiler will guide you to adjust all other relevant places as well.


### Schema
(Note that we load more data, in both code bases but only use those listed here for prediction.)

| column name                              | interpretation                 |
|------------------------------------------|--------------------------------|
| `marktwert`                              | asking price                   |
| `kurzgutachten.objektangabenBaujahr`     | year of construction           |
| `kurzgutachten.objektangabenWohnflaeche` | living area                    |
| `grundstuecksgroesseInQuadratmetern`     | lot area                       |
| `average_loca`                           | average local price            |
| `balcony_area`                           | balcony area                   |
| `wertermittlungsstichtag`                | offer date                     |
| `walk_distance1`                         | distance to public transport   |
| `convenience_distance`                   | distance to convenicence store |
| `school_ele_distance`                    | distance to elementary school  |
| `school_jun_distance`                    | distance to junior high school |
| `distance_parking`                       | distance to parking            |
| `house_kaisuu`                           | floor                          |
| `plane_location`                         | location (coordinate tuple)    |
| `objektunterart`                         | object type                    |
| `land_toshi`                             | urbanity score                 |

## Running the CBR+EA

### Installation
You need a recent rust compiler with `cargo`. We recommend installation by [rustup](https://rustup.rs/). You additonally need openssl, pkgconfig and dhall.

### Running
Note, place commandline arguments passed to the algorithm *after* the double dash `--`. Arguments before the double dash are passed to the Rust compiler.

From within the `rust-db` folder, run
```bash
$ cargo run -p executables --bin evo --release -- --help

USAGE:
    evo [FLAGS] [OPTIONS]

FLAGS:
        --evaluate-all-generations-on-test-set    Evaluate the best individual of each generation the test set
        --expert
    -h, --help                                    Prints help information
        --local-search
        --median
            Use weighted median instead of weighted average prediction. Default: false

        --unclean                                 Don't perform data cleaning
    -V, --version                                 Prints version information

OPTIONS:
    -c, --collection <collection>                   [default: cleaned_80]
        --config <config-path>
    -l <limit>
    -o, --output <output-path>
    -q, --query <query-path>
    -s, --seed <seed-path>
        --split-at-date <split-at-date>
            Split train/validation data at a given date. Enter in YYYY-MM-DD format.

        --split-by-hash-key <split-by-hash-key>
            Split into training and test set deterministically based on hashing an object with following key

    -w, --weight-output <weight-output-path>
        --write-to-mongo <write-to-mongo>          Write predictions to mongodb. Argument: key to use in the database.
```

## Running the comparison DNNs
### Installation
Install dependencies via [pip](https://pypi.org/project/pip/):
``` sh
pip install -r python-db/requirements.txt
```

### Run the DNNs

From within `python-db/machine_learning/deep_neural_networks`, run
``` sh
$ python main.py --help

usage: main.py [-h] [-l [LIMIT]] [-c [COLLECTION]] [-n [NETWORK]] [-is [INPUT_SCALER]] [-os [OUTPUT_SCALER]]
               [-sp [TRAIN_TEST_SPLIT_DATE]] [-k [K_FOLD]] [--predefined PREDEFINED] [--epochs EPOCHS]
               [--network-from-pickle NETWORK_FROM_PICKLE]
               [--network-from-pickle-nth-best NETWORK_FROM_PICKLE_NTH_BEST] [--unclean-japan]
               [--write-to-mongo WRITE_TO_MONGO] [--sort-by-date] [--sort-by-hash SORT_BY_HASH]
               [--cbr-column CBR_COLUMN] [--tensorflow-random-seed TENSORFLOW_RANDOM_SEED]

Train and score a DNN on real estate data from germany

optional arguments:
  -h, --help            show this help message and exit
  -l [LIMIT], --limit [LIMIT]
                        set the limit for reading data
  -c [COLLECTION], --collection [COLLECTION]
                        set the collection to use
  -n [NETWORK], --network [NETWORK]
                        set the network to use
  -is [INPUT_SCALER], --input-scaler [INPUT_SCALER]
                        set the input scaler to use
  -os [OUTPUT_SCALER], --output-scaler [OUTPUT_SCALER]
                        set the output scaler to use
  -sp [TRAIN_TEST_SPLIT_DATE], --train-test-split-date [TRAIN_TEST_SPLIT_DATE]
                        Set the splitting date. Everything before is training-data, after is test data. Format:
                        yyyy-mm-dd
  -k [K_FOLD], --k-fold [K_FOLD]
                        turn on k-fold cross validation
  --predefined PREDEFINED
                        use tabnet/...
  --epochs EPOCHS       max number of epoch to train
  --network-from-pickle NETWORK_FROM_PICKLE
                        Load network from pickle. Set --network-from-pickle-nth-best to not use the best
  --network-from-pickle-nth-best NETWORK_FROM_PICKLE_NTH_BEST
                        See --network-from-pickle
  --unclean-japan       Dont filter by marktwert. Don't remove none values
  --write-to-mongo WRITE_TO_MONGO
                        write predictions to column in mongo collection
  --sort-by-date        sort all immos by date first. Doing this makes the train-val-split a "into-the-
                        future"-split
  --sort-by-hash SORT_BY_HASH
                        <seed>. Sort all immos by deterministic hash with seed. useful for comparable random
                        split
  --cbr-column CBR_COLUMN
                        read cbr results from column in mongo
  --tensorflow-random-seed TENSORFLOW_RANDOM_SEED
                        Sets the seed used for initalization of random weights
```

## Notes
The file `mongo/japan_recode_sales.json` contains code to transform a collection as read from the [data provided by the Japanese National Institute of Informatics](https://www.nii.ac.jp/dsc/idr/en/lifull/) into a collection format our code can work with.
