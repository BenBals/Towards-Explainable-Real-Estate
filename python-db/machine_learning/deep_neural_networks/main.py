#!/usr/bin/env python
"""Creates a network and evaluates it"""
import logging
import os
import sys
from datetime import datetime, date
import argparse
import tensorflow as tf

from machine_learning.deep_neural_networks.train import train_and_score, compile_model
from machine_learning.deep_neural_networks.get_dnn_data import get_dataset, get_immo_data
from machine_learning.deep_neural_networks import designer_dnns
from machine_learning.deep_neural_networks import scalers
import pickle
from deap import creator, base
from machine_learning.deep_neural_networks.network import Network
from machine_learning.helpers import find_vcs_root

def print_accuracy(accuracy):
    """prints the accuracy"""
    for attribute, value in accuracy.__dict__.items():
        logging.info(str(attribute) + ": "  + str(value))


def analyze(network, args, in_scaler=None, out_scaler=None):
    """analyses and evaluates the given network and creates plots"""
    logging.info("getting data... ")

    immo_data = get_immo_data(collection=args.collection, limit=args.limit,
                              unclean=args.unclean_japan, args=args)

    logging.info('getting dataset... ')
    learning_data = get_dataset(
        immo_data,
        train_test_split_date=args.train_test_split_date,
        sort_by_date=args.sort_by_date,
        sort_by_hash=args.sort_by_hash
    )

    logging.info('compiling...')
    model = compile_model(network, input_dim=learning_data.x_train.shape[1], args=args)

    logging.info('scaling...')
    learning_data.fit_scaler(in_scaler=in_scaler, out_scaler=out_scaler)
    learning_data.scale()

    logging.info('Input (head)')
    logging.info(learning_data.get_input_vector()[:10])

    logging.info('Output (head)')
    logging.info(learning_data.get_output_vector()[:10])

    logging.info('training...')
    accuracy, history = train_and_score(model, network, learning_data, max_epochs=args.epochs,
                                        args=args, write_to_mongo=args.write_to_mongo)

    logging.info('printing accuracy...')
    print_accuracy(accuracy)

    logging.info("# Summary\n" +
            "## Results\n" +
            "- MAE: "+ str(accuracy.mae) + '\n' +
            "- MAPE: " + str(accuracy.mape) + '\n' +
            "- MPE: " + str(accuracy.mpe) + '\n' +
            "- MSE: " + str(accuracy.mse) + '\n' +
            "- MSLE: " + str(accuracy.msle) + '\n' +
            "- Median APE: " + str(accuracy.medianape) + '\n' +
            "- PE 5, 10, 15, 20: " + str((accuracy.pe5, accuracy.pe10, accuracy.pe15, accuracy.pe20)) +
            '\n' +
            "- RMSLE: " + str(accuracy.rmsle) + '\n' +
            "- R_squared: " + str(accuracy.r_squared) + '\n' +
            "## Net Layout\n" +
            str(network) +
            "## Input\n" + str("\n- ".join(learning_data.feature_labels)) + "\n" +
            "## Output\n[Add Manually]\n" +
            "## Misc\n")


def main():
    """analyzing manual designed networks starts here"""
    parser = argparse.ArgumentParser(
        description='Train and score a DNN on real estate data from germany')
    parser.add_argument('-l', '--limit', nargs='?', default=10_000_000, type=int,
                        help='set the limit for reading data')
    parser.add_argument('-c', '--collection', nargs='?', default='cleaned_80', type=str,
                        help='set the collection to use')
    parser.add_argument('-n', '--network', nargs='?', default='default_unscaled_without_skip',
                        type=str,
                        help='set the network to use')
    parser.add_argument('-is', '--input-scaler', nargs='?', default='standardized', type=str,
                        help='set the input scaler to use')
    parser.add_argument('-os', '--output-scaler', nargs='?', default='unscaled', type=str,
                        help='set the output scaler to use')
    parser.add_argument('-sp', '--train-test-split-date', nargs='?', default=None,
                        type=date.fromisoformat,
                        help='Set the splitting date. Everything before is training-data, after is test data. Format: yyyy-mm-dd')
    parser.add_argument('-k', '--k-fold', nargs='?', const=5, default=0, type=int,
                        help='turn on k-fold cross validation')
    parser.add_argument('--predefined', help='use tabnet/...', default='')
    parser.add_argument('--epochs', help='max number of epoch to train', type=int, default=50)
    parser.add_argument('--network-from-pickle', type=str,
                        help='Load network from pickle. Set --network-from-pickle-nth-best to not use the best')
    parser.add_argument('--network-from-pickle-nth-best', type=int,
                        help='See --network-from-pickle')
    parser.add_argument('--unclean-japan', action='store_true', default=False,
                        help="Dont filter by marktwert. Don't remove none values")
    parser.add_argument('--write-to-mongo', default=None, type=str,
                        help='write predictions to column in mongo collection')
    parser.add_argument('--sort-by-date', action='store_true', default=False,
                        help='sort all immos by date first. Doing this makes the train-val-split a "into-the-future"-split')
    parser.add_argument('--sort-by-hash', type=str, default=None,
                        help='<seed>. Sort all immos by deterministic hash with seed. useful for comparable random split')
    parser.add_argument('--cbr-column', default=None, type=str,
                        help='read cbr results from column in mongo')
    parser.add_argument('--tensorflow-random-seed', type=int, default=42,
                        help='Sets the seed used for initalization of random weights')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=f'{find_vcs_root()}/logs/run_dnn_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.write_to_mongo}.log',
                        filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(args)

    tf.random.set_seed(args.tensorflow_random_seed)

    if args.network_from_pickle is None and args.network_from_pickle_nth_best is not None:
        logging.info("Must set --network-from-pickle to use --network-from-pickle-nth-best")
        sys.exit(1)

    try:
        if args.network_from_pickle:
            logging.info("Using network from pickle")
            network = network_from_file_and_index(args.network_from_pickle,
                                                  args.network_from_pickle_nth_best)
        else:
            network = getattr(designer_dnns, args.network)()
        in_scaler = getattr(scalers, args.input_scaler)()
        out_scaler = getattr(scalers, args.output_scaler)()
    except AttributeError:
        logging.info("Specified Network/scalers do not exist in designer_dnns.py/scalers.py respectively")
        sys.exit(404)

    analyze(network, args, in_scaler, out_scaler)


def network_from_file_and_index(filename, index):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", Network, fitness=creator.FitnessMin)

    with open(filename, "rb") as cp_file:
        indis = pickle.load(cp_file)

    indis.sort(key=lambda indi: indi.accuracy.mape)

    indi = indis[index or 0]

    logging.info(
        f"Running saved individual with MSE {indi.accuracy.mse}, MAPE {indi.accuracy.mape} in evo out of {len(indis)} total individuals")
    return indi


if __name__ == '__main__':
    main()
