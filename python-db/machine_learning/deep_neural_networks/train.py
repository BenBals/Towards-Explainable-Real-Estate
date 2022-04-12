"""deep learning to predict the marktwert value (not normalized) via deep neural network"""
import logging
from time import time
from copy import deepcopy
from math import log
import os
import pandas as pd

import numpy as np
import pymongo
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, \
    mean_squared_log_error
from tensorflow import pad, keras
from tensorflow.keras.layers import Add, Input, Dense, Concatenate, Embedding, Reshape, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tabnet import TabNetRegressor
from tqdm import tqdm

from data.database import get_mongo_hostname
from machine_learning.gpu import setup_gpu
from machine_learning.helpers import create_dirs_if_not_existing
from machine_learning.deep_neural_networks.dnn_results import DnnResults
from machine_learning.deep_neural_networks.get_dnn_data import scale_input
from machine_learning.deep_neural_networks.early_stopper_for_bad_last_layer import \
    EarlyStopperForBadLastLayer

setup_gpu()


def percentage_error_below(treshhold, y_true, y_pred):
    """calculates how many of the guesses have a abslute percentage error better then the treshhold"""
    percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
    return sum(percentage_error < treshhold for percentage_error in percentage_errors) / len(y_true)


def median_absolute_percentage_error(y_true, y_pred):
    """calculates median absolute percentage error"""
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_percentage_error(y_true, y_pred):
    """returns mean absolute percentage error between y_true and y_pred.
    sklearn isn't able to do this in the version we use, so we had to implemented on our own."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_percentage_error(y_true, y_pred):
    """returns mean absolute percentage error between y_true and y_pred.
    sklearn isn't able to do this in the version we use, so we had to implemented on our own."""
    return np.mean(((y_true - y_pred) / y_true)) * 100


def compile_model(network, *, input_dim=None, embedding_input_layers=None,
                  embedding_output_layers=None, args=None):
    """Compiles a functional keras model using the specifications from a network and returns it.
    Optionally takes the input from a model with embedded categorical attributes"""

    predefined = args.predefined if args is not None else ''

    if predefined.lower() == 'tabnet':
        logging.info('using tabnet')
        return tabnet(input_dim)
    if predefined.lower() == 'kaggle-house-prices':
        logging.info('using kaggle house prices')
        return kaggle_house_price(input_dim)
    if predefined.lower() == 'kaggle-baseline':
        logging.info('using kaggle baseline')
        return kaggle_baseline(input_dim)
    if predefined:
        raise ValueError(f'unknown predefined NN: {predefined}')

    logging.info('using old compile_model method')

    list_of_layers = []
    if embedding_input_layers is not None:
        list_of_layers.append(embedding_input_layers)
        list_of_layers.append(Concatenate()(embedding_output_layers))
    else:
        list_of_layers.append(Input(shape=input_dim))

    max_weight_norm = MaxNorm(3)
    for block in network.blocks[0:-1]:
        current_layer_number = len(list_of_layers) - 1
        for layer in block.layers:
            list_of_layers.append(Dense(
                block.get_number_of_neurons(),
                layer.activation,
                kernel_initializer="uniform",
                kernel_constraint=max_weight_norm,
                name=f"{len(list_of_layers)}Activation-{layer.activation}")(list_of_layers[-1]))
            list_of_layers.append(
                Dropout(layer.dropout, name=f"{len(list_of_layers)}dropout-{layer.dropout}")(
                    list_of_layers[-1]))
        if block.has_skip:
            layer1 = list_of_layers[-1]
            layer2 = list_of_layers[current_layer_number]
            if layer1.shape[1] < layer2.shape[1]:
                layer1 = pad(layer1, paddings=[[0, 0], [0, layer2.shape[1] - layer1.shape[1]]],
                             constant_values=0)
            else:
                layer2 = pad(layer2, paddings=[[0, 0], [0, layer1.shape[1] - layer2.shape[1]]],
                             constant_values=0)
            list_of_layers.append(Add()([layer1, layer2]))

    for layer in network.blocks[-1].layers[0:-1]:
        list_of_layers.append(Dense(
            network.blocks[-1].get_number_of_neurons(),
            layer.activation,
            kernel_initializer="uniform",
            kernel_constraint=max_weight_norm,
            name=f"{len(list_of_layers)}Activation-{layer.activation}")(list_of_layers[-1]))
        list_of_layers.append(
            Dropout(layer.dropout, name=f"{len(list_of_layers)}dropout-{layer.dropout}")(
                list_of_layers[-1]))

    list_of_layers.append(
        Dense(1, activation=network.blocks[-1].layers[-1].activation)(list_of_layers[-1]))

    model = Model(inputs=list_of_layers[0], outputs=list_of_layers[-1])

    model.compile(loss='mse', optimizer=network.optimizer,
                  metrics=['mse', 'mae', 'mape'])

    return model


def tabnet(input_dim: int):
    """returns a compiled tabnet model"""
    model = TabNetRegressor(feature_columns=None, num_features=input_dim, num_regressors=1,
                            feature_dim=128)

    model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae', 'mape'])

    return model


def kaggle_house_price(input_dim: int):
    """returns a compiled kaggle model for house prices"""
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mse', optimizer=keras.optimizers.Adadelta(),
                  metrics=['mse', 'mae', 'mape'])

    return model


def kaggle_baseline(input_dim: int):
    """returns a compiled kaggle model which is used as a baseline for real estate"""
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss=root_mean_squared_error,
                  optimizer=adam, metrics=['mse', 'mae', 'mape'])
    return model


def kaggle_callback():
    """returns a callback function which sets the learning rate accordingly for `kaggle_baseline`"""

    def step_decay(epoch):
        initial_lrate = 0.01
        drop = 0.8
        epochs_drop = float(20)
        lrate = initial_lrate * \
                np.power(drop, np.floor((1 + epoch) / epochs_drop))
        return lrate

    return LearningRateScheduler(step_decay)


def train_and_score_k_fold(network, learning_data, input_scaler, num_folds=5):
    """does train_and_score on a k-fold"""
    k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    accuracies = []
    histories = []
    learning_datas = []
    for train_index, test_index in k_fold.split(learning_data.get_input_vector(),
                                                learning_data.get_output_vector()):
        fold_learning_data = deepcopy(learning_data)
        fold_learning_data.x_train, fold_learning_data.x_test = learning_data.get_input_vector()[
                                                                    train_index], \
                                                                learning_data.get_input_vector()[
                                                                    test_index]
        fold_learning_data.y_train, fold_learning_data.y_test = learning_data.get_output_vector()[
                                                                    train_index], \
                                                                learning_data.get_output_vector()[
                                                                    test_index]
        fold_learning_data.x_train, fold_learning_data.x_val, fold_learning_data.y_train, fold_learning_data.y_val = train_test_split(
            fold_learning_data.x_train, fold_learning_data.y_train, test_size=0.2, random_state=42)
        scale_input(input_scaler, fold_learning_data)

        model_copy = compile_model(network, input_dim=fold_learning_data.x_train.shape[1])
        accuracy, history = train_and_score(model_copy, network, fold_learning_data)
        accuracies.append(accuracy)
        histories.append(history)
        learning_datas.append(fold_learning_data)

    results = DnnResults(
        np.mean([accuracy.mae for accuracy in accuracies]),
        np.mean([accuracy.mape for accuracy in accuracies]),
        np.mean([accuracy.medianape for accuracy in accuracies]),
        np.mean([accuracy.mpe for accuracy in accuracies]),
        np.mean([accuracy.mse for accuracy in accuracies]),
        np.mean([accuracy.msle for accuracy in accuracies]),
        np.mean([accuracy.pe5 for accuracy in accuracies]),
        np.mean([accuracy.pe10 for accuracy in accuracies]),
        np.mean([accuracy.pe15 for accuracy in accuracies]),
        np.mean([accuracy.pe20 for accuracy in accuracies]),
        np.mean([accuracy.r_squared for accuracy in accuracies]))

    return results, histories, learning_datas


# pylint: disable = too-many-arguments
def train_model(model, network, learning_data, additional_callback=None, max_epochs=50, args=None):
    """trains the model"""

    """
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    checkpoint_filepath = './checkpoints/cp' + str(time()) + '.txt'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    """

    early_stopper = EarlyStopping(patience=20, min_delta=0)
    # Some Networks have a last layer which doesn't fit the scaling. As a result, they have a really high mape. We want
    # to stop early, if such a network is trained.

    early_stopper_for_bad_last_layer = EarlyStopperForBadLastLayer(patience=2, baseline=90,
                                                                   monitor="val_mape")
    callbacks = [early_stopper,
                 # model_checkpoint_callback,
                 early_stopper_for_bad_last_layer]

    if additional_callback is not None:
        callbacks.append(additional_callback)

    if args is not None:
        if args.predefined.lower() == 'kaggle-baseline':
            callbacks.append(kaggle_callback())
        if args.predefined.lower() in ['tabnet', 'kaggle-baseline', 'kaggle-house-prices']:
            callbacks.remove(early_stopper_for_bad_last_layer)

    history = model.fit(learning_data.x_train, learning_data.y_train,
                        batch_size=network.batch_size,
                        epochs=max_epochs,  # using early stopping, so no real limit
                        verbose=True,
                        validation_data=(
                            learning_data.x_validation, learning_data.y_validation),
                        callbacks=callbacks)

    # model.load_weights(checkpoint_filepath)

    return history


def prefecture_error(learning_data, predictions):
    """converts vectors to pandas dataframe grouped by plz or kreis"""

    fehler = {}
    error_sum = 0

    for sample_index in range(len(predictions)):
        prefecture = learning_data.z_test[sample_index][0]

        error = abs(learning_data.y_test[sample_index][0] - predictions[sample_index][0]) / \
                learning_data.y_test[sample_index][0]
        error_sum = error + error_sum

        if prefecture in fehler:
            e = fehler[prefecture]
            fehler[prefecture] = (e[0] + 1, e[1] + error)
        else:
            fehler[prefecture] = (1, error)

    logging.info(f'Prefecture MAPE: {error_sum / len(predictions)}')

    ort_df = pd.DataFrame.from_dict(fehler, orient="index", columns=["count", "summed_error"])
    ort_df['Average Error'] = ort_df['summed_error'] / ort_df['count']

    create_dirs_if_not_existing("export")
    export_file = open(f"export/prefecture_error.txt", "w", encoding="utf-8")
    export_file.write(ort_df.sort_values(by=['Average Error'], ascending=False).to_string())
    export_file.close()

    return ort_df


def write_predictions_to_mongo(column_name, collection, predictions, prediction_ids):
    def update_operation(prediction, id):
        return pymongo.UpdateOne(
            {"_id": id},
            {"$set": {f'predictions.{column_name}': prediction}})

    with pymongo.MongoClient(host=get_mongo_hostname()) as client:
        database = client.get_default_database()
        coll = database[collection]
        pbar = tqdm(total=len(predictions))

        logging.info(f'Writing {len(predictions)} predictions to collection {collection}...')
        buf = []

        BATCH_SIZE = 10000
        for pred, id in zip(predictions, prediction_ids):
            buf.append(update_operation(pred, id))

            if len(buf) >= BATCH_SIZE:
                coll.bulk_write(buf)
                buf = []
                pbar.update(BATCH_SIZE)

        coll.bulk_write(buf)


def evaluate(model, network, learning_data, *, write_to_mongo=None):
    """evaluates the model"""
    # pylint: disable = too-many-locals

    logging.info(learning_data.input_scaler)
    logging.info(learning_data.output_scaler)

    logging.info('Training history...')
    logging.info(model.history.history)

    predictions = model.predict(
        learning_data.x_test, batch_size=network.batch_size)
    logging.info(f'avg pred: {np.mean(predictions)}')
    logging.info(f'avg real: {np.mean(learning_data.y_test)}')

    y_test = learning_data.output_scaler.inverse_transform(
        learning_data.y_test)

    logging.info(f'y_test[1] = {y_test[1]}')
    logging.info(f'pred[1] = {predictions[1]}')
    predictions = learning_data.output_scaler.inverse_transform(predictions)
    logging.info(f'pred[1] = {predictions[1]}')
    np.clip(predictions, 0, None, out=predictions)
    logging.info(f'pred[1] = {predictions[1]}')

    logging.info(f'avg pred: {np.mean(predictions)}')
    logging.info(f'avg real: {np.mean(y_test)}')

    if write_to_mongo is not None:
        write_predictions_to_mongo(write_to_mongo[1], write_to_mongo[0], [float(p[0]) for p in predictions],
                                   learning_data.ids_test)

    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    mpe = mean_percentage_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    medianape = median_absolute_percentage_error(y_test, predictions)
    msle = mean_squared_log_error(y_test, predictions)
    pe5 = percentage_error_below(5, y_test, predictions)
    pe10 = percentage_error_below(10, y_test, predictions)
    pe15 = percentage_error_below(15, y_test, predictions)
    pe20 = percentage_error_below(20, y_test, predictions)

    log_predictions = np.array(list(map(lambda x: log(x) if x > 0 else 1, predictions)))
    log_y_test = np.array(list(map(lambda x: log(x), y_test)))
    log_mape = mean_absolute_percentage_error(log_y_test, log_predictions)

    prefecture_error(learning_data, predictions)

    return DnnResults(mae, mape, medianape, mpe, mse, msle, pe5, pe10, pe15, pe20, r_squared,
                      log_mape)


# pylint: disable = too-many-arguments
def train_and_score(model, network, learning_data, *, additional_callback=None, max_epochs=50,
                    args=None, write_to_mongo=None):
    """Train the model, return test loss."""
    history = train_model(model, network, learning_data, additional_callback=additional_callback,
                          max_epochs=max_epochs, args=args)
    return evaluate(model, network, learning_data,
                    write_to_mongo=(
                        args.collection,
                        write_to_mongo) if write_to_mongo is not None else None), history
