"""Prepare data to be used in dnn"""
import logging
import math
from datetime import date

import numpy as np

from sklearn.model_selection import train_test_split

from machine_learning.deep_neural_networks.learning_data import LearningData
from data.normalize import remove_none_values
from data.database import load_and_normalize
import hashlib


def deterministic_random(id, seed):
    def sha(string):
        return int(hashlib.sha256(string.encode()).hexdigest()[:2], 16)
    return sha(id) ^ sha(seed)


def get_days_from_2015(new_date):
    """calculates the number of days since 1st january of 2015"""
    try:
        return (new_date - date(2015, 1, 1)).days
    except:
        return (new_date.date() - date(2015, 1, 1)).days



def train_test_split_by_percentage(input_data, output_data, z_data, test_percentage, *,
                                   immo_ids=None):
    """split data by percentage"""
    output_ids = True
    if immo_ids is None:
        output_ids = False
        immo_ids = [None for _ in input_data]

    split_point = len(input_data) - math.floor(len(input_data) * test_percentage)
    x_train, x_test = input_data[:split_point], input_data[split_point:]
    y_train, y_test = output_data[:split_point], output_data[split_point:]
    z_train, z_test = z_data[:split_point], z_data[split_point:]
    ids_train, ids_test = immo_ids[:split_point], immo_ids[split_point:]

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if output_ids:
        return x_train, x_test, y_train, y_test, z_train, z_test, ids_train, ids_test
    return x_train, x_test, y_train, y_test, z_train, z_test


def train_test_split_by_date(input_data, output_data, z_data, split_date, feature_labels, *,
                             immo_ids=None):
    """splits data into training and testing data by date"""
    output_ids = True
    if immo_ids is None:
        output_ids = False
        immo_ids = [None for _ in input_data]

    x_train = []
    y_train = []
    z_train = []
    ids_train = []
    x_test = []
    y_test = []
    z_test = []
    ids_test = []
    for i, label in enumerate(feature_labels):
        if label == "anzahl_tage_1.1.2015_bis_bewertung":
            days_from_2015_index = i
    treshhold = get_days_from_2015(split_date)
    for input_sample, output_sample, z_sample, id in zip(input_data, output_data, z_data, immo_ids):
        if input_sample[days_from_2015_index] < treshhold:
            x_train.append(input_sample)
            y_train.append(output_sample)
            z_train.append(z_sample)
            ids_train.append(id)
        else:
            x_test.append(input_sample)
            y_test.append(output_sample)
            z_test.append(z_sample)
            ids_test.append(id)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if output_ids:
        return x_train, x_test, y_train, y_test, z_train, z_test, ids_train, ids_test
    return x_train, x_test, y_train, y_test, z_train, z_test


def get_attribute_values(data, attribute):
    """returns deduplicated list of the values of an attribute"""
    value_set = set()
    for immobilie in data:
        dot_index_in_string = attribute.find('.')
        if dot_index_in_string != -1:
            value_set.add(getattr(getattr(immobilie, attribute[:dot_index_in_string]),
                                  attribute[dot_index_in_string + 1:]))
            continue
        value_set.add(getattr(immobilie, attribute))

    return list(value_set)


def label_encode(data, attribute):
    """returns ditionary containing a label encoding for the given attribute"""
    unique_values = sorted(get_attribute_values(data, attribute))
    encoding_dict = {}
    for index, value in enumerate(unique_values):
        encoding_dict[value] = index

    return encoding_dict


def get_feature_labels():
    """returns the labels of the input vector"""
    feature_labels = [
        "baujahr",
        "grundstuecksgroesse",
        "lat",
        "lng",
        "wohnflaeche",
        "anzahl_tage_1.1.2015_bis_bewertung",
        "wertermittlungsstichtag.year",
        "wertermittlungsstichtag.month",
        "wertermittlungsstichtag.day",
        "balcony_area",
        "walk_distance1",
        "land_toshi",
        "house_kaisuu",
        "school_el_distance",
        "school_jun_distance",
    ]

    feature_labels += ([
        "dully_japan",
    ])

    return feature_labels


def get_immo_data(collection, *, limit=10000, unclean=False, args=None):
    """gets the data representing real estate"""
    query = {
        "plane_location.0": {"$gt": -1140843.77, "$lt": 1414596.17},
        "plane_location.1": {"$gt": -401614.57, "$lt": 2618746.47},
        "duplicate": {"$ne": True},
        "objektunterart": {"$ne": "invalid"},
        "kurzgutachten.objektangabenBaujahr": {"$gt": 1500, "$lt": 2025},
        "kurzgutachten.objektangabenWohnflaeche": {"$gt": 20, "$lt": 2000},
        "last_entry_with_bukken_id": True,
    }

    if not unclean:
        query["marktwert"] = {"$lte": 300000000}

    logging.info(f"Getting data with query {query}")

    data = load_and_normalize(limit, attributes_to_normalize=[], attributes_to_scale=[],
                              other_attributes=[
                                  'ea_japan',
                                  'dully_japan',
                                  'dully_japan_unclean',
                                  'anzahl_carport',
                                  'anzahl_garagen',
                                  'anzahl_stellplaetze_aussen',
                                  'anzahl_stellplaetze_innen',
                                  'anzahl_wohneinheiten',
                                  'ausgebauter_spitzboden',
                                  'ausstattung',
                                  'ausstattungGuessed',
                                  'baujahr',
                                  'centrality',
                                  'dully_discrete_split',
                                  'grundstuecksgroesse',
                                  'kaufpreis',
                                  'kreis',
                                  'lagequalitaet',
                                  'lat',
                                  'lng',
                                  'plane_x',
                                  'plane_y',
                                  'location',
                                  'marktwert',
                                  'objektunterart',
                                  'plz',
                                  'pois',
                                  'prefecture',
                                  'regiotyp',
                                  'restnutzungsdauer',
                                  'scores_all',
                                  'verwertbarkeit',
                                  'verwertbarkeitGuessed',
                                  'wertermittlungsstichtag',
                                  'wohnflaeche',
                                  'zustand',
                                  "balcony_area",
                                  "walk_distance1",
                                  "land_toshi",
                                  "house_kaisuu",
                                  "school_el_distance",
                                  "school_jun_distance",
                              ],
                              remove_none_for_others=False,
                              query=query,
                              collection=collection,
                              args=args)
    logging.info(f'Loaded: {len(data)}')

    attributes_to_remove_none_values = [
        'dully_japan',
        'baujahr',
        'marktwert',
        'plane_x',
        'plane_y',
        'quadratmeterpreis',
        'wertermittlungsstichtag',
        'wohnflaeche',
        "balcony_area",
        "walk_distance1",
        "land_toshi",
        "house_kaisuu",
        "school_el_distance",
        "school_jun_distance",
    ]

    if args is not None and args.cbr_column is not None:
        attributes_to_remove_none_values.append("cbr_prediction")

    logging.info(f'Example Immo: {data[0]}')
    if not unclean:
        data = remove_none_values(data, attributes_to_remove_none_values)

    logging.info(f'Still: {len(data)}')

    if not data:
        logging.error('No data loaded. Exciting...')
        exit(1)

    return data


def scale_input(input_scaler, learning_data):
    """scales the input data"""
    input_scaler.fit(learning_data.x_train)
    learning_data.x_train = input_scaler.transform(learning_data.x_train)
    learning_data.x_validation = input_scaler.transform(learning_data.x_validation)
    learning_data.x_test = input_scaler.transform(learning_data.x_test)
    learning_data.input_scaler = input_scaler


def scale_output(output_scaler, learning_data):
    """scales the output data"""
    output_scaler.fit(learning_data.get_output_vector())
    learning_data.y_train = output_scaler.transform(learning_data.y_train)
    learning_data.y_validation = output_scaler.transform(learning_data.y_validation)
    learning_data.y_test = output_scaler.transform(learning_data.y_test)
    learning_data.output_scaler = output_scaler


def split_vectors_to_learning_data(in_vector, out_vector, train_test_split_date, feature_labels):
    """splits input and output vector and writes it into a learning_data object"""
    if train_test_split_date is None:
        x_train, x_test, y_train, y_test = train_test_split(in_vector, out_vector, test_size=0.2,
                                                            random_state=42)
    else:
        x_train, x_test, y_train, y_test = train_test_split_by_date(in_vector, out_vector,
                                                                    train_test_split_date,
                                                                    feature_labels)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                      random_state=42)
    return LearningData(x_train, y_train, x_test, y_test, x_val, y_val)


def immo_to_vector(entry):
    vector = [
        entry.baujahr,
        entry.grundstuecksgroesse or 0.0,
        entry.plane_x,
        entry.plane_y,
        entry.wohnflaeche,
        (entry.wertermittlungsstichtag - date(2015, 1, 1)).days,
        entry.wertermittlungsstichtag.year,
        entry.wertermittlungsstichtag.month,
        entry.wertermittlungsstichtag.day,
        entry.balcony_area,
        entry.walk_distance1,
        entry.land_toshi,
        entry.house_kaisuu,
        entry.school_el_distance,
        entry.school_jun_distance,
    ]

    vector += ([
        (entry.cbr_prediction or 0.0) / entry.wohnflaeche
    ])

    assert (all(map(lambda x: x is not None, vector)))
    return vector


def get_dataset(immo_data, *, train_test_split_date=None, sort_by_date=False, sort_by_hash=None):
    """returns the dataset as an learningData object containing numpy arrays"""

    if sort_by_date:
        immo_data.sort(key=lambda immo: immo.wertermittlungsstichtag)

    if sort_by_hash is not None:
        immo_data.sort(key=lambda immo: (deterministic_random(str(immo.id), sort_by_hash), str(immo.id)))

    # keep order synced with feature_labels
    in_vector = np.array([immo_to_vector(entry) for entry in immo_data])

    in_vector = in_vector.astype(np.float32)
    feature_labels = get_feature_labels()

    out_vector = np.array([[entry.quadratmeterpreis] for entry in immo_data])
    z_vector = np.array([[entry.prefecture] for entry in immo_data])
    assert (all(map(lambda x: x[0] is not None, out_vector)))

    print(f'Example in/out: {in_vector[0]} \n {out_vector[0]}')

    ids_train, ids_test = None, None
    if train_test_split_date is None:
        x_train, x_test, y_train, y_test, z_train, z_test, ids_train, ids_test = train_test_split_by_percentage(in_vector, out_vector,
                                                                             z_vector,
                                                                             test_percentage=0.2,
                                                                             immo_ids=[immo.id for immo in immo_data])
    else:
        x_train, x_test, y_train, y_test, z_train, z_test, ids_train, ids_test = train_test_split_by_date(
            in_vector, out_vector, z_vector, train_test_split_date, feature_labels,
            immo_ids=[immo.id for immo in immo_data])
    x_train, x_val, y_train, y_val, z_train, z_val = train_test_split_by_percentage(x_train,
                                                                                    y_train,
                                                                                    z_train,
                                                                                    test_percentage=0.2)

    learning_data = LearningData(x_train, y_train, x_test, y_test, x_val, y_val,
                                 feature_labels=feature_labels, ids_train=ids_train,
                                 ids_test=ids_test)
    learning_data.z_train = z_train
    learning_data.z_val = z_val
    learning_data.z_test = z_test
    print(len(learning_data.x_train), len(learning_data.x_test), len(learning_data.x_validation))

    return learning_data
