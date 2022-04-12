"""this module contains code related to database connection and reading data from it"""
import logging
import os
from typing import List, Dict
import pymongo

from data.data import Immobilie, attribute_name_to_database_column
from data.normalize import remove_none_values, normalize, standard_scale


def get_mongo_hostname() -> str:
    """get the mongo hostname from environment if present, otherwise returns default"""
    env_var = os.environ.get('MONGO_HOST')
    if env_var is not None and env_var != "":
        return env_var
    return "localhost"


def read(limit: int = 10000, projection=None, collection="ZIMDB_joined", query=None, args=None) -> List[
    Immobilie]:
    """returns specified number of database rows, applies projection if passed"""
    if query is None:
        query = {}
    query["glaubhaft"] = {"$ne": False}
    data = []
    with pymongo.MongoClient(host=get_mongo_hostname()) as client:
        if get_mongo_hostname() == 'localhost':
            database = client['hpidb_bp']
        else:
            database = client.get_default_database()

        logging.info(f"Connected to database {database.name}")

        coll = database[collection]

        logging.info(f'using projection: {projection}')
        for row in coll.find(query, projection).limit(limit):
            if not data:
                logging.info(f'first row: {row}')
            data.append(Immobilie(row, args.cbr_column if args is not None else None))

    return data


def load_and_normalize(limit: int = 100000, *, attributes_to_normalize=[], attributes_to_scale=[],
                       other_attributes, remove_none_for_others=True, query=None,
                       collection='ZIMDB_joined', args=None) -> List[Immobilie]:
    """Read limit entries from database and clean and normalize them"""
    if query is None:
        query = {}
    attributes = list(set(attributes_to_normalize +
                          attributes_to_scale + other_attributes))

    logging.info(f'reading from cbr-column: {args.cbr_column if args is not None else None}')
    cbr_column = [f'predictions.{args.cbr_column}'] if args is not None and args.cbr_column is not None else []
    data = read(limit, list(map(attribute_name_to_database_column, attributes)) + cbr_column, query=query,
                collection=collection, args=args)
    logging.info(f'{len(data)} database entries loaded!')

    if remove_none_for_others:
        data = remove_none_values(data, attributes + cbr_column)
    else:
        data = remove_none_values(
            data, list(set(attributes_to_normalize + attributes_to_scale)))

    # convert enums to ints
    for immobilie in data:
        for attr in immobilie.__dict__:
            attr_value = getattr(immobilie, attr)
            setattr(immobilie, attr, getattr(attr_value, 'value', attr_value))

    data = normalize(data, attributes_to_normalize)
    data = standard_scale(data, attributes_to_scale)

    logging.info(f'Normalized and remove None values\nRemaining rows: {len(data)}')

    return data
