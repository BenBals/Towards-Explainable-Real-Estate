"""
This module provides normalization and scaling for data as MongoDatClasses
All operations are applied attribute-wise.
"""
from typing import List
from math import sqrt, log

from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from scipy import stats

from data.data import Immobilie


def standard_scale(data: List[Immobilie], attributes_to_normalize: List[str]) -> List[Immobilie]:
    """Scale all attributes given as input by the default scaling method"""
    if not data:
        return []
    if not attributes_to_normalize:
        return data

    pool = Pool()
    normalized_attributes = pool.map(scale_attribute, map(
        lambda attr: get_attribute_values_as_array(data, attr), attributes_to_normalize))
    for attribute, normalized_values in zip(attributes_to_normalize, normalized_attributes):
        for obj, normalized_value in zip(data, normalized_values):
            setattr(obj, attribute, normalized_value[0])

    return data


def normalize(data: List[Immobilie], attributes_to_normalize: List[str]) -> List[Immobilie]:
    """Normalize all attributes given as input by the default scaling method"""
    if not data:
        return []
    if not attributes_to_normalize:
        return data

    for attr in attributes_to_normalize:
        log_norm_normalize(data, attr)

    return data


def get_attribute_values_as_array(data: List[Immobilie], attribute: str):
    """get an array of of values of a given attribute"""
    return [[getattr(obj, attribute)] for obj in data]


def scale_attribute(attribute_values: List[float]) -> List[float]:
    """scale a given attribute on all objects
    in the input to have mean 0 and standard deviation 1"""
    scaler = StandardScaler()
    attribute_values = scaler.fit_transform(attribute_values)
    return attribute_values


def has_no_none_in_attributes(obj: Immobilie, attributes: List[str]) -> bool:
    """check if an object has no none values for a given set of attributes"""
    for attr in attributes:
        if getattr(obj, attr) is None:
            return False
    return True


def remove_none_values(data: List[Immobilie], attributes: List[str]) -> List[Immobilie]:
    """remove all data objects that have at least one none for a given list of attributes"""
    return list(filter(lambda obj: has_no_none_in_attributes(obj, attributes), data))


def log_norm_normalize(data: List[Immobilie], attribute: str) -> List[Immobilie]:
    """normalize a list of data objects in a specific attribute by taking the log"""
    for obj in data:
        if getattr(obj, attribute) is None:
            setattr(obj, attribute, 0)
        setattr(obj, attribute, log(getattr(obj, attribute)
                                    if getattr(obj, attribute) > 1 else 1))
    return data


def box_cox_normalize(data: List[Immobilie], attribute: str) -> List[Immobilie]:
    """normalize a list of data objects in a specific attribute by the box cox method"""
    attribute_values = list(
        map(lambda x: max(0.1, x[0]), get_attribute_values_as_array(data, attribute)))
    transformed_values, _ = stats.boxcox(attribute_values)

    for index, obj in enumerate(data, start=0):
        setattr(obj, attribute, transformed_values[index])
    return data


def standard_scale_list(arr: List[float]) -> List[float]:
    """scale data using standard scaling"""
    std_div, avg = std_div_and_avg(arr)
    return list(map(lambda x: standard_scale_value(std_div, avg, x), arr))


def standard_scale_value(std_div: float, avg: float, val: float):
    """calculate standard scaling value"""
    return (val - avg) / std_div


def std_div_and_avg(arr: List[float]) -> (float, float):
    """calculate standard deviation and average???"""
    sum_of_squares = 0
    sum_normal = 0
    n_var = 0

    for data_point in arr:
        n_var += 1
        sum_of_squares += data_point ** 2.0
        sum_normal += data_point

    avg = sum_normal / n_var
    var = sum_of_squares / n_var - ((sum_normal / n_var) ** 2.0)

    return sqrt(var), avg
