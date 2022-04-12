from machine_learning.deep_neural_networks.get_dnn_data import deterministic_random
from bson.objectid import ObjectId


def main():
    assert deterministic_random(str(ObjectId('610ce9d3d1a638258e871b95')), 'seeed') == 227
    assert deterministic_random(str(ObjectId('abcdef1234567890abcdef12')), '42') == 81
    assert deterministic_random(str(ObjectId('103456107ab5ebc42e4ea103')), 'dnn-ea-cbr-los-randomness-is-deterministic') == 238
    assert deterministic_random(str(ObjectId('1303abec1032cdf442323e20')), 'hund sind able(n/c)kend') == 26
    assert deterministic_random(str(ObjectId('123cde212376abc125666666')), 'numberphile') == 155


if __name__ == '__main__':
    main()