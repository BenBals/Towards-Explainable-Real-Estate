"""Contains functions which return a Network.
   If a Network is updated, please add a new another one with an increased id and return the new one in the function
   without the id"""
from machine_learning.deep_neural_networks.network import Network, Block, SkiplessBlock, Layer


def japan_evo_220():
    network = Network()
    network.optimizer = 'adamax'
    network.batch_size = 256

    network.blocks = [None] * 8
    network.blocks[0] = SkiplessBlock()
    network.blocks[1] = SkiplessBlock()
    network.blocks[2] = SkiplessBlock()
    network.blocks[3] = SkiplessBlock()
    network.blocks[4] = SkiplessBlock()
    network.blocks[5] = SkiplessBlock()
    network.blocks[6] = SkiplessBlock()
    network.blocks[7] = SkiplessBlock()
    network.blocks[0].n_neurons = 2048
    network.blocks[1].n_neurons = 1448
    network.blocks[2].n_neurons = 256
    network.blocks[3].n_neurons = 64
    network.blocks[4].n_neurons = 181
    network.blocks[5].n_neurons = 181
    network.blocks[6].n_neurons = 362
    network.blocks[7].n_neurons = 1
    network.blocks[0].layers[0].activation = 'tanh'
    network.blocks[1].layers[0].activation = 'elu'
    network.blocks[2].layers[0].activation = 'relu'
    network.blocks[3].layers[0].activation = 'sigmoid'
    network.blocks[4].layers[0].activation = 'relu'
    network.blocks[5].layers[0].activation = 'relu'
    network.blocks[6].layers[0].activation = 'selu'
    network.blocks[7].layers[0].activation = 'elu'
    network.blocks[0].layers[0].dropout = 0.2
    network.blocks[1].layers[0].dropout = 0.16
    network.blocks[2].layers[0].dropout = 0.08
    network.blocks[3].layers[0].dropout = 0.14
    network.blocks[4].layers[0].dropout = 0.23
    network.blocks[5].layers[0].dropout = 0.24
    network.blocks[6].layers[0].dropout = 0.22
    network.blocks[7].layers[0].dropout = 0.21

    return network


def default_unscaled_with_skip():
    """Input: Standardized, Output, Unscaled"""
    # pylint: disable=too-many-statements
    network = Network()
    network.optimizer = 'adamax'
    network.batch_size = 256

    network.blocks = [None] * 7

    network.blocks[0] = Block()
    network.blocks[0].has_skip = True
    network.blocks[0].n_neurons = 573
    network.blocks[0].layers = [None] * 6

    network.blocks[0].layers[0] = Layer()
    network.blocks[0].layers[0].activation = 'elu'
    network.blocks[0].layers[0].dropout = 0.18
    network.blocks[0].layers[1] = Layer()
    network.blocks[0].layers[1].activation = 'tanh'
    network.blocks[0].layers[1].dropout = 0.17
    network.blocks[0].layers[2] = Layer()
    network.blocks[0].layers[2].activation = 'relu'
    network.blocks[0].layers[2].dropout = 0.23
    network.blocks[0].layers[3] = Layer()
    network.blocks[0].layers[3].activation = 'relu'
    network.blocks[0].layers[3].dropout = 0.24
    network.blocks[0].layers[4] = Layer()
    network.blocks[0].layers[4].activation = 'relu'
    network.blocks[0].layers[4].dropout = 0.22
    network.blocks[0].layers[5] = Layer()
    network.blocks[0].layers[5].activation = 'relu'
    network.blocks[0].layers[5].dropout = 0.3

    network.blocks[1] = Block()
    network.blocks[1].has_skip = True
    network.blocks[1].n_neurons = 841
    network.blocks[1].layers = [None] * 4

    network.blocks[1].layers[0] = Layer()
    network.blocks[1].layers[0].activation = 'elu'
    network.blocks[1].layers[0].dropout = 0.27
    network.blocks[1].layers[1] = Layer()
    network.blocks[1].layers[1].activation = 'relu'
    network.blocks[1].layers[1].dropout = 0.23
    network.blocks[1].layers[2] = Layer()
    network.blocks[1].layers[2].activation = 'relu'
    network.blocks[1].layers[2].dropout = 0.22
    network.blocks[1].layers[3] = Layer()
    network.blocks[1].layers[3].activation = 'elu'
    network.blocks[1].layers[3].dropout = 0.23

    network.blocks[2] = Block()
    network.blocks[2].has_skip = True
    network.blocks[2].n_neurons = 493
    network.blocks[2].layers = [None] * 8

    network.blocks[2].layers[0] = Layer()
    network.blocks[2].layers[0].activation = 'elu'
    network.blocks[2].layers[0].dropout = 0.27
    network.blocks[2].layers[1] = Layer()
    network.blocks[2].layers[1].activation = 'elu'
    network.blocks[2].layers[1].dropout = 0.21
    network.blocks[2].layers[2] = Layer()
    network.blocks[2].layers[2].activation = 'softplus'
    network.blocks[2].layers[2].dropout = 0.19
    network.blocks[2].layers[3] = Layer()
    network.blocks[2].layers[3].activation = 'selu'
    network.blocks[2].layers[3].dropout = 0.22
    network.blocks[2].layers[4] = Layer()
    network.blocks[2].layers[4].activation = 'tanh'
    network.blocks[2].layers[4].dropout = 0.14
    network.blocks[2].layers[5] = Layer()
    network.blocks[2].layers[5].activation = 'relu'
    network.blocks[2].layers[5].dropout = 0.25
    network.blocks[2].layers[6] = Layer()
    network.blocks[2].layers[6].activation = 'relu'
    network.blocks[2].layers[6].dropout = 0.13
    network.blocks[2].layers[7] = Layer()
    network.blocks[2].layers[7].activation = 'relu'
    network.blocks[2].layers[7].dropout = 0.20

    network.blocks[3] = Block()
    network.blocks[3].has_skip = True
    network.blocks[3].n_neurons = 493
    network.blocks[3].layers = [None] * 3

    network.blocks[3].layers[0] = Layer()
    network.blocks[3].layers[0].activation = 'softplus'
    network.blocks[3].layers[0].dropout = 0.15
    network.blocks[3].layers[1] = Layer()
    network.blocks[3].layers[1].activation = 'elu'
    network.blocks[3].layers[1].dropout = 0.12
    network.blocks[3].layers[2] = Layer()
    network.blocks[3].layers[2].activation = 'elu'
    network.blocks[3].layers[2].dropout = 0.23

    network.blocks[4] = Block()
    network.blocks[4].has_skip = True
    network.blocks[4].n_neurons = 89
    network.blocks[4].layers = [None] * 5

    network.blocks[4].layers[0] = Layer()
    network.blocks[4].layers[0].activation = 'elu'
    network.blocks[4].layers[0].dropout = 0.25
    network.blocks[4].layers[1] = Layer()
    network.blocks[4].layers[1].activation = 'softplus'
    network.blocks[4].layers[1].dropout = 0.23
    network.blocks[4].layers[2] = Layer()
    network.blocks[4].layers[2].activation = 'sigmoid'
    network.blocks[4].layers[2].dropout = 0.25
    network.blocks[4].layers[3] = Layer()
    network.blocks[4].layers[3].activation = 'sigmoid'
    network.blocks[4].layers[3].dropout = 0.17
    network.blocks[4].layers[4] = Layer()
    network.blocks[4].layers[4].activation = 'relu'
    network.blocks[4].layers[4].dropout = 0.21

    network.blocks[5] = Block()
    network.blocks[5].has_skip = True
    network.blocks[5].n_neurons = 1153
    network.blocks[5].layers = [None] * 4

    network.blocks[5].layers[0] = Layer()
    network.blocks[5].layers[0].activation = 'elu'
    network.blocks[5].layers[0].dropout = 0.14
    network.blocks[5].layers[1] = Layer()
    network.blocks[5].layers[1].activation = 'relu'
    network.blocks[5].layers[1].dropout = 0.20
    network.blocks[5].layers[2] = Layer()
    network.blocks[5].layers[2].activation = 'elu'
    network.blocks[5].layers[2].dropout = 0.21
    network.blocks[5].layers[3] = Layer()
    network.blocks[5].layers[3].activation = 'selu'
    network.blocks[5].layers[3].dropout = 0.33

    network.blocks[6] = Block()
    network.blocks[6].has_skip = False
    network.blocks[6].n_neurons = 395
    network.blocks[6].layers = [None] * 6

    network.blocks[6].layers[0] = Layer()
    network.blocks[6].layers[0].activation = 'relu'
    network.blocks[6].layers[0].dropout = 0.20
    network.blocks[6].layers[1] = Layer()
    network.blocks[6].layers[1].activation = 'relu'
    network.blocks[6].layers[1].dropout = 0.17
    network.blocks[6].layers[2] = Layer()
    network.blocks[6].layers[2].activation = 'relu'
    network.blocks[6].layers[2].dropout = 0.17
    network.blocks[6].layers[3] = Layer()
    network.blocks[6].layers[3].activation = 'softplus'
    network.blocks[6].layers[3].dropout = 0.21
    network.blocks[6].layers[4] = Layer()
    network.blocks[6].layers[4].activation = 'relu'
    network.blocks[6].layers[4].dropout = 0.23
    network.blocks[6].layers[5] = Layer()
    network.blocks[6].layers[5].activation = 'selu'
    network.blocks[6].layers[5].dropout = 0.18

    return network


def default_unscaled_without_skip():
    """returns a standard dnn for unscaled output predictions"""
    network = Network()
    network.optimizer = 'adamax'
    network.batch_size = 256

    network.blocks = [None] * 5
    network.blocks[0] = SkiplessBlock()
    network.blocks[1] = SkiplessBlock()
    network.blocks[2] = SkiplessBlock()
    network.blocks[3] = SkiplessBlock()
    network.blocks[4] = SkiplessBlock()
    network.blocks[0].n_neurons = 594
    network.blocks[1].n_neurons = 1169
    network.blocks[2].n_neurons = 1274
    network.blocks[3].n_neurons = 114
    network.blocks[4].n_neurons = 1
    network.blocks[0].layers[0].activation = 'tanh'
    network.blocks[1].layers[0].activation = 'sigmoid'
    network.blocks[2].layers[0].activation = 'relu'
    network.blocks[3].layers[0].activation = 'elu'
    network.blocks[4].layers[0].activation = 'relu'
    network.blocks[0].layers[0].dropout = 0.25
    network.blocks[1].layers[0].dropout = 0.18
    network.blocks[2].layers[0].dropout = 0.14
    network.blocks[3].layers[0].dropout = 0.21
    network.blocks[4].layers[0].dropout = 0.0

    return network


def default_scaled():
    """returns a default dnn for standardized/normalized output predictions"""
    network = Network()
    network.optimizer = 'nadam'
    network.batch_size = 155

    network.blocks = [None] * 3
    network.blocks[0] = SkiplessBlock()
    network.blocks[1] = SkiplessBlock()
    network.blocks[2] = SkiplessBlock()
    network.blocks[0].n_neurons = 566
    network.blocks[1].n_neurons = 926
    network.blocks[2].n_neurons = 1
    network.blocks[0].layers[0].activation = 'relu'
    network.blocks[1].layers[0].activation = 'relu'
    network.blocks[2].layers[0].activation = 'tanh'
    network.blocks[0].layers[0].dropout = 0.1669213469520731
    network.blocks[1].layers[0].dropout = 0.25711308087339435
    network.blocks[2].layers[0].dropout = 0

    return network


def random_net():
    """returns a random net"""
    return Network()
