"""class that represents a DNN, used as an Indiviuum for the Evo"""
import logging
import random
import copy
from machine_learning.deep_neural_networks.train import train_and_score, compile_model

import numpy as np

activation_functions = \
    ["relu"] * 2 + \
    ["elu"] * 2 + \
    ["sigmoid"] + \
    ["tanh"] + \
    ["selu"] + \
    ["softplus"]


class Layer:
    """Layer of a Network object"""

    # pylint: disable=too-few-public-methods

    def __init__(self):
        """Initialize the Layer by Random"""
        self.activation = random.choice(activation_functions)
        self.dropout = round(min(max(random.gauss(0.2, 0.05), 0), 0.99), 2)

    def __str__(self):
        """Prints information about the layer"""
        information = "\t\tActivation: {}\n".format(self.activation)
        information += "\t\tDropout: {}\n".format(self.dropout)
        return information

    def __eq__(self, obj):
        return self.activation == obj.activation and self.dropout == obj.dropout


class Block:
    """A block of layers"""

    def __init__(self):
        """Initialize the Layer by Random"""
        self.layers = []
        self.init_layers()
        self.n_neurons_exp = random.randint(0, 22)
        self.n_neurons = None  # legacy code
        self.has_skip = True

    def init_layers(self):
        """Add layers"""
        for _ in range(random.randint(2, 5)):
            self.layers.append(Layer())

    def get_number_of_neurons(self):
        """returns the number of neurons wich is normally sqrt(2)^n_neurons_exp but for legacy reasons sometimes
        n_neurons"""
        if self.n_neurons is not None:
            return self.n_neurons
        return int(np.round(np.power(2, self.n_neurons_exp / 2)))

    def add_layer(self):
        """Adds a new Layer as the new last one."""
        insert_id = random.randint(0, len(self.layers))
        self.layers.insert(insert_id, Layer())

    def copy_layer(self):
        """Copies a randomly choosen layer in place"""
        layer_to_copy = random.randint(0, len(self.layers) - 1)
        self.layers.insert(layer_to_copy + 1, copy.deepcopy(self.layers[layer_to_copy]))

    def remove_layer(self):
        """Pops a random layer"""
        layer_number = random.randint(0, len(self.layers) - 1)
        self.layers.pop(layer_number)

    def __str__(self):
        """Prints information about the block"""
        information = "\tNumber of Neurons: {}\n".format(self.get_number_of_neurons())
        information += "\tHas Skip Connection: {}\n".format(self.has_skip)
        for layer in self.layers:
            information += layer.__str__()
        return information

    def __eq__(self, obj):
        if len(self.layers) != len(obj.layers):
            return False

        for self_layer, obj_layer in zip(self.layers, obj.layers):
            if self_layer != obj_layer:
                return False
        return self.get_number_of_neurons() == obj.get_number_of_neurons() and self.has_skip == obj.has_skip


class SkiplessBlock(Block):
    """A Block of one layers without Skip Connection"""

    def __init__(self):
        """Inits the differences from SkiplessBlock to Block"""
        super().__init__()
        self.has_skip = False

    def init_layers(self):
        """Creates just one random layer"""
        self.layers.append(Layer())

    def add_layer(self):
        """we don't want to add a layer in skipless blocks"""

    def copy_layer(self):
        """we don't want to add a layer in skipless blocks"""

    def remove_layer(self):
        """we don't want to remove layer in skipless blocks"""


class Network(list):
    """represents the newtwork"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, allow_skip_connections=True):
        """Initialize the network"""
        super().__init__()
        self.accuracy = None
        self.history = None
        self.allow_skip_connections = allow_skip_connections
        self.blocks = []
        self.gen_number = 0
        for _ in range(random.randint(3, 8)):
            self.add_block()

        self.optimizer = 'adamax'
        self.batch_size = 256

    def add_block(self):
        """Adds a new block as the new last one."""
        insert_id = random.randint(0, len(self.blocks))
        if self.allow_skip_connections:
            new_block = Block()
        else:
            new_block = SkiplessBlock()
        self.blocks.insert(insert_id, new_block)

    def copy_block(self):
        """Copies a randomly choosen block in place"""
        block_to_copy = random.randint(0, len(self.blocks) - 1)
        self.blocks.insert(block_to_copy + 1, copy.deepcopy(self.blocks[block_to_copy]))

    def remove_block(self):
        """Pops a random block"""
        block_number = random.randint(0, len(self.blocks) - 1)
        self.blocks.pop(block_number)

    # pylint: disable=too-many-arguments
    def get_accuracy(self, learning_data, additional_callback, max_epochs,
                     write_to_mongo=None):
        """returns the network's accuracy"""
        logging.info("start evaluation")
        model = compile_model(self, input_dim=learning_data.x_train.shape[1])
        logging.info(
            f'Training set size: {len(learning_data.x_train)}, val set size: {len(learning_data.x_validation)}, test set size: {len(learning_data.x_test)}')
        accuracy, history = train_and_score(model, self, learning_data,
                                            additional_callback=additional_callback,
                                            max_epochs=max_epochs, write_to_mongo=write_to_mongo)
        self.accuracy = accuracy
        self.history = history.history
        logging.info("finished evaluation")
        return self.accuracy.mse

    def __str__(self):
        """Returns information about the network as string"""
        information = ""
        if self.accuracy is not None:
            if self.accuracy.mse is not None:
                information += "MSE: {}\n".format(self.accuracy.mse)
            if self.accuracy.mae is not None:
                information += "MAE: {}\n".format(self.accuracy.mae)
            if self.accuracy.mape is not None:
                information += "mape: {}\n".format(self.accuracy.mape)
        information += "Optimizer: {}\n".format(self.optimizer)
        information += "Batch Size: {}\n".format(self.batch_size)
        for block in self.blocks:
            information += block.__str__()

        return information

    def __eq__(self, obj):
        if len(self.blocks) != len(obj.blocks):
            return False

        for self_block, obj_block in zip(self.blocks, obj.blocks):
            if self_block != obj_block:
                return False

        return self.optimizer == obj.optimizer and \
               self.batch_size == obj.batch_size
