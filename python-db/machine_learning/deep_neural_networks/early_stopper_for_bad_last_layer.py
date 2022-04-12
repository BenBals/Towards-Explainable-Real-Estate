"""Implement callback function to stop training
when accuracy is better then the best for too long"""
import logging

import numpy as np
from tensorflow.keras.callbacks import Callback


class EarlyStopperForBadLastLayer(Callback):
    """ Implement callback function to stop training
    when accuracy is better then the best for too long"""

    # pylint: disable=too-many-instance-attributes
    # we want to use many instance attributes, since this module orientates on the tensorflow callback module

    def __init__(self, baseline=90, patience=2, monitor="val_mape"):
        super().__init__()
        self.patience = patience
        self.baseline = baseline
        self.mode = np.less
        self.monitor = monitor
        self.wait = None
        self.stopped_epoch = None
        self.epoch = None
        self.monitor_op = None

    def on_train_begin(self, logs=None):
        """initialalize some parameters when the training begins"""
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.less
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        """check if there is improvement on end of epoch"""
        if logs is None:
            logs = {}
        current = self.get_monitor_value(logs)
        if current is None:
            return

        self.wait += 1
        if self.is_acceptable(current):
            self.wait = 0

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
        self.epoch += 1

    def get_monitor_value(self, logs):
        """returns the monitored value"""
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.info(f'Early stopping conditioned on metric `{self.monitor}` which is not available. Available metrics are: {",".join(list(logs.keys()))}')

        return monitor_value

    def is_acceptable(self, monitor_value):
        """checks if current value is improvement"""
        return self.monitor_op(monitor_value, self.baseline)
