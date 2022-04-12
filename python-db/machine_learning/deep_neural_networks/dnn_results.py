"""This class contains information about the accuracy of a DNN"""
from dataclasses import dataclass
from math import sqrt


@dataclass
class DnnResults:
    """Contains information about the accuracy of a DNN"""

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(self, mae, mape, medianape, mpe, mse, msle, pe5, pe10, pe15, pe20, r_squared,
                 log_mape=None):
        """Initialize by given values"""
        self.mse = mse
        self.mae = mae
        self.mape = mape
        self.mpe = mpe
        self.r_squared = r_squared
        self.medianape = medianape
        self.msle = msle
        self.pe5 = pe5
        self.pe10 = pe10
        self.pe15 = pe15
        self.pe20 = pe20
        self.rmsle = sqrt(msle)
        self.log_mape = log_mape
