"""holds all our scalers we might want to use for DNNs"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standardized():
    """returns a scaler for standardization"""
    return StandardScaler()


def normalized():
    """returns a scaler for normalization"""
    return MinMaxScaler((0, 1))


def unscaled():
    """returns a scaler that doesn't scale"""
    return StandardScaler(copy=True, with_mean=False, with_std=False)


def tanh_scaler():
    """returns scaler for tanh output layers"""
    return MinMaxScaler((-1, 1))
