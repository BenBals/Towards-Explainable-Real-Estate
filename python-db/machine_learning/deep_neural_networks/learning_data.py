"""The class contains data prepared to learn from and the according output scaler"""
from dataclasses import dataclass
from numpy import concatenate


@dataclass
class LearningData:
    """Contains the Data prepared to learn from and the according output scaler"""

    # pylint: disable=too-many-arguments

    # pylint: disable=too-many-instance-attributes
    def __init__(self, x_train, y_train, x_test, y_test, x_validation=None, y_validation=None, *,
                 output_scaler=None,
                 input_scaler=None, feature_labels=None,
                 label_decoding_maps=None, ids_train=None, ids_test=None):
        """Initialize the data by given values"""
        self.is_scaled = False
        self.x_train = x_train
        self.x_validation = x_validation
        self.x_test = x_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        self.ids_train = ids_train  # immo ids in training vector
        self.ids_test = ids_test  # immo ids in test vector
        self.output_scaler = output_scaler
        self.input_scaler = input_scaler
        self.feature_dict = {}
        self.set_feature_labels(feature_labels)
        self.label_decoding_maps = label_decoding_maps

    def set_feature_labels(self, feature_labels):
        """sets feature labels and its inversed dict"""
        self.feature_labels = feature_labels
        self.feature_dict = {}
        if feature_labels is not None:
            for index, label in enumerate(self.feature_labels):
                self.feature_dict[label] = index

    def get_output_vector(self):
        """returns the output vector of the learning data"""
        return concatenate((self.y_train, self.y_validation, self.y_test))

    def get_input_vector(self):
        """returns the input vector of the learning data"""
        return concatenate((self.x_train, self.x_validation, self.x_test))

    def get_index(self, feature):
        """returns the index of the feature"""
        return self.get_unembedded_index(feature)

    def get_value(self, input_vector, sample_index, feature):
        """returns the value of the feature"""
        return self.get_unembedded_value(input_vector, sample_index, feature)

    def set_value(self, input_vector, sample_index, feature, value):
        """sets the value of the feature"""
        return self.set_unembedded_value(input_vector, sample_index, feature, value)

    def get_unembedded_index(self, feature):
        """returns the index within the samples for a given feature"""
        return self.feature_dict[feature]

    def get_unembedded_value(self, input_vector, sample_index, feature):
        """returns the value of a feature for one sample"""
        return input_vector[sample_index][self.get_index(feature)]

    def set_unembedded_value(self, input_vector, sample_index, feature, value):
        """sets the value of a feature for one sample"""
        input_vector[sample_index][self.get_index(feature)] = value

    def unscale(self):
        """unscale the data"""
        if self.is_scaled:
            self.x_train = self.input_scaler.inverse_transform(self.x_train)
            self.x_validation = self.input_scaler.inverse_transform(self.x_validation)
            self.x_test = self.input_scaler.inverse_transform(self.x_test)

            self.y_train = self.output_scaler.inverse_transform(self.y_train)
            self.y_validation = self.output_scaler.inverse_transform(self.y_validation)
            self.y_test = self.output_scaler.inverse_transform(self.y_test)
            self.is_scaled = False

    def scale(self):
        """scale the data"""
        if not self.is_scaled:
            self.x_train = self.input_scaler.transform(self.x_train)
            self.x_validation = self.input_scaler.transform(self.x_validation)
            self.x_test = self.input_scaler.transform(self.x_test)

            self.y_train = self.output_scaler.transform(self.y_train)
            self.y_validation = self.output_scaler.transform(self.y_validation)
            self.y_test = self.output_scaler.transform(self.y_test)
            self.is_scaled = True

    def fit_scaler(self, in_scaler=None, out_scaler=None):
        """fits the scalers"""
        if in_scaler is not None:
            self.input_scaler = in_scaler
        if out_scaler is not None:
            self.output_scaler = out_scaler
        self.input_scaler.fit(self.x_train)
        self.output_scaler.fit(self.get_output_vector())
