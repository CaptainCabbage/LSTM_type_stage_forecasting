from keras.models import model_from_json
import numpy as np


class BaseClassifier(object):
    def __init__(self, model_filename, weights_filename):
        """
        Construct a model from files

        :param model_filename: /path/to/the/model/file.json
        :param weights_filename: /path/to/the/weights_file.h5
        """
        self.model = self.load_model(model_filename, weights_filename)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    @staticmethod
    def load_model(model_filename, weights_filename):
        """
        Load a model from files

        :param model_filename:
        :param weights_filename:
        :return:
        """
        # load json and create model
        json_file = open(model_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weights_filename)
        return loaded_model

    def predict(self, x):
        """
        Predict the outcome of one or more new observations

        The result is an array of arrays.
        Each element comprises an array of probability of each class.

        :param x: a numpy array.
        It is either a 2D or 3D matrix. One time frame per row and one feature per column.
        The format of a 3D matrix is N x T x F, which N is the number of observations,
        T is the number of time frames, and F is number of features.
        F should be 8.
        T needs to be the same as the <longest length in training data>. If it's shorter, you can
        zero pad the sequences at the beginning or the end.

        2D is a special case of 3D, where N = 1. Again F must be 8 and T needs to match.

        :return: 2D array
        """

        # check the shape of x
        if len(x.shape) == 2:
            # convert 2D to 3D
            data = np.zeros((1, x.shape[0], x.shape[1]))
            data[0] = x
        elif len(x.shape) == 3:
            data = x
        else:
            raise ValueError('x must be a 2D or 3D numpy array')

        y_hat = self.model.predict(data)
        y_hat_converted = []
        for y in y_hat:
            y_hat_converted.append(y.tolist())
        return y_hat_converted

    def reset_state(self):
        self.model.reset_state()
