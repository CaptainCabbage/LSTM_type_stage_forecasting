from base_classifier import BaseClassifier


class TypeClassifier(BaseClassifier):
    def __init__(self, model_filename, weights_filename):
        super(TypeClassifier, self).__init__(model_filename, weights_filename)
        self.classes = ['success', 'noscrew', 'no_hole_found', 'crossthread',
                        'stripped_no_engage', 'stripped', 'partial']

    def predict(self, x):
        """
        Predict the outcome of one or more new observations

        The result is a tuple of two arrays.
        The 1st array comprises class names, while the 2nd is an array of array.
        Each element comprises an array of probability of each class.

        :param x: a numpy array.
        It is either a 2D or 3D matrix. One time frame per row and one feature per column.
        The format of a 3D matrix is N x T x F, which N is the number of observations,
        T is the number of time frames, and F is number of features.
        F should be 8.
        T needs to be the same as the <longest length in training data>. If it's shorter, you can
        zero pad the sequences at the beginning or the end.

        2D is a special case of 3D, where N = 1. Again F must be 8 and T needs to match.

        :return: a tuple of two arrays
        """
        y_hat_converted = super(TypeClassifier, self).predict(x)
        return self.classes, y_hat_converted

    def reset_state(self):
        self.model.reset_state()
