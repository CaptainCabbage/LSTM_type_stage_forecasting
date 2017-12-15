from ..type_classifier import TypeClassifier
import numpy as np
import unittest
import os


class TestTypeClassifier(unittest.TestCase):

    def test_predict_for_2d(self):
        model_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm.json')
        weights_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm.h5')

        classifier = TypeClassifier(model_filename, weights_filename)

        nums_timestamp = 4479
        num_features = 8
        x = np.random.rand(nums_timestamp, num_features)

        prediction = classifier.predict(x)

        num_classes = 7
        self.assertIsNotNone(prediction)
        self.assertEqual(len(prediction), 2)
        self.assertEqual(len(prediction[1][0]), num_classes)

    def test_predict_for_3d(self):
        model_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm.json')
        weights_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm.h5')

        classifier = TypeClassifier(model_filename, weights_filename)

        num_observations = 3
        numb_timestamps = 4479
        num_features = 8
        x = np.random.rand(num_observations, numb_timestamps, num_features)

        prediction = classifier.predict(x)

        num_classes = 7
        self.assertIsNotNone(prediction)
        self.assertEqual(len(prediction), 2)
        self.assertEqual(len(prediction[1]), num_observations)
        for p in prediction[1]:
            self.assertEqual(len(p), num_classes)


class TestStageClassifier(unittest.TestCase):

    def test_predict_for_2d(self):
        model_filename = os.path.join(os.path.dirname(__file__), '../models/stage_lstm.json')
        weights_filename = os.path.join(os.path.dirname(__file__), '../models/stage_lstm.h5')

        classifier = TypeClassifier(model_filename, weights_filename)

        nums_timestamp = 4479
        num_features = 8
        x = np.random.rand(nums_timestamp, num_features)

        prediction = classifier.predict(x)

        num_classes = 9
        self.assertIsNotNone(prediction)
        self.assertEqual(len(prediction), 2)
        self.assertEqual(len(prediction[1][0]), num_classes)

    def test_predict_for_3d(self):
        model_filename = os.path.join(os.path.dirname(__file__), '../models/stage_lstm.json')
        weights_filename = os.path.join(os.path.dirname(__file__), '../models/stage_lstm.h5')

        classifier = TypeClassifier(model_filename, weights_filename)

        num_observations = 3
        numb_timestamps = 4479
        num_features = 8
        x = np.random.rand(num_observations, numb_timestamps, num_features)

        prediction = classifier.predict(x)

        num_classes = 9
        self.assertIsNotNone(prediction)
        self.assertEqual(len(prediction), 2)
        self.assertEqual(len(prediction[1]), num_observations)
        for p in prediction[1]:
            self.assertEqual(len(p), num_classes)


if __name__ == '__main__':
    unittest.main()
