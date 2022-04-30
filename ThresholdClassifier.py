import numpy as np
from scipy.optimize import fmin
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    '''
    Simple threshold-based classifier conforming to scikit-learn API. Tries 100 possible thresholds
    on the training data and picks the one with the best F1-score. Predictions are post-processed to
    apply a positive label to each data point preceding a positive label, due to a quirk in the Gaze
    Data dataset.

    Attributes:
    ---
    ratio: float
        an optional pre-determined ratio between the minimum and maximum of the given data at which
        to set the threshold
    '''

    def __init__(self, ratio=None):
        self.ratio = ratio

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = sorted(unique_labels(y))

        if len(self.classes_) != 2:
            raise ValueError('ThresholdClassifier only supports binary classification')
        if X.shape[1] > 1:
            raise ValueError(
                'ThresholdClassifier only supports data with a single feature (shape = (n, 1))')

        self.X_ = X
        self.y_ = y

        if self.ratio:
            self.threshold_ = X.min() + (X.max() - X.min()) * self.ratio
        else:
            possible_thresholds = X.min() + (X.max() - X.min()) * np.linspace(0, 1, 101)

            def score(threshold):
                mask = (X < threshold)[:, 0]
                pred = np.zeros(len(X), dtype=y.dtype)
                pred[mask] = 0
                pred[np.logical_not(mask)] = 1
                return f1_score(y, pred)

            scores = np.array([score(t) for t in possible_thresholds])
            self.threshold_ = possible_thresholds[scores.argmax()]

        return self

    def apply_back(self, y):
        back_idx = np.where(y == 1)[0] - 1
        non_negative = np.where(back_idx >= 0)[0]
        if len(back_idx) >= 1:
            back_idx = back_idx[non_negative[0]:]
            y[back_idx] = 1
        return y

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        mask = (X < self.threshold_)[:, 0]
        pred = np.zeros(len(X), dtype=self.y_.dtype)
        pred[mask] = 0
        pred[np.logical_not(mask)] = 1

        pred = self.apply_back(pred)

        return pred
