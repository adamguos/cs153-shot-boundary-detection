import numpy as np
from sklearn.svm import SVC


class SVM(SVC):
    '''
    Support vector machine implemented by sklearn. Predictions are post-processed to apply a
    positive label to each data point preceding a positive label, due to a quirk in the Gaze Data
    dataset.
    '''

    def apply_back(self, y):
        back_idx = np.where(y == 1)[0] - 1
        non_negative = np.where(back_idx >= 0)[0]
        if len(back_idx) >= 1:
            back_idx = back_idx[non_negative[0]:]
            y[back_idx] = 1
        return y

    def predict(self, X):
        pred = super().predict(X)
        pred = self.apply_back(pred)
        return pred
