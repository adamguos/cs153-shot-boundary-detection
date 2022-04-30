'''
Runs tests for the report.

Functions:
---
run_all_tests: runs the tests for each feature and classifier combination and saves F1-scores to
scores.csv. Takes an optional parameter that determines whether to run the TransNetV2 tests (which
are time-consuming).
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Clip import Clip
import feature_extraction
from Labels import Labels
from SVM import SVM
from ThresholdClassifier import ThresholdClassifier

from TransNetV2.inference.transnetv2 import TransNetV2

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


def run_all_tests(transnet=True):
    feature_methods = [
        feature_extraction.sum_of_squares_diff, feature_extraction.keypoint_flow,
        feature_extraction.all_features
    ]
    feature_names = [m.__name__ for m in feature_methods]

    if not transnet:
        df = pd.read_csv('scores.csv', index_col=0)
        trans_score = df.loc['TransNetV2']
    df = pd.DataFrame(columns=feature_names)
    clip_names = sorted([os.path.splitext(f)[0] for f in os.listdir('clips')])
    clips = [Clip(c) for c in clip_names]
    labels = [Labels(c, binary=True) for c in clip_names]
    ss = StandardScaler()

    def test_classifier(clf):
        f1_scores = []
        for feature_method in feature_methods:
            confusion = np.zeros((2, 2), dtype=int)
            for clip, label in zip(clips, labels):
                X = feature_method(clip)
                X = ss.fit_transform(X)
                y = label.labels
                if len(np.unique(y)) != 2:
                    continue
                try:
                    pred = cross_val_predict(clf, X, y)
                except ValueError as e:
                    break
                confusion += metrics.confusion_matrix(y, pred)
            print(f'{type(clf).__name__}, {feature_method.__name__}')
            print(confusion)
            print()
            if confusion.any():
                tn, fp, fn, tp = confusion.ravel()
                f1 = tp / (tp + 0.5 * (fp + fn))
            else:
                f1 = 0
            f1_scores.append(f1)
        return f1_scores

    classifiers = [ThresholdClassifier(), SVM()]
    for classifier in classifiers:
        df.loc[type(classifier).__name__] = test_classifier(classifier)

    if transnet:
        transnet = TransNetV2()
        confusion = np.zeros((2, 2), dtype=int)
        for clip, label in zip(clips, labels):
            y = label.labels
            if len(np.unique(y)) != 2:
                continue
            clip_name = clip.clip_name
            path = f'clips/{clip_name}.mp4'
            video_frames, single_frame_predictions, all_frame_predictions = transnet.predict_video(
                path)
            scenes = transnet.predictions_to_scenes(single_frame_predictions)
            pred = np.zeros(len(clip), dtype=int)
            pred[scenes[:, 1][:-1]] = 1
            pred[scenes[:, 0][1:]] = 1
            confusion += metrics.confusion_matrix(y, pred)
        print(type(transnet).__name__)
        print(confusion)
        print()
        tn, fp, fn, tp = confusion.ravel()
        f1 = tp / (tp + 0.5 * (fp + fn))
        df.loc[type(transnet).__name__] = [f1] * len(feature_methods)
    else:
        df.loc['TransNetV2'] = trans_score

    print(df)
    df.to_csv('scores.csv')


if __name__ == '__main__':
    run_all_tests()
