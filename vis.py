'''
Generates some diagrams for the report.

Functions:
---
plot_features: plots a feature (sos or keypoint) against the frame index
plot_keypoints: draws keypoint matches on some sample frames
'''
import pickle

from Clip import Clip
from Labels import Labels
import feature_extraction

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_feature():
    clip_name = 'departed'
    clip = Clip(clip_name)
    labels = Labels(clip_name)

    # X = feature_extraction.sum_of_squares_diff(clip)
    X = feature_extraction.keypoint_flow(clip)
    # X = feature_extraction.lucas_kanade_optical_flow(clip)
    # X = np.clip(X, 0, 100)
    y = labels.labels

    plt.plot(np.arange(len(X)), X, label='Keypoint match discrepancy')
    first_vline = True
    for i in range(len(y)):
        if y[i] == 1:
            if first_vline:
                plt.axvline(x=i, color='red', linestyle=':', label='Ground truth cut')
                first_vline = False
            else:
                plt.axvline(x=i, color='red', linestyle=':')
    plt.xlabel('Frame number')
    plt.ylabel('Keypoint match discrepancy')
    plt.title('Keypoint match discrepancy for departed.mp4')
    plt.legend()
    plt.show()


def plot_keypoints():
    name = 'departed'
    clip = Clip(name)
    path = f'features/{name}-keypoint-matches.pkl'
    pickle_f = open(path, 'rb')
    src = []
    dst = []

    while True:
        try:
            src_pts, dst_pts = pickle.load(pickle_f)
            src.append(src_pts)
            dst.append(dst_pts)
        except (EOFError, pickle.UnpicklingError):
            break

    frames = [14, 15, 16]
    for i in frames:
        frame1 = clip[i]
        frame2 = clip[i + 1]
        for p, q in zip(src[i], dst[i]):
            cv2.circle(frame1, (int(p[0]), int(p[1])), 5, (0, 255, 0))
            cv2.circle(frame2, (int(q[0]), int(q[1])), 5, (0, 255, 0))
        cv2.imshow('asdf', frame1)
        breakpoint()
        cv2.imshow('asdf', frame2)
        breakpoint()


if __name__ == '__main__':
    plot_feature()
