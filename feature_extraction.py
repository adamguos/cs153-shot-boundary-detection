'''
Contains functions for computing, saving, and loading features. In general, every function starting
with 'compute' performs the actual computation and saves the results to the features/ directory. The
other functions load the features if they exist, and run the compute functions if they don't.

optical_flow and lucas_kanade_optical_flow are unused in the report and probably don't work.
'''
import os
import pickle

from Clip import Clip

import cv2
import numpy as np


def sum_of_squares_diff(clip):
    name = clip.clip_name
    path = f'features/{name}-sum-of-squares.npy'
    if not os.path.exists(path):
        compute_sum_of_squares_diff(clip)
    X = np.load(path)
    return X


def compute_sum_of_squares_diff(clip):
    name = clip.clip_name
    path = f'features/{name}-sum-of-squares.npy'

    X = np.zeros((len(clip), 1))
    prev = clip[0]

    for i in range(1, len(clip)):
        if i % 100 == 0:
            print(f'frame {i}/{len(clip)}...', end='\r')

        curr = clip[i]
        X[i] = ((curr - prev)**2).sum()
        prev = curr

    print(' ' * 50, end='\r')
    os.makedirs('features', exist_ok=True)
    np.save(path, X)


def keypoint_flow(clip):
    name = clip.clip_name
    path = f'features/{name}-keypoint-matches.pkl'
    if not os.path.exists(path):
        compute_keypoint_matches(clip)
    pickle_f = open(path, 'rb')
    X = np.zeros((len(Clip(name)), 1))

    for i in range(1, len(X)):
        src_pts, dst_pts = pickle.load(pickle_f)
        X[i] = ((src_pts - dst_pts)**2).sum() / max(len(src_pts)**2, 1)

    return X


def compute_keypoint_matches(clip):
    name = clip.clip_name
    path = f'features/{name}-keypoint-matches.pkl'

    detector = cv2.SIFT_create()
    prev = clip[0]
    prev_kp, prev_des = detector.detectAndCompute(prev, None)
    flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

    os.makedirs('features', exist_ok=True)
    pickle_f = open(path, 'wb')

    for i in range(1, len(clip)):
        if i % 100 == 0:
            print(f'frame {i}/{len(clip)}...', end='\r')

        curr = clip[i]
        curr_kp, curr_des = detector.detectAndCompute(curr, None)

        if prev_des is None or curr_des is None:
            src_pts = np.array([])
            dst_pts = np.array([])
        else:
            if len(prev_des) == 1 or len(curr_des) == 1:
                matches = flann.knnMatch(prev_des, curr_des, k=1)
                good = [matches[0][0]]
            else:
                matches = flann.knnMatch(prev_des, curr_des, k=2)
                # store good matches based on Lowe's ratio test
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)

            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good]).reshape(-1, 2)

        pickle.dump([src_pts, dst_pts], pickle_f)

        prev = curr
        prev_kp, prev_des = curr_kp, curr_des

    pickle_f.close()
    print(f'done computing keypoint matches for {name}')


def optical_flow(clip):
    name = clip.clip_name
    path = f'features/{name}-optical-flow'
    if not os.path.exists(path):
        compute_optical_flow(clip)
    clip = Clip(name)
    X = np.zeros((len(clip), 1))

    for i in range(1, len(clip)):
        flow = cv2.imread(f'{path}/clip_{i:05d}.jpg')
        X[i] = flow.sum()

    return X


def compute_optical_flow(clip):
    name = clip.clip_name
    path = f'features/{name}-optical-flow'

    os.mkdir(path)

    prev = clip[0]
    hsv = np.zeros_like(prev)
    hsv[..., 1] = 255
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for i in range(1, len(clip)):
        if i % 100 == 0:
            print(f'frame {i}/{len(clip)}...', end='\r')

        curr = cv2.cvtColor(clip[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(f'{path}/clip_{i:05d}.jpg', bgr)
        prev = curr

    print(f'done computing keypoint matches for {name}')


def lucas_kanade_optical_flow(clip):
    name = clip.clip_name
    path = f'features/{name}-lucas-kanade.pkl'
    if not os.path.exists(path):
        compute_lucas_kanade_optical_flow(clip)
    pickle_f = open(path, 'rb')
    X = np.zeros((len(Clip(name)), 1))

    for i in range(1, len(X)):
        good_new, good_old = pickle.load(pickle_f)
        X[i] = np.linalg.norm(good_new - good_old)

    return X


def compute_lucas_kanade_optical_flow(name):
    clip = Clip(name)
    path = f'features/{name}-lucas-kanade.pkl'
    pickle_f = open(path, 'wb')

    feature_params = {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7}
    lk_params = {
        'winSize': (15, 15),
        'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }
    color = np.random.randint(0, 255, (100, 3))

    old_frame = clip[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    for i in range(1, len(clip)):
        frame = clip[i]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            pickle.dump([good_new, good_old], pickle_f)
        else:
            raise ValueError('p1 is None')

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    pickle_f.close()


def all_features(clip):
    return np.hstack((sum_of_squares_diff(clip), keypoint_flow(clip)))
