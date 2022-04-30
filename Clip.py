'''
Represents the frames of a clip.

Methods:
---
__init__:
    clip_name: str
__getitem__: returns 3D np.array representing a frame
    frame_index: int
__len__: returns number of frames in the clip
'''
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Clip:
    def __init__(self, clip_name):
        if not os.path.isdir(os.path.join('frames', clip_name)):
            raise FileNotFoundError(f'{clip_name} not found in frames directory')

        self.clip_name = clip_name
        self.clip_dir = os.path.join('frames', clip_name)
        self.frame_names = sorted(os.listdir(self.clip_dir))

    def __getitem__(self, frame_index):
        if frame_index >= len(self.frame_names):
            raise IndexError(
                f'{frame_index} is beyond the length of {self.clip_name} ({len(self.frame_names)})')
        return cv2.imread(os.path.join(self.clip_dir, self.frame_names[frame_index]))

    def __len__(self):
        return len(self.frame_names)


if __name__ == '__main__':
    clip = Clip('amadeus')
    diffs = np.zeros(len(clip.frame_names) - 1)
    for i in range(len(clip.frame_names) - 1):
        diffs[i] = np.linalg.norm(clip[i] - clip[i + 1])
    plt.scatter(np.arange(len(clip.frame_names) - 1), diffs)
    plt.show()
