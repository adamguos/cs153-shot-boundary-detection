'''
Represents labels for a given clip.

Methods:
---
__init__:
    clip_name: str
    binary: bool, if True then treat all cut types the same (True is used in report)
__getitem__: return label at frame index
    frame_index: int
'''
import os


class Labels:

    def __init__(self, clip_name, binary=True):
        file_name = clip_name + '_hcode.txt'
        if not os.path.isfile(os.path.join('hand_coding', file_name)):
            raise FileNotFoundError(f'{file_name} not found in hand_coding directory')

        self.clip_name = clip_name
        self.file_name = file_name

        self.read_cut_labels(binary)

    def read_all_labels(self):
        with open(os.path.join('hand_coding', self.file_name), 'r') as f:
            # Obtain the number of frames in this clip by looking at the last line
            for line in f:
                pass
            num_frames = int(line.strip().split()[-1])
            labels = [[] for _ in range(num_frames)]

            f.seek(0)

            for line in f:
                line = line.strip()
                if line[:2] == '//' or len(line) == 0:
                    continue

                feature, start_frame, end_frame = line.split()
                start_frame, end_frame = int(start_frame), int(end_frame)
                for i in range(start_frame - 1, end_frame):
                    labels[i].append(feature)

            self.labels = labels
            self.num_frames = num_frames

    def read_cut_labels(self, binary):
        '''
        0 = no cut; 1 = plain cut; 2 = motion matched cut; 3 = cross fade
        '''
        if binary:
            cut_labels = {'c': 1, 'mmc': 1, 'xf': 1}
        else:
            cut_labels = {'c': 1, 'mmc': 2, 'xf': 3}
        with open(os.path.join('hand_coding', self.file_name), 'r') as f:
            # Obtain the number of frames in this clip by looking for the end label
            for line in f:
                tokens = line.strip().split()
                if len(tokens) > 0 and tokens[0] == 'end':
                    num_frames = int(tokens[-1])
                    break
            labels = [0 for _ in range(num_frames)]

            f.seek(0)

            for line in f:
                line = line.strip()
                if line[:2] == '//' or len(line) == 0:
                    continue

                try:
                    feature, start_frame, end_frame = line.split()
                except ValueError as e:
                    breakpoint()
                start_frame, end_frame = int(start_frame), int(end_frame)
                if feature in cut_labels.keys():
                    for i in range(start_frame - 1, end_frame):
                        if labels[i] != 0:
                            raise ValueError(f'Overlapping labels: frame {i}, clip '
                                             f'{self.clip_name}, existing label {labels[i]}, new '
                                             f'label {cut_labels[feature]}')
                        labels[i] = cut_labels[feature]

            self.labels = labels
            self.num_frames = num_frames

    def __getitem__(self, frame_index):
        return self.labels[frame_index]


if __name__ == '__main__':
    labels = Labels('amadeus')
    breakpoint()
