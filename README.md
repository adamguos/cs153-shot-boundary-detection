# Automatic Feature Labeling: Shot Boundary Detection

## Acknowledgements

[Tomáš Souček and Jakub Lokoč, TransNet V2: An effective deep network architecture for fast
shot transition detection](https://github.com/soCzech/TransNetV2)

## Setup

Install required packages using pip: `pip install -r requirements.txt`. Also install Tensorflow.

Download the dataset and place the files like so:

```
shot_boundary_detection
--> clips
    --> amadeus.mp4
    --> ...
--> frames
    --> amadeus
        --> clip_00001.jpg
        --> ...
    --> ...
--> hand_coding
    --> amadeus_hcode.txt
    --> ...
--> ...
```

Use `Clip` and `Labels` to access the clip frames and hand-coded labels.

```
from Clip import Clip
from Labels import Labels

amadeus_clip = Clip('amadeus')
amadeus_labels = Labels('amadeus')

amadeus_clip[0]     # returns a 3D np.array containing the first frame of
                    # amadeus
amadeus_labels[0]   # returns label of first frame of amadeus (0 = no cut,
                    # 1 = cut)
```

## Computing the features

Use methods in `feature_extraction`. These methods save their outputs to the `features` directory
and load them next time automatically. If you want to re-compute the features, delete the
corresponding file in the `features` directory.

```
import feature_extraction

sos = feature_extraction.sum_of_squares_diff(amadeus_clip)
keypoint = feature_extraction.keypoint_flow(amadeus_clip)
```

## Using the models

Use `ThresholdClassifier` and `SVM`. Both conform to `sklearn` classifier API.

```
from ThresholdClassifier import ThresholdClassifier
from SVM import SVM

tc = ThresholdClassifier()
tc.fit(X_train)
tc.predict(X_test)

svm = SVM()
svm.fit(X_train)
svm.predict(X_test)

from sklearn.model_selection import cross_validate
cross_validate(tc, X, y)
cross_validate(svm, X, y)
```

`TransNetV2` is also setup and available to use.

```
tn = TransNetV2()
video_frames, single_frame_predictions, all_frame_predictions = \
    transnet.predict_video('clips/amadeus.mp4')
scenes = transnet.predictions_to_scenes(single_frame_predictions)

# Use the following to get predictions in the same format as
# ThresholdClassifier and SVM
pred = np.zeros(len(clip), dtype=int)
pred[scenes[:, 1][:-1]] = 1
pred[scenes[:, 0][1:]] = 1
```

See [this issue](https://github.com/soCzech/TransNetV2/issues/1#issuecomment-647357796) if you run
into a problem with corrupted weights.

## Running the tests

Run `python tests.py`. F1-scores are written to `stdout` and saved to `scores.csv`.
