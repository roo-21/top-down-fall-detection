# mediapipe_top_down_fd

Top-down person tracking + pose estimation + fall detection pipeline.

This README documents the files in this repo as well as the party models that's needed to be downloaded and their target locations.

## Contents of this repo

- `helpers/streamer.py` — video/source streaming utilities.
- `helpers/person_detector.py` — person detection wrapper.
- `helpers/tracker.py` — person tracking logic.
- `helpers/pose_estimator.py` — pose estimation wrapper.
- `helpers/tracker_live.py` — tracking functionality in live feeds.
- `helpers/classifier_ov.py` — OpenVINO-based classification helper.
- `models/classification_head/` - Folder containing classification model files for fall detection.

## Unavailable components
Download these models at specified paths for the system to work. Empty folders are also mentioned here.

- Person detector model path: `models/openvino/intel/person-detection-0202/FP16-INT8/`. The `xml` and `bin` file can be downloaded from [here](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/1/person-detection-0202/FP16-INT8/)
- Pose estimator model path: `models/mediapipe/`. All three versions of the model is directly available from [here](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- `test_` folder is used for storing video files to test the system. currently empty.
- `outputs` detections are written to this folder if *write* argument is set to True 

## Running
- Start at `main.py` to see supported arguments and defaults.
