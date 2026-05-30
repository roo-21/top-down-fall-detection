import numpy as np
import openvino as ov
from pathlib import Path
from sklearn.metrics import accuracy_score

model_path = "models/classification_head/model.xml"
SEQ_LEN = 45
NUM_JOINTS = 13
WORLD_LM_FEATURES = NUM_JOINTS * 3
FRAME_FEATURES = WORLD_LM_FEATURES + 2 + 1 + 1 + 1 # 39 world landmarks, 2 centroid coordinates, 1 height, 1 width, 1 mean visibility

class Model:
    connections=[(0,1), (0,2), (1,2), (1,3), (2,4), (3,5), (4,6), (1,7), (2,8), (7,8), (7,9), (8,10), (9,11), (10,12)]
    def __init__(self, model_path, gpu=False):
        self.device = "GPU" if gpu else "CPU"
        self.compiled_model = self._load_model(model_path, self.device)
        self.in_length = SEQ_LEN

    def _load_model(self, model_path, device="CPU"):
        core = ov.Core()
        compiled_model = core.compile_model(model_path, device)
        return compiled_model

    def predict(self, lm, world_lm, vis):
        features = self._compute_features_from_lms(lm, world_lm, vis)
        if self.in_length > features.shape[0]:
            pad_size = self.in_length - features.shape[0]
            pad = np.zeros((pad_size, features.shape[1]), dtype=features.dtype)
            features = np.concatenate((features, pad), axis=0)
        elif self.in_length < features.shape[0]:
            features = features[:self.in_length, :]
        features = features.astype(np.float32)
        if features.ndim == 2:
            features = np.expand_dims(features, axis=0)
        
        input_tensor = ov.Tensor(array=features, shared_memory=True)
        infer_request = self.compiled_model.create_infer_request()
        infer_request.set_input_tensor(input_tensor)
        infer_request.infer()
        out = infer_request.get_output_tensor(0).data
        return 1 / (1 + np.exp(-out))  # return as probability using sigmoid activation
    
    def _compute_features_from_lms(self, lm, world_lm, vis):
        left_shoulder = lm[:, 1, :2]
        right_shoulder = lm[:, 2, :2]
        left_hip = lm[:, 7, :2]
        right_hip = lm[:, 8, :2]

        torso_centroid = (left_shoulder + right_shoulder + left_hip + right_hip) / 4

        widths = lm[:, :, 0].max(axis=1) - lm[:, :, 0].min(axis=1)
        heights = lm[:, :, 1].max(axis=1) - lm[:, :, 1].min(axis=1)

        mean_vis = np.nanmean(vis, dtype=np.float32)
        if np.isnan(mean_vis):
            mean_vis = 0.0
        mean_vis_column = np.full((world_lm.shape[0], 1), mean_vis, dtype=np.float32)

        data = np.concatenate(
            (   
                world_lm.reshape(world_lm.shape[0], WORLD_LM_FEATURES),
                torso_centroid,
                heights.reshape(-1, 1),
                widths.reshape(-1, 1),
                mean_vis_column,
            ),
            axis=1,
        ).astype(np.float32)
        assert data.shape[1] == FRAME_FEATURES, f"expected {FRAME_FEATURES} features, got {data.shape[1]}"

        return data

if __name__ == "__main__":
    
    model = Model(model_path)
    pose_data_path="test_img/Pose_data"
    data={}
    for file in Path(pose_data_path).iterdir():
        if file.suffix==".npz":
            pose_data=np.load(file, allow_pickle=True)
            if len(data.keys())==0:
                data["filename"]=file.stem
                for key in pose_data.keys():
                    data[key]=[]
            for key in pose_data.keys():
                data[key].append(pose_data[key])

    lms=np.array(data["pose"])
    world_lms=np.array(data["world"])
    labels=np.array(data["label"])

    if SEQ_LEN < lms.shape[1]:
        lms=lms[:,:SEQ_LEN]
        world_lms=world_lms[:,:SEQ_LEN]
    synthetic_lm = np.zeros((SEQ_LEN, NUM_JOINTS, 3), dtype=np.float32)
    synthetic_world = np.zeros((SEQ_LEN, NUM_JOINTS, 3), dtype=np.float32)
    synthetic_vis = np.ones((SEQ_LEN, NUM_JOINTS), dtype=np.float32)
    synthetic_features = model._compute_features_from_lms(synthetic_lm, synthetic_world, synthetic_vis)
    assert synthetic_features.shape == (SEQ_LEN, FRAME_FEATURES), synthetic_features.shape
    predictions=[]
    vis_data = np.array(data["visibility"]) if "visibility" in data else np.ones((len(lms), SEQ_LEN, NUM_JOINTS), dtype=np.float32)
    for lm, world_lm, vis, label in zip(lms, world_lms, vis_data, labels):
        predictions.append(int(model.predict(lm, world_lm, vis)[0]>0.5))

        print(f"Label: {label}, Prediction: {predictions[-1]}")
    print("\n-----------------------------------\n")
    print(f"Accuracy: {accuracy_score(labels, predictions)*100:<.2f}%")