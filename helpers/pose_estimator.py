import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import cv2
from helpers.person_detector import FrameDet, detect_persons
from pathlib import Path

# PoseDetector class that wraps mediapipe pose landmarker and helper functions
class PoseDetector:
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    joint_subset_idx=[0,11,12,13,14,15,16,23,24,25,26,29,30]
    connections=[(0,1), (0,2), (1,2), (1,3), (2,4), (3,5), (4,6), (1,7), (2,8), (7,8), (7,9), (8,10), (9,11), (10,12)]

    def __init__(self,model_version="lite"):

        self._model_path = f"models/mediapipe/pose_landmarker_{model_version}.task"
        self.options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self._model_path),
            running_mode=self.VisionRunningMode.IMAGE)

        self.landmarker=self.PoseLandmarker.create_from_options(self.options)

    def detect_pose(self, bgr_image, return_world_landmarks=False, map_to_image_coordinates=None):
        # Handle empty or invalid images by returning NaN arrays
        if bgr_image is None or bgr_image.size == 0:
            if not return_world_landmarks:
                return np.full((13,3), np.nan),np.full((13), np.nan)
            else:
                return np.full((13,3), np.nan), np.full((13,3), np.nan),np.full((13), np.nan)
        
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = self.landmarker.detect(mp_image)
        if not result.pose_landmarks:
            if not return_world_landmarks:
                return np.full((13,3), np.nan),np.full((13), np.nan)
            else:
                return np.full((13,3), np.nan), np.full((13,3), np.nan),np.full((13), np.nan)

        normalized_lm = np.array([(result.pose_landmarks[0][idx].x, result.pose_landmarks[0][idx].y, result.pose_landmarks[0][idx].z) for idx in self.joint_subset_idx])
        visibility = np.array([result.pose_world_landmarks[0][idx].visibility for idx in self.joint_subset_idx])

        # map the coordinates from bounding box to full image 
        if map_to_image_coordinates:
            normalized_lm = self._correct_pose_landmarks(normalized_lm, map_to_image_coordinates["bbox"])
        if return_world_landmarks:
            world_lm=np.array([(result.pose_world_landmarks[0][idx].x, result.pose_world_landmarks[0][idx].y, result.pose_world_landmarks[0][idx].z) for idx in self.joint_subset_idx])
            return normalized_lm, world_lm, visibility
        else:
            return normalized_lm, visibility
        
    def _correct_pose_landmarks(self, landmarks, bbox):
        """Convert normalized pose landmarks from bbox coordinates to image coordinates"""
        for lm in landmarks:
            lm[0] = (bbox[0] + lm[0] * (bbox[2] - bbox[0]))
            lm[1] = (bbox[1] + lm[1] * (bbox[3] - bbox[1]))
        return landmarks
    
    @classmethod
    def draw_landmarks(cls, image, pose_landmarks):
        arr=np.nan_to_num(pose_landmarks, nan=0)
        if np.all(arr==0):
            return image

        h, w = image.shape[:2]
        for lm in arr:
            x = int(lm[0] * w)
            y = int(lm[1] * h)
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

        for a,b in cls.connections:
            x1 = int(arr[a, 0] * w)
            y1 = int(arr[a, 1] * h)
            x2 = int(arr[b, 0] * w)
            y2 = int(arr[b, 1] * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return image

    def detect_and_draw(self, image, map_to_image_coordinates=None):
        pose_landmarks, _ = self.detect_pose(image, map_to_image_coordinates=map_to_image_coordinates)
        annotated_img=self.draw_landmarks(image.copy(), pose_landmarks)
        return annotated_img

def interpolate_nans(seq):
        """
        Linearly interpolate NaNs over time for each joint/channel
        """
        seq = np.array(seq, dtype=np.float32, copy=True)
        T, J, C = seq.shape
        for j in range(J):
            for c in range(C):
                x = seq[:, j, c]
                idx = np.arange(T)
                mask = ~np.isnan(x)
                if not mask.any():
                    seq[:, j, c] = 0.0
                elif mask.all():
                    continue
                else:
                    seq[:, j, c] = np.interp(idx, idx[mask], x[mask])
        return seq

# The class and functions below are only used for testing the PoseDetector class and helpers. They are not relevant anywhere else in this project.
       
class FramePoseDet():
    # performs cropping each detected person from the frame, detects pose for each cropped person, and maps coordinates back to original frame's coordinate space.
    def __init__(self, frameDet: FrameDet, pose_detector, world_landmarks=False):
        self.frameDet=frameDet 
        self.pose_detector=pose_detector
        self._detect_poses(return_world_landmarks=world_landmarks)
    
    def pose_image(self, draw_bbox=False):
        image=self.frameDet.img.copy()
        for pose_landmarks in self.pose_landmarks_list:
            image=PoseDetector.draw_landmarks(image, pose_landmarks)

        if draw_bbox:
            for person in self.frameDet.personlist:
                x1=int(person.xmin*image.shape[1])
                x2=int(person.xmax*image.shape[1])
                y1=int(person.ymin*image.shape[0])
                y2=int(person.ymax*image.shape[0])
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0),2)

        return image
        
        
    def _detect_poses(self, return_world_landmarks=False):
        cropped_images, h, w, boxes = self._crop_detected_persons()
        self.pose_landmarks_list=[]
        if return_world_landmarks:
            self.world_landmarks_list=[]
        for idx, img in enumerate(cropped_images):
            if return_world_landmarks:
                pose_landmarks, world_landmarks, _ = self.pose_detector.detect_pose(img, return_world_landmarks=True, map_to_image_coordinates={"bbox": boxes[idx], "image_size": (h, w)})
                self.pose_landmarks_list.append(pose_landmarks)
                self.world_landmarks_list.append(world_landmarks)
            else:
                pose_landmarks, _ = self.pose_detector.detect_pose(img, return_world_landmarks=False, map_to_image_coordinates={"bbox": boxes[idx], "image_size": (h, w)})
                self.pose_landmarks_list.append(pose_landmarks)

    def _crop_detected_persons(self):
        cropped_images=[]
        image_size=self.frameDet.img.shape[:2]
        h, w = image_size
        boxes=[]
        for person in self.frameDet.personlist:
            bbox=person.bbox
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            cropped_image=self.frameDet.img[y1:y2, x1:x2]
            cropped_images.append(cropped_image)
            boxes.append(bbox)
        return cropped_images, h, w, boxes

def detect_video_pose(path, pose_detector):
    video_capture = cv2.VideoCapture(str(path))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames = []
    pose_detections = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
        frame_det = detect_persons(frame)
        frame_pose_det = FramePoseDet(frame_det, pose_detector=pose_detector)
        pose_detections.append(frame_pose_det)

    video_capture.release()
    return {"frames": frames, "pose_detections": pose_detections, "fps": fps}  

def display_vid_pose_detections(pose_detections, savefile=None, show_bbox=False):
    fps = pose_detections["fps"]
    frames = [det.pose_image(draw_bbox=show_bbox) for det in pose_detections["pose_detections"]]
    h, w, _ = frames[0].shape

    writer = None
    if savefile:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        Path("outputs/video_pose/").mkdir(parents=True, exist_ok=True) 
        writer = cv2.VideoWriter(f"outputs/video_pose/{savefile}", fourcc, fps if fps > 0 else 30, (w, h))
    if fps > 0:
        delay = int(1000 / fps)
    else:
        delay = int(1000 / 30)

    display_flag = True
    for frame in frames:
        if not savefile and not display_flag:
            break
        if display_flag:
            cv2.imshow("pose estimated video", frame)
        if writer:
            writer.write(frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            display_flag = False
            cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"Saved annotated video to: outputs/video_pose/{savefile}")
        
if __name__=="__main__":
    path="test_img/people-detection.mp4"
    pose_detector=PoseDetector(model_version="heavy")  
    output=detect_video_pose(path, pose_detector=pose_detector)
    display_vid_pose_detections(output, "out.mp4", show_bbox=True)
    pose_detector.landmarker.close()

    

