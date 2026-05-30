from helpers.pose_estimator import PoseDetector, interpolate_nans
from  helpers.person_detector import detect_persons
import numpy as np
import cv2
from  pathlib import Path
from scipy.optimize import linear_sum_assignment
from collections import deque
from helpers.classifier_ov import Model
import time
import os

classification_model_path = "models/classification_head/model.xml"
classifier = Model(classification_model_path)
SEQ_LEN = 45

# Track class to hold and manage information of each traked person detection.
class Track:
    def __init__(self, detection, track_id, init_threshold, init_iou_threshold, pred_interval=30):
        self.id = track_id
        self.bbox = detection.bbox
        self.conf = detection.conf
        self.init_threshold = init_threshold
        self.current_threshold = init_threshold
        self.init_iou_threshold = init_iou_threshold
        self.current_iou_threshold = init_iou_threshold
        self.min_iou_threshold = max(0.0, init_iou_threshold - 0.1)
        self.inactive = 0

        self.pose_history = deque(maxlen=SEQ_LEN)
        self.wlm_history = deque(maxlen=SEQ_LEN)
        self.vis_history = deque(maxlen=SEQ_LEN)
        self.last_pose_frame = -1
        self.last_pred = "NA"
        self.pred_trigger = 1 # prediction performed when this reaches 0
        self.pred_interval = pred_interval  # frames between predictions
        

    def update(self, detection):
        self.bbox = detection.bbox
        self.conf = detection.conf
        self.current_threshold = min(self.init_threshold, max(self.current_threshold, detection.conf))
        self.current_iou_threshold = self.init_iou_threshold
        self.inactive = 0

    def predict(self):
        pose_history = np.array(list(self.pose_history), dtype=np.float32)
        wlm_history = np.array(list(self.wlm_history), dtype=np.float32)
        vis_history = np.array(list(self.vis_history), dtype=np.float32)

        interp_pose_history = pose_history.copy()
        interp_wlm_history = wlm_history.copy()
        pred_vis_history = vis_history.copy()

        low_vis_mask = np.isnan(pred_vis_history) | (pred_vis_history < 0.5)
        interp_pose_history = np.where(low_vis_mask[..., None], np.nan, interp_pose_history)

        interp_pose_history = interpolate_nans(interp_pose_history)
        interp_wlm_history = interpolate_nans(interp_wlm_history)
        pred_vis_history = np.nan_to_num(pred_vis_history, nan=0.0)

        overall_mean_vis = float(pred_vis_history.mean()) if pred_vis_history.size else 0.0
        if overall_mean_vis < 0.5:
            self.last_pred = "NA"
            return

        inputs = [interp_pose_history, interp_wlm_history, pred_vis_history]
        pred=classifier.predict(*inputs).item()
        self.last_pred = "Fall" if pred>0.5 else "No Fall"
       

    def decay_thresholds(self, conf_decay, min_conf, iou_decay):
        self.current_threshold = max(self.current_threshold - conf_decay, min_conf)
        self.current_iou_threshold = max(
            self.current_iou_threshold - iou_decay,
            self.min_iou_threshold
        )

    @property
    def last_pose(self):
        if self.pose_history:
            return self.pose_history[-1]
        return None
    
# Tracker class to manage multiple tracks, perform associations, and handle track lifecycle.
class Tracker:
    def __init__(self, pred_interval = 30, max_inactive=12, iou_threshold=0.3, base_conf=0.5, conf_decay=0.05, min_conf=0.25):
        self.tracks = []
        self.next_id = 1

        self.pred_interval = pred_interval
        self.max_inactive = max_inactive
        self.iou_threshold = iou_threshold
        self.conf_decay = conf_decay
        self.base_conf = base_conf
        self.min_conf = min_conf  

        self.frames_in_tracks=dict()
        
    def _has_overlap_with_existing_track(self, det, overlap_threshold=0.2):
        for track in self.tracks:
            if self._iou(track.bbox, det.bbox) > overlap_threshold:
                return True
        return False

    def update_tracker(self, frame_det):
        detections = frame_det.personlist.copy()

        if self.tracks and detections:
            cost = self._build_cost_matrix(self.tracks, detections)

            row_idx, col_idx = linear_sum_assignment(cost)

            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(row_idx, col_idx):
                if cost[r, c] >= 1e6:
                    continue

                track = self.tracks[r]
                det = detections[c]

                track.update(det)
                det.id = track.id

                matched_tracks.add(r)
                matched_dets.add(c)
                self.frames_in_tracks[track.id] += 1
        else:
            matched_tracks = set()
            matched_dets = set()
        unmatched_tracks = [
            (i, t) for i, t in enumerate(self.tracks)
            if i not in matched_tracks
        ]
        unmatched_dets = [
            (j, d) for j, d in enumerate(detections)
            if j not in matched_dets
        ]
        
        if unmatched_tracks and unmatched_dets:
            # decay thresholds 
            for _, track in unmatched_tracks:
                track.decay_thresholds(self.conf_decay, self.min_conf, 0.05)

            tracks_2 = [t for _, t in unmatched_tracks]
            dets_2 = [d for _, d in unmatched_dets]

            cost2 = self._build_cost_matrix(tracks_2, dets_2)
            row2, col2 = linear_sum_assignment(cost2)

            for r, c in zip(row2, col2):
                if cost2[r, c] >= 1e6:
                    continue

                track_idx = unmatched_tracks[r][0]
                det_idx = unmatched_dets[c][0]

                track = self.tracks[track_idx]
                det = detections[det_idx]

                track.update(det)
                det.id = track.id
                self.frames_in_tracks[track.id] += 1

                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.inactive += 1

        # -------- Remove dead tracks --------
        self.tracks = [
            t for t in self.tracks
            if t.inactive < self.max_inactive
        ]

        # -------- Create new tracks --------
        for j, det in enumerate(detections):
            if j in matched_dets or det.conf < self.base_conf:
                continue

            if self._has_overlap_with_existing_track(det, overlap_threshold=0.2):
                continue
            
            det.id = self.next_id
            self.frames_in_tracks[det.id] = 0
            self.tracks.append(
                Track(
                    det,
                    self.next_id,
                    self.base_conf,
                    self.iou_threshold,
                    self.pred_interval
                )
            )
            self.next_id += 1


    def _build_cost_matrix(self, tracks, detections):
        T, D = len(tracks), len(detections)
        cost = np.full((T, D), 1e6, dtype=np.float32)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):

                if det.conf < track.current_threshold:
                    continue

                iou = self._iou(track.bbox, det.bbox)
                if iou < track.current_iou_threshold:
                    continue

                cost[i, j] = 1.0 - iou

        return cost

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return inter / (areaA + areaB - inter + 1e-6)
    
#subsample a video to reduce fps
def subsample(original_fps, target_fps):
    return max(1, int(original_fps / target_fps))

# Main function to perform detection and tracking on a video, with options for visualization and writing output video.
def detect_with_track(
    path,
    target_fps=12,
    pred_interval=30,
    max_inactive=5,
    iou_threshold=0.3,
    base_conf=0.5,
    conf_decay=0.05,
    min_conf=0.3,
    mp_model="lite",
    write=False,
    show=False,
    draw_bbox=True,
    return_frames=False
):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error: Cannot open video")
        return None

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = target_fps
        
    print(f"Original FPS: {original_fps:.2f}", f"Target FPS: {target_fps}")

    subsample_factor = subsample(original_fps, target_fps)

    pose_detector = PoseDetector(model_version=mp_model)

    tracker = Tracker(
        max_inactive=max_inactive,
        iou_threshold=iou_threshold,
        base_conf=base_conf,
        conf_decay=conf_decay,
        min_conf=min_conf,
        pred_interval=pred_interval
    )

    writer = None

    if write:
        output_dir = "outputs/tracked/"
        os.makedirs(output_dir, exist_ok=True)

        input_name = os.path.basename(path)
        output_name = f"{input_name}_tracked_{int(time.time())}.avi"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            os.path.join(output_dir, output_name),
            cv2.VideoWriter_fourcc(*"MJPG"),
            target_fps,
            (width, height)
        )

    frame_num = 0
    processed_frames = 0

    start = time.monotonic()

    frames = [] if return_frames else None
    detections = [] if return_frames else None
    tracked_frames = [] if return_frames else None
    
    profile_log = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        
        frame_num += 1

        ret, frame = cap.read()

        if not ret:
            break

        if frame is None:
            continue

        # Subsample frames
        if frame_num % subsample_factor != 0:
            continue
        
        frame_start = time.perf_counter()
        
        processed_frames += 1

        # Detection + Tracking
        output = detect_persons(frame, conf_threshold=min_conf)
        
        num_persons = len(output.personlist)

        tracker.update_tracker(output)

        h, w = frame.shape[:2]

        tracked_img = output.img.copy()

        # Per-track processing
        for track in tracker.tracks:

            if track.inactive == 0 and track.conf >= base_conf:

                x1 = int(track.bbox[0] * w)
                y1 = int(track.bbox[1] * h)
                x2 = int(track.bbox[2] * w)
                y2 = int(track.bbox[3] * h)

                # Clamp bbox
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:

                    pose, wlm, vis = pose_detector.detect_pose(
                        crop,
                        return_world_landmarks=True,
                        map_to_image_coordinates={
                            "bbox": track.bbox,
                            "image_size": frame.shape[:2]
                        }
                    )

                else:
                    pose = np.zeros((13, 3), dtype=np.float32)
                    wlm = np.zeros((13, 3), dtype=np.float32)
                    vis = np.zeros((13,), dtype=np.float32)

                track.pose_history.append(pose)
                track.wlm_history.append(wlm)
                track.vis_history.append(vis)

                # Optional consistency with offline tracking
                track.last_pose_frame = frame_num

            else:

                pose = np.zeros((13, 3), dtype=np.float32)
                wlm = np.zeros((13, 3), dtype=np.float32)
                vis = np.zeros((13,), dtype=np.float32)

                track.pose_history.append(pose)
                track.wlm_history.append(wlm)
                track.vis_history.append(vis)

            # Prediction trigger
            if len(track.pose_history) >=SEQ_LEN and track.pred_trigger == 0 or track.inactive == max_inactive or frame_num + subsample_factor >= total_frames:
                track.predict()
                
            if len(track.pose_history) + track.pred_interval > SEQ_LEN:
                track.pred_trigger = (track.pred_trigger + 1) % track.pred_interval
            
            # Draw tracking info
            bbox = track.bbox.copy()

            bbox[0] *= tracked_img.shape[1]
            bbox[1] *= tracked_img.shape[0]
            bbox[2] *= tracked_img.shape[1]
            bbox[3] *= tracked_img.shape[0]

            if draw_bbox:
                cv2.rectangle(
                    tracked_img,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    2
                )

            label = (
                f"ID:{track.id} "
                f"{track.conf:.2f} "
                f"status:{track.last_pred}"
            )

            cv2.putText(
                tracked_img,
                label,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            tracked_img = PoseDetector.draw_landmarks(
                tracked_img,
                pose
            )

        frame_time = time.perf_counter() - frame_start
        fps = 1.0 / frame_time
        
        cv2.putText(
            tracked_img,
            f"FPS: {fps:.2f}",
            (tracked_img.shape[1] - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        
        # Optional visualization
        if show:
            cv2.imshow("Tracked Poses", tracked_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Optional video writing
        if writer is not None:
            writer.write(tracked_img)
        
        # Optional memory storage
        if return_frames:

            frames.append(frame)
            detections.append(output)
            tracked_frames.append(tracked_img)
        
        profile_log.append({
            "frame": processed_frames,
            "fps": fps,
            "num_persons": num_persons,
            "num_tracks": len(tracker.tracks),
            "frame_time_ms": frame_time * 1000
        })

    for track in tracker.tracks:
        track.predict()
    
    # Cleanup    
    cap.release()

    if writer is not None:
        writer.release()

    if show:
        cv2.destroyAllWindows()

    end = time.monotonic()

    detected_fps = processed_frames / (end - start)

    print(f"Processed FPS: {detected_fps:.2f}")

    for k, v in tracker.frames_in_tracks.items():
        print(f"Track ID: {k}, Frames in track: {v}")

    tracked_video = {
        "frames": frames,
        "detections": detections,
        "tracked_frames": tracked_frames,
        "original_fps": original_fps,
        "new_fps": min(original_fps, target_fps),
        "processed_frames": processed_frames,
        "detected_fps": detected_fps,
        "tracker": tracker,
        "profile_log": profile_log
    }

    return tracked_video

def display_vid_track(video_data, display=False, savefile = "output_tracked.mp4"):
    base_path = Path("outputs/video_det")
    base_path.mkdir(parents=True, exist_ok=True)
    path = str(base_path / savefile)

    fps = video_data["new_fps"]
    fps = fps if fps>0 else 30

    frames = video_data["tracked_frames"]
    if not frames:
        print("No frames found in video data.")
        return None
    
    height, width, channels = frames[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    stop_display_flag=False
    
    for frame in frames:
        out.write(frame)
        if display and not stop_display_flag:
            cv2.imshow("Tracked Video", frame)
            if cv2.waitKey(max(int(1000 / fps), 1)) & 0xFF == ord('q'):
                stop_display_flag=True
                
    cv2.destroyAllWindows()
    out.release()

    print(f"Saved tracked video to: {str(path)}")
    
    return path


def display_comparison(video_data, display=False, savefile="comparison.mp4"):
    base_path = Path("outputs/video_det")
    base_path.mkdir(parents=True, exist_ok=True)

    fps = video_data.get("new_fps", 30)
    fps = fps if fps > 0 else 30

    comparison_frames = []
    for det, tracked in zip(video_data["detections"], video_data["tracked_frames"]):
        # Get detection frame
        det_frame = det.annotated_img

        # Resize tracked frame to match detection height if needed
        h1, w1 = det_frame.shape[:2]
        h2, w2 = tracked.shape[:2]
        if h1 != h2:
            tracked = cv2.resize(tracked, (int(w2 * h1 / h2), h1))

        # Add labels
        cv2.putText(det_frame, "Detection Only", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(tracked, "With Tracking", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine side by side
        combined = np.hstack([det_frame, tracked])
        comparison_frames.append(combined)

    if not comparison_frames:
        print("No frames to process.")
        return None

    height, width, _ = comparison_frames[0].shape
    path = str(base_path / savefile)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    stop_display_flag = False
    delay = max(int(1000 / fps), 1)

    for frame in comparison_frames:
        out.write(frame)
        if display and not stop_display_flag:
            cv2.imshow("Comparison", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                stop_display_flag = True
                cv2.destroyWindow("Comparison")

    out.release()
    cv2.destroyAllWindows()

    print(f"Saved comparison video to: {path}")
    return path


if __name__=="__main__":
    path="test_img/people-detection.mp4"
    output=detect_with_track(path, max_inactive=3)
    display_vid_track(output,savefile="out_track.mp4")
    display_comparison(output,display=True, savefile="out_comparison.mp4")
