import os
from helpers.pose_estimator import PoseDetector, interpolate_nans
from helpers.classifier_ov import Model
import numpy as np
from helpers.tracker import SEQ_LEN, Tracker
from helpers.streamer import VideoStream
from helpers.person_detector import detect_persons
import cv2
import time


classification_model_path = "models/classification_head/model.xml"
classifier = Model(classification_model_path)

# function to run fall detection on livestream videos. offline version of this function can be found in tracker.py
def detect_with_track_live(
    camera, 
    target_fps=15, 
    pred_interval=30,
    max_inactive=5, 
    iou_threshold=0.3, 
    base_conf=0.5, 
    conf_decay=0.05, 
    min_conf=0.3, 
    mp_model="full",
    write=False, 
    draw_bbox=True
):
    if write:
        output_dir = "outputs/livestream/"
        os.makedirs(output_dir, exist_ok=True)
        output_name = f"tracked_output_{int(time.time())}.avi"
        writer = cv2.VideoWriter(output_dir + output_name, cv2.VideoWriter_fourcc(*"MJPG"), target_fps, (640, 480))
    pose_detector = PoseDetector(model_version=mp_model)
    tracker = Tracker(max_inactive=max_inactive, iou_threshold=iou_threshold, base_conf=base_conf, conf_decay=conf_decay, min_conf=min_conf, pred_interval=pred_interval)
    

    frame_num=0
    vs = VideoStream(src=camera, fps=target_fps)
    start = time.monotonic()
    while True:
        frame_num+=1
        frame = vs.read()
        if frame is None:
            continue
        
        output = detect_persons(frame, conf_threshold=min_conf)
        tracker.update_tracker(output)

        h, w = frame.shape[:2]

        if len(tracker.tracks) > 0:
            print(output)
        tracked_img = output.img.copy()
        for track in tracker.tracks:
            
            if track.inactive == 0 and track.conf >= base_conf:

                x1 = int(track.bbox[0] * w)
                y1 = int(track.bbox[1] * h)
                x2 = int(track.bbox[2] * w)
                y2 = int(track.bbox[3] * h)
                crop = frame[y1:y2, x1:x2]

                pose, wlm, vis = pose_detector.detect_pose(
                    crop,
                    return_world_landmarks=True,
                    map_to_image_coordinates={
                        "bbox": track.bbox,
                        "image_size": frame.shape[:2]
                    }
                )
                
                track.pose_history.append(pose)
                track.wlm_history.append(wlm)
                track.vis_history.append(vis)
            else:
                pose = np.zeros((13,3), dtype=np.float32)
                wlm = np.zeros((13,3), dtype=np.float32)
                vis = np.zeros((13,), dtype=np.float32)
                track.pose_history.append(pose)
                track.wlm_history.append(wlm)
                track.vis_history.append(vis)
            if len(track.pose_history) >=SEQ_LEN and track.pred_trigger == 0 or track.inactive == max_inactive-5:
                track.predict()
            if len(track.pose_history) + track.pred_interval > SEQ_LEN:
                track.pred_trigger = (track.pred_trigger + 1) % track.pred_interval
            bbox = track.bbox.copy()
            bbox[0]*=tracked_img.shape[1]
            bbox[1]*=tracked_img.shape[0]
            bbox[2]*=tracked_img.shape[1]
            bbox[3]*=tracked_img.shape[0]
            # Draw bounding box
            if draw_bbox:
                cv2.rectangle(tracked_img,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 255, 0), 2)

            # Draw track ID
            label = f"ID:{track.id} {track.conf:.2f} status: {track.last_pred}"
            cv2.putText(tracked_img, label,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            tracked_img = PoseDetector.draw_landmarks(tracked_img, pose)

        cv2.imshow("Tracked Poses", tracked_img)
        if write:
            writer.write(tracked_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end = time.monotonic()
    detected_fps = frame_num / (end - start)
    print("FPS:", detected_fps)

if __name__ == "__main__":
    detect_with_track_live(0, target_fps=15)
