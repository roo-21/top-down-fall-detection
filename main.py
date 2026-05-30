from helpers.tracker import detect_with_track
from helpers.tracker_live import detect_with_track_live
# camera index or video path as path "variable". 
path = "test_/S1-P2-F1-FALL-2.mp4"

"""
    arguments:
        path: video path or camera index
        target_fps: target frames per second of output video    
        pred_interval: number of frames between each fall prediction
        max_inactive: number of frames before deleting a missing ID
        iou_threshold: minimum IOU threshold for matching detections to existing IDs
        base_conf: least confidence for new detections to be considered valid
        conf_decay: confidence reduce base_conf and iou_threshold by this value every missed detection (limited by default minimumn value for both)
        min_conf: minimum confidence for detections to be considered valid (same value for minimum iou_threshold)        
        mp_model: mediapipe pose model to use ("lite", "full", "heavy")
        write: whether to write output video to file
        show: whether to display output video in a window (only if input is video path. always displayed if livestream)
        draw_bbox: whether to draw bounding boxes around detected people
        return_frames: whether to return list of output frames (only if input is video path)
"""
if type(path) == str:
    detect_with_track(
        path,
        target_fps=15,
        pred_interval=20,
        max_inactive=10,
        iou_threshold=0.3,
        base_conf=0.5,
        conf_decay=0.05,
        min_conf=0.1,
        mp_model="full",
        write=False,
        show=True,
        draw_bbox=False,
        return_frames=False
    )
elif type(path) == int:
    detect_with_track_live(
        camera=path,
        target_fps=15,
        pred_interval=20,
        max_inactive=10,
        iou_threshold=0.3,
        base_conf=0.5,
        conf_decay=0.05,
        min_conf=0.1,
        mp_model="full",
        write=False,
        draw_bbox=False
    )