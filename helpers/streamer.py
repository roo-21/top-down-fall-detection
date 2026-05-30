import time
import cv2
from threading import Thread, Condition
import time

class VideoStream:
    def __init__(self, src=0, fps=15, cam_test_duration=5):
        """
            Reads frames from webcam, and resamples them to meet the target FPS.
        Args:
            src (int): Camera ID. Defaults to 0.
            fps (int): Target FPS. Defaults to 15.
            cam_test_duration (int): Initial Camera FPS measure duration. Defaults to 10.
        """
        
        self.cap = cv2.VideoCapture(src)
        source_fps = self._get_cam_fps(duration=cam_test_duration)
        self.step_size = self.get_resample_step_size(fps, source_fps)
        self.cond = Condition()
        self.frame = None
        self.frame_id = 0
        self.last_id = -1
        self.running = True
        self.phase = 0.0
        self.cap=cv2.VideoCapture(src)
        Thread(target=self.update, daemon=True).start()


    def update(self):
        while self.running:
            self.phase += self.step_size
            if self.phase >= 1.0:
                ret, frame = self.cap.read()
                if not ret:
                    self.running=False
                    self.cond.notify_all()
                    break
                with self.cond:
                    self.frame = frame
                    self.frame_id = (self.frame_id + 1)
                    self.cond.notify_all()
                self.phase -= 1.0
            else:
                self.cap.read()
            time.sleep(0.001)

    def read(self):
        with self.cond:
            while self.frame_id == self.last_id and self.running:
                self.cond.wait()
            self.last_id = self.frame_id
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

    def _get_cam_fps(self, duration):
        print(f"Testing camera FPS. Please wait {duration}s...")
        frame_count=0
        start=time.monotonic()
        while True:
            self.cap.grab()
            frame_count+=1
            if time.monotonic()-start>=duration:
                elapsed=time.monotonic()-start
                fps=round(frame_count/elapsed)
                print(f"Camera FPS: {fps}")
                return fps
            

    def get_resample_step_size(self, target_fps, src_fps):
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than 0")
        if src_fps <= 0:
            raise ValueError("src_fps must be greater than 0")
        if target_fps>=src_fps:
            return 1
        step = target_fps / src_fps
        return step


if __name__=="__main__":
    vs=VideoStream(cam_test_duration=30)
    while True:
        frame=vs.read()
        if frame is None:
            continue
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
