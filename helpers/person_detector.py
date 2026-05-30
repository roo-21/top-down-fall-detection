import numpy as np
import cv2
import openvino as ov
from pathlib import Path


model_xml = "models/openvino/intel/person-detection-0202/FP16-INT8/person-detection-0202.xml" # detection model path
core = ov.Core()
compiled_model = core.compile_model(model_xml, "CPU") #load and compile the model
infer_request=compiled_model.create_infer_request()
model_in_size=(512, 512) #input image size of the model (width, height)

#class to store keypoints and id of each person in a detection
class Person:
    def __init__(self, detection, id=None):
        self.conf, self.xmin, self.ymin, self.xmax, self.ymax = detection
        self.id=id

    @property
    def bbox(self):
        """Return bbox in [xmin, ymin, xmax, ymax] format"""
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @property
    def center(self):
        """Return center point (cx, cy)"""
        return [(self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2]
    
    
    
#class to store person detection result for a single image/frame
class FrameDet:
    def __init__(self, img, detections, conf_threshold=0.5):
        self.detections = detections[detections[:, 0] > conf_threshold] if detections.size > 0 else np.empty((0, 5)) #filter out low confidence results
        self.img=img
        self.personlist=[Person(det) for det in self.detections] #create a person object for each person detected
        self.annotated_img=self._plot_detections(self.img.copy())

    def _plot_detections(self, img):
        if len(self.personlist)!=0:
            for person in self.personlist:
                x1=int(person.xmin*img.shape[1])
                x2=int(person.xmax*img.shape[1])
                y1=int(person.ymin*img.shape[0])
                y2=int(person.ymax*img.shape[0])
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0),2) # draw bounding box
                label = f"ID:{person.id} {person.conf:.2f}" if person.id else f"{person.conf:.2f}"
                cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) # specify confidence of each box
        return img

    def __str__(self):
        return f"<FrameDet: {len(self.personlist)} persons detected>"
    
def detect_persons(img, conf_threshold=0.5):
    # resizing it, and converting it into openvino tensor of compatible shape (HWC->HCW and adding a batch dimension)
    resized=cv2.resize(img, model_in_size)
    input_tensor = ov.Tensor(array=np.ascontiguousarray(np.expand_dims(np.transpose(resized, (2, 0, 1)),axis=0).astype(np.float32)), shared_memory=True)
    infer_request.set_input_tensor(input_tensor) #pass the input to model
    infer_request.infer() # perform detection
    detections  = infer_request.get_output_tensor(0).data[0,0,:,2:] #extract bounding box coordinates and confidence scores

    return FrameDet(img,detections, conf_threshold)

#function to perform detection frame bu frame, and return input frame, detections (FrameDet object), and fps of the video
def detect_video(path):
    """
    Args:
        path (str|pathlib.Path): path to input video
    
    Returns:
        A dictionary of "frames" "detections" and "fps"
    """
    
    cap=cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return
    else:
        fps=cap.get(cv2.CAP_PROP_FPS)
        frames=[]
        detections=[]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            output=detect_persons(frame)
            frames.append(frame)
            detections.append(output)
            
        cap.release()
    return {"frames": frames, "detections":detections,"fps": fps}

#function to display video with bounding boxes
def display_vid_detections(detections, savefile=None):
    fps=detections["fps"]
    frames = [det.annotated_img for det in detections["detections"]]
    h, w, _ = frames[0].shape
    
    writer = None
    if savefile:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        Path("outputs/video_det/").mkdir(parents=True, exist_ok=True) 
        writer = cv2.VideoWriter(f"outputs/video_det/{savefile}", fourcc, fps if fps > 0 else 30, (w, h))
    if fps>0:
        delay=int(1000/fps)
    else:
        delay=int(1000/30)
    
    display_flag=True
    for frame in frames:
        if not savefile and not display_flag:
            break
        if display_flag:
            cv2.imshow("detected video",frame)
        if writer:
            writer.write(frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            display_flag=False
            cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"Saved annotated video to: outputs/video_det/{savefile}")

if __name__=='__main__':
    path="test_img/people-detection.mp4"
    output=detect_video(path)
    display_vid_detections(output, "out.mp4")
    

    