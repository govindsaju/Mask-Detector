from face_detector import FaceDetector
import cv2

class VideoWrapper:
    def __init__(self,videoname='video.mp4'):
        self.model = FaceDetector()
        self.video = cv2.VideoCapture(videoname)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        ret,frame = self.video.read()
        if ret==False:
            return None
        frame = self.model.predict(frame)
        ret,frame = cv2.imencode('.jpg', frame)
        return frame.tobytes()
