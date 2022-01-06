from face_detector import FaceDetector
import cv2

if __name__=='__main__':
    vid = cv2.VideoCapture('video.mp4')
    f = FaceDetector()
    while True:
        ret,frame = vid.read()
        frame = f.predict(frame)
        cv2.imshow('WebCam',frame)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
