import argparse
import cv2
from model import Trainer
import numpy as np

class FaceDetector:
	def __init__(self):
		self.prototxtpath = "deploy.prototxt.txt"
		self.weightspath = "res10_300x300_ssd_iter_140000.caffemodel"
		self.net = cv2.dnn.readNet(self.prototxtpath, self.weightspath)
		self.min_confidence = 0.50
		self.model = Trainer()
		self.model.load_model()

	def predict(self,image):
		h,w = image.shape[:2]
		blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
		self.net.setInput(blob)
		detections = self.net.forward()
		for i in range(detections.shape[2]):
			confidence = detections[0,0,i,2]
			if confidence > self.min_confidence:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
				face = image[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				label,accu = self.model.predict(face)
				color = (0, 255, 0) if label == "without_mask" else (0, 0, 255)
				if label=="without_mask":
					label = "YES"
				else:
					label = "NO"
				label = "{}: {:.1f}%".format(label, accu * 100)
				cv2.putText(image, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

		return image


if __name__=='__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-i','--image',required=True,help='Path to image')
	arg_parser.add_argument('-o','--output',required=True,help='Path to output image')
	args = vars(arg_parser.parse_args())
	image = cv2.imread(args["image"])
	f = FaceDetector()
	image = f.predict(image)
	#cv2.imshow("Output", image)
	#time.sleep(5)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	cv2.imwrite(args['output'],image)