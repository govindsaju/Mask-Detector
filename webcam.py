from face_detector import FaceDetector
import cv2
import time
import argparse

if __name__=='__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-i','--input',required=True,help='Path to video')
	arg_parser.add_argument('-o','--output',required=True,help='Path to output video')
	args = vars(arg_parser.parse_args())
	vid = cv2.VideoCapture(args['input'])
	frame_width = int(vid.get(3))
	frame_height = int(vid.get(4))
	size = (frame_width, frame_height)
	vidwriter = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'),5,size)
	counter = 0
	f = FaceDetector()
	while True:
		ret,frame = vid.read()
		if ret==True:
			frame = f.predict(frame)
			vidwriter.write(frame)
			if counter%10==0:
				cv2.imwrite('temp/'+str(int(counter//10))+'.jpg',frame)
				print(counter)
			counter+=1
		else:
			break

	vid.release()
	vidwriter.release()
