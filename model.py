from loader import Loader
import argparse
import numpy as np
import os
import cv2
import tensorflow as tf
from datetime import datetime

# Images of size 60x60, divided by 255


class Trainer:
	def __init__(self):

		np.random.seed(12345)
		tf.random.set_seed(12345)

		self.num_epochs = 5

		self.model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255,input_shape=(60,60,3)),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(2),
  tf.keras.layers.Softmax()
])
		self.loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.optimizer = 'adam'
		self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=['accuracy'])

	def load_data(self):

		self.loader = Loader()
		self.train_data = tf.convert_to_tensor(self.loader.train_data/255.0)
		self.test_data = tf.convert_to_tensor(self.loader.test_data/255.0)
		l = []
		t = []
		for i in range(self.loader.train_labels.shape[0]):
			if self.loader.train_labels[i]=='with_mask':
				l.append(1)
			else:
				l.append(0)
		for i in range(self.loader.test_labels.shape[0]):
			if self.loader.test_labels[i]=='with_mask':
				t.append(1)
			else:
				t.append(0)
		self.train_labels = np.array(l)
		self.test_labels = np.array(t)
		print("Loading complete")

	def save_model(self):
		self.model.save('saved_model')

	def load_model(self):
		if os.path.exists('saved_model'):
			self.model = tf.keras.models.load_model('saved_model')
		else:
			raise Exception('Model not trained')

	def train(self):
		if not self.model:
			return
		self.model.fit(self.train_data,self.train_labels,epochs=self.num_epochs,batch_size=24)
		self.save_model()

	def test(self):
		self.model.evaluate(self.test_data,self.test_labels)

	def predict(self, image):
		image = cv2.resize(image,(60,60))
		image = image/255
		image = np.expand_dims(image,0)
		prediction = 'None'
		if not self.model:
			return prediction

		prediction = self.model.predict(image)
		ind = np.argmax(prediction)
		classes = ['without_mask','with_mask']
		print(prediction)
		return (classes[ind],prediction[0,ind])



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Trainer')
	parser.add_argument('-train', action='store_true', help='Train the model')
	parser.add_argument('-test', action='store_true', help='Test the trained model')
	parser.add_argument('-preview', action='store_true', help='Show a preview of the loaded test images and their corresponding labels')
	parser.add_argument('-predict', action='store_true', help='Make a prediction on a randomly selected test image')

	options = parser.parse_args()

	t = Trainer()
	if options.train:
		t.load_data()
		t.train()
		t.train()
		t.test()
	if options.test:
		t.load_data()
		t.load_model()
		t.test()
	if options.preview:
		t.load_data()
		t.loader.preview()
	if options.predict:
		t.load_data()
		try:
			t.load_model()
		except:
			t.train()
		np.random.seed(int(round(datetime.now().timestamp())))
		i = np.random.randint(0,t.loader.test_data.shape[0])

		print(f'Predicted: {t.predict(t.loader.test_data[i])}')
		print(f'Actual: {t.loader.test_labels[i]}')

		image = t.loader.test_data[i]
		image = cv2.resize(image, (0,0), fx=6, fy=6)
		cv2.imshow('Face', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()