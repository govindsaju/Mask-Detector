import numpy as np
import cv2
import sklearn
import os

class Loader:
    def __init__(self):
        self.img_dims = (60,60)
        train_dir = 'data/'
        test_dir = 'test/'
        classes = ['with_mask','without_mask']
        self.train_data = []
        self.train_labels = []
        for i in classes:
            path = train_dir+i
            for file in os.listdir(path):
                img = cv2.imread(path+'/'+file)
                img = cv2.resize(img,self.img_dims)
                self.train_data.append(img)
                self.train_labels.append(i)
        self.train_data,self.train_labels = sklearn.utils.shuffle(self.train_data,self.train_labels)

        self.test_data = []
        self.test_labels = []
        for i in classes:
            path = test_dir+i
            for file in os.listdir(path):
                img = cv2.imread(path+'/'+file)
                img = cv2.resize(img,self.img_dims)
                self.test_data.append(img)
                self.test_labels.append(i)
        self.test_data,self.test_labels = sklearn.utils.shuffle(self.test_data,self.test_labels)

        self.train_data = np.array(self.train_data,dtype=np.uint8)
        self.test_data = np.array(self.test_data,dtype=np.uint8)
        self.train_labels = np.array(self.train_labels)
        self.test_labels = np.array(self.test_labels)


    def preview(self):
        indices = np.random.randint(0, self.test_data.shape[0], (4,4))
        images = [ [ self.test_data[indices[i,j]] for j in range(4)] for i in range(4) ]
        final = np.full((60 * 4 + 4 * 20, 60 * 4 + 3 * 5,3),255,np.uint8)
        for i in range(4):
            for j in range(4):
                final[j*(60+20):j*(60+20)+60,i*(60+5):i*(60+5)+60] = images[i][j]
                final = cv2.putText(final,
                    str('Yes' if self.test_labels[indices[i,j]]=='with_mask' else 'No'),
                    (i*(60+5), (j+1)*(60+18)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0,0,0),
                    1,
                    cv2.LINE_AA)

        final = cv2.resize(final, (0,0), fx=2, fy=2)
        cv2.imshow('Preview', final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__=='__main__':
    l = Loader()
    l.preview()


