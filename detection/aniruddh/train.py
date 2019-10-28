import cv2
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import re
import pickle

pos_path = 'C:/Users/aniru/OneDrive/Desktop/final/dataset/NCAA/'
neg_path = 'C:/Users/aniru/OneDrive/Desktop/final/dataset/neg/'
width, height = 64, 128

def getPos(data_path):
	samples = []
	hog = cv2.HOGDescriptor()
	for game in os.listdir(data_path)[200:]:
		for clip in os.listdir(data_path + game):
			frames = os.listdir(data_path + game + '/' + clip)
			N = (len(frames) - 1) // 2
			for i in range(N):
				# im data
				im_path = data_path + game + '/' + clip + '/' + '{:02}.png'.format(i + 1)
				im = np.uint8(cv2.imread(im_path))

				# box data
				label_path = data_path + game + '/' + clip + '/' + '{:02}_info.csv'.format(i + 1)
				label = pd.read_csv(label_path)
				for _, bbox in label.iterrows():
					x0, y0, x1, y1 = int(bbox.x), int(bbox.y), int(bbox.x + bbox.w), int(bbox.y + bbox.h)
					x0, y0 = max(0, x0), max(0,y0)
					im_crop = cv2.resize(im[y0:y1, x0:x1], (width, height))
					samples.append(hog.compute(im_crop))
	return np.array(samples)

def getNeg(data_path):
	samples = []
	hog = cv2.HOGDescriptor()
	for file in os.listdir(data_path):
		im = np.uint8(cv2.imread(data_path + file))
		im_crop = cv2.resize(im, (width, height))
		samples.append(hog.compute(im_crop))
	return np.array(samples)

def main():
	# pos = getPos(pos_path)
	# neg = getNeg(neg_path)

	# p, n = pos.shape[0], neg.shape[0]
	# samples = np.squeeze(np.concatenate((pos, neg), axis=0))
	# labels = np.concatenate((np.ones(p), np.zeros(n)))
	# labels = np.int32(np.expand_dims(labels,1))

	# print (samples.shape, labels.shape)
	# # # Shuffle Samples
	# rand = np.random.RandomState(321)
	# shuffle = rand.permutation(len(samples))
	# samples = samples[shuffle]
	# labels = labels[shuffle]    

	# # Create SVM classifier
	# svm = cv2.ml.SVM_create()
	# svm.setKernel(cv2.ml.SVM_LINEAR)
	# svm.setType(cv2.ml.SVM_C_SVC)
	# svm.setC(2.67)
	# svm.setGamma(5.383)

	# # train the classifier
	# svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
	# svm.save('svm.xml')
	tree = ET.parse('svm.xml')
	root = tree.getroot()
	# now this is really dirty, but after ~3h of fighting OpenCV its what happens :-)
	SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0] 
	rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
	svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
	svmvec.append(-rho)
	pickle.dump(svmvec, open("svm.pkl", 'wb'))

if __name__ == "__main__":
	main()
