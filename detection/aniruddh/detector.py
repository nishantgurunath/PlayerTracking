import cv2
import time


def detect(im):
	start_time = time.time()
	gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # HOG needs a grayscale image

	# ================= Haar Cascade Detector ===================
	# person_cascade = cv2.CascadeClassifier('./detection/haarcascade_fullbody.xml')
	# rects = person_cascade.detectMultiScale(gray_im)

	# ================== HOG Detector========================
	hog = cv2.HOGDescriptor()
	# svm = pickle.load(open("svm.pkl"))
	# hog.setSVMDetector(svm)
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	rects, weights = hog.detectMultiScale(gray_im)

	end_time = time.time()
	print("Elapsed time:", end_time-start_time)

	return rects