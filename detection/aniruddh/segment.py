import cv2
import os
import numpy as np
import time
from scipy.cluster.vq import vq, kmeans

datapath = 'C:/Users/aniru/OneDrive/Desktop/final/dataset/'
# Color ranges
# lower_green = np.array([40,0, 0])
# upper_green = np.array([70, 255, 255])

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

lower_red = np.array([0,31,255])
upper_red = np.array([176,255,255])

lower_white = np.array([0,0,0])
upper_white = np.array([0,0,255])

def cluster(image, K):
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	Z = hsv.reshape((-1,3))
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res = res.reshape((image.shape))

	res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
	res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	return res_gray

def get_dominant(hsv):
	h, w, _ = hsv.shape
	hist, edges = np.histogram(hsv[...,0], bins= 12, range=(0,360))
	idx = np.argmax(hist)
	lower = np.array([edges[idx], 0, 0])
	upper = np.array([edges[idx + 1], 255 , 255])

	if hist[idx] / (h*w) > 0.6 : view = 1
	else: view = 0
	return lower, upper, view

def get_playable(hsv):
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

def detect(hsv, lower, upper):
    # applying mask and get grayscale
	mask = cv2.inRange(hsv, lower, upper)
	res = cv2.bitwise_and(hsv, hsv, mask=mask)
	res_gray = res[..., -1]

	# res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
	# cv2.imshow('Gray Scale', res_gray)

    # removing audience using morphology operation
	thresh = cv2.threshold(res_gray,100,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	# cv2.imshow('Threshold', thresh)
	kernel = np.ones((13,13),np.uint8)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	cv2.imshow('morphology', thresh)

    #find contours in threshold image  
	contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(image, contours, -1, (0,255,0), 2)
	# cv2.imshow('contours', image)

	bbox = []
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		# if(h>=(1.5)*w):
		# 	if(w>15 and h>= 15):
		area = w * h
		if (100 < area < 15000 and h > 50):
				bbox.append([x,y,w,h])
	return np.array(bbox)

def clf_team(bbox, image):
	font = cv2.FONT_HERSHEY_SIMPLEX
	x,y,w,h = bbox
	player_img = image[y:y+h,x:x+w]
	player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)

	#If player has blue jersy
	mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
	res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
	res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
	res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
	nzCount = cv2.countNonZero(res1)

	#If player has red jersy
	mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
	res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
	res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
	res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
	nzCountred = cv2.countNonZero(res2)
	if(nzCount > nzCountred):
		# cv2.putText(image, 'France', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

	else:
		# cv2.putText(image, 'Belgium', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
	


def main():
	#Read the video frame by frame
	for count in range(800):
		impath = datapath+'frames/%d.png' % (count)
		image = cv2.imread(impath)

		hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
		lower, upper, view = get_dominant(hsv)

		# if view == 0: continue
		bboxes = detect(hsv, lower, upper)
		for bbox in bboxes:
			clf_team(bbox, image)
		
		count += 1
		# cv2.imshow('Match Detection',image)

		# cv2.imshow('Match Detection', cluster(image, 2))
		time.sleep(0.05)
		if cv2.waitKey(1) & 0xFF == ord('q'): break
	    
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()