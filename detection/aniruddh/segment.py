import cv2
import os
import numpy as np
import time

datapath = 'C:/Users/aniru/OneDrive/Desktop/final/dataset/'
# Color ranges
lower_green = np.array([30,40, 40])
upper_green = np.array([70, 255, 255])

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

lower_red = np.array([0,31,255])
upper_red = np.array([176,255,255])

lower_white = np.array([0,0,0])
upper_white = np.array([0,0,255])

def detect(image):
	#converting into hsv image
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # applying mask and get grayscale
	mask = cv2.inRange(hsv, lower_green, upper_green)
	cv2.imshow('Mask', mask)
	res = cv2.bitwise_and(image, image, mask=mask)
	res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
	res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	# cv2.imshow('Gray Scale', res_gray)

    # removing audience using morphology operation
	thresh = cv2.threshold(res_gray,100,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	# thresh = cv2.adaptiveThreshold(res_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
 #            cv2.THRESH_BINARY,11,2)
	# cv2.imshow('Threshold', thresh)
	kernel = np.ones((13,13),np.uint8)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	# cv2.imshow('Morhology', mask)

    #find contours in threshold image  
	contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	bbox = []
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		if(h>=(1.5)*w):
			if(w>15 and h>= 15):
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
	if(nzCount >= 20):
		# cv2.putText(image, 'France', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

	#If player has red jersy
	mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
	res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
	res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
	res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
	nzCountred = cv2.countNonZero(res2)
	if(nzCountred>=20):
		# cv2.putText(image, 'Belgium', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
	


def main():
	#Read the video frame by frame
	for count in range(100):
		impath = datapath+'frames/%d.png' % (count)
		image = cv2.imread(impath)

		bboxes = detect(image)
		for bbox in bboxes:
			clf_team(bbox, image)
		
		count += 1
		# cv2.imshow('Match Detection',image)
		time.sleep(0.1)
		if cv2.waitKey(1) & 0xFF == ord('q'): break
	    
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()