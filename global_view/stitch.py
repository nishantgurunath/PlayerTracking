#Import libraries
import cv2
import os
import numpy as np
from panaroma import Panaroma


datapath = 'C:/Users/aniru/OneDrive/Desktop/final/dataset/'
panaroma = Panaroma()

def main():
	#Read the video frame by frame
	for count in range(1,800,4):
		impath = datapath+'pano/%d.png' % (count)
		image = cv2.imread(impath)
		prev_impath = datapath + 'pano/%d.png' % (count - 1)
		prev_image = cv2.imread(prev_impath)

		(result, matched_points) = panaroma.image_stitch([image, prev_image], match_status=True)
		cv2.imshow('stitch', result)

		if cv2.waitKey(1) & 0xFF == ord('q'): break
	    
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()