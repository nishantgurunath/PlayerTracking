import pandas as pd
# import matplotlib.pyplot as plt
import cv2
from detection.aniruddh.detector import detect

data_dir = './dataset/NCAA/'
game_name = '0En5pOUZN5M'
clip_name = 'clip_46'
img_num = 1

for i in range(5):
	im_path = data_dir + game_name + '/' + clip_name + '/' + '{:02}.png'.format(i + 1)
	im = cv2.imread(im_path)
	rects = detect(im)

	for i, (x, y, w, h) in enumerate(rects):
		# if weights[i] < 0.7:
		#     continue
		cv2.rectangle(im, (x,y), (x+w,y+h),(0,255,0),2)

	cv2.imshow("preview", im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Ground truth boxes
# bb_path = data_dir + game_name + '/' + clip_name + '/' + '{:02}_info.csv'.format(img_num)
# bbs_data = pd.read_csv(bb_path)
# for _, bbox in bbs_data.iterrows():
# 	print (bbox)
# 	x0, y0, x1, y1 = int(bbox.x), int(bbox.y), int(bbox.x + bbox.w), int(bbox.y + bbox.h)
	
# 	cv2.rectangle(im, (x0,y0), (x1,y1), (255, 0, 0))

# cv2.imshow('image', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

