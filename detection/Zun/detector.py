# import libraries

import cv2
import os 
import numpy as np
import glob
import matplotlib.pyplot as plt

class Detection: 
    def __init__(self):
        self.data = []
        self.image_files = []
        self.iamge_folders = []
        self.data_directory = "./../dataset/NCAA/data/" 
        
        # Court color range
        court_color = np.uint8([[[160,221,248]]])
        court_hsv = cv2.cvtColor(court_color, cv2.COLOR_BGR2HSV)
        hue = court_hsv[0][0][0]

        self.court_lower_color = np.array([hue - 30,10,10])
        self.court_upper_color = np.array([hue+ 30,255,255])
        self.lower_blue = np.array([70,50,50])
        self.upper_blue = np.array([110,255,255])
        self.jersey_color = np.array([])
   
    def load_image_folders(self):
        self.image_folders = [os.path.join(self.data_directory,name) for name in os.listdir(self.data_directory)] 
        #image_files = glob.glob("./../dataset/NCAA/*.png")
        
    def detect_people(self):
      # Save the image for one
      # Reading the video 
      vidcap  = cv2.VideoCapture('cutvideo.mp4')
      success,image = vidcap.read()
      count = 0
      success = True
      idx = 0
      
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

      #green range 
      lower_green = np.array([40,40,40])
      upper_green = np.array([70,255,255])

      mask = cv2.inRange(hsv, lower_green, upper_green)
      res = cv2.bitwise_and(image, image, mask = mask)

      res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
      res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

      kernel = np.ones((13,13), np.uint8)
      thresh = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV |
          cv2.THRESH_OTSU)[1]
      thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) 

      #find contours in threshold image
      contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      areas = [cv2.contourArea(c) for c in contours]
      max_index = np.argmax(areas)
      max_cnt=contours[max_index]
      x,y,w,h = cv2.boundingRect(max_cnt)
      cv2.rectangle(thresh,(x,y),(x+w,y+h),(100,255,0),2)


      plt.imshow(res_gray), plt.show()


      #plt.imshow(mac_cnt), plt.show()
      #cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
      #plt.imshow(image), plt.show()
    def get_jersey_color(self):
      jersey_image = cv2.imread('france.jpeg')
def main():
    obj = Detection()
    obj.get_jersey_color()
    obj.load_image_folders()
    obj.detect_people()

if __name__== "__main__":
    main()

    
