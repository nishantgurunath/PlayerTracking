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
   
    def load_image_folders(self):
        self.image_folders = [os.path.join(self.data_directory,name) for name in os.listdir(self.data_directory)] 
        #image_files = glob.glob("./../dataset/NCAA/*.png")
        
    def detect_people(self):
      # Save the image for one
      image_files = [f for f in glob.glob(self.image_folders[0] + "/**/*.png",
        recursive=True)]
      csv_files =  [f for f in glob.glob(self.image_folders[0] + "/**/*.csv",
        recursive=True)]
      image_files.sort()
      csv_files.sort()

      # converting into hsv image
      image = cv2.imread(image_files[0])
      csv = csv_files[0]
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      
      # Mask with additional color
      mask2 = cv2.inRange(hsv,self.lower_blue,self.upper_blue)
      res2 = cv2.bitwise_and(image,image, mask=mask2)
      # Mask with court color
      mask1 = cv2.inRange(hsv,self.court_lower_color,self.court_upper_color)
      res1 = cv2.bitwise_and(image,image, mask=mask1)

      res = res1 + res2
      # Convert hsv to gray
      res_bgr = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
      res_gray = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
      
      # Defining a kernel to do morp
      kernel = np.zeros((5,5),np.uint8)
      thresh = cv2.threshold(res_gray,85,205,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
      thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
      thresh2 = cv2.cornerHarris(res_gray,3,3,0.1)
      contours, hierarchy = cv2.findContours(res_gray, cv2.RETR_TREE,
          cv2.CHAIN_APPROX_SIMPLE)

      plt.imshow(res_gray), plt.show()
      plt.imshow(thresh), plt.show()
      plt.imshow(thresh1), plt.show()
      plt.imshow(thresh2), plt.show()
      cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
      plt.imshow(image), plt.show()
def main():
    obj = Detection()
    obj.load_image_folders()
    obj.detect_people()

if __name__== "__main__":
    main()

    
