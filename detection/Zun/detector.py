# import libraries

import cv2
import os 
import numpy as np
import glob
print(glob.glob("/home/adam/*.txt"))

class Detection: 
    
    image_files
    iamge_folders

    # Court color range
    lower_brown = np.array([])
    upper_brown = np.array([])

    def __init__(self):
        self.data = []
   
    def load_image_folders(self):
        image_folders = [os.path.abspath(name) for name in
            os.listdir("./../dataset/NCAA") if
            os.path.isdir(name)]
        #image_files = glob.glob("./../dataset/NCAA/*.png")
        
    def detect_people(self):
      # converting into hsv image
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     

      # 
def main():
    obj = Detection()
    obj.load_image_folders()

if __name__== "__main__":
    main()

    
