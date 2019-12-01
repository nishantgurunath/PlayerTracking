import cv2 as cv
import numpy as np
import skimage
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from tracker_lk import Tracker as tracker_lk
import glob
import time
import copy
import re
from imutils.video import FPS
sys.path.append('../../detection/aniruddh')
from detector import detect



def box_suppression(centers):
    valid = np.ones(len(centers))
    for i in range(len(centers)):
        if((1.5*centers[i][2] > centers[i][3]) or (centers[i][2]*centers[i][3] < 2000) or (centers[i][2]*centers[i][3] > 8e10)):
            valid[i] = 0
    return centers[valid==1]
                
class objectTracking(object):

    def LK(self,path): # Lucas Kanade Tomasi, detections - just the first frame
   
        print (cv.TERM_CRITERIA_EPS, cv.TERM_CRITERIA_COUNT)
 
        ## Video Read 
        cap = cv.VideoCapture(path)

        ## Sequence of Images
        #image_list = glob.glob(path+"*.png")
        #image_list.sort()
        #images = []
        #for image in image_list:
        #    image = cv.imread(image,-1)
        #    images.append(image)
        #images = np.array(images)

        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0,255,(100,3))
        # Take first frame and find corners in it

        #tracker
        tracker = tracker_lk(30, 1000, 30000000, 100)

        # Video
        ret, old_frame = cap.read()
        # Images
        #old_frame = images[0]

        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        detections = detect(old_frame)
        #detections = box_suppression(detections) 

        

        # Video
        while(1):
            #try:
                ret,frame = cap.read()
                if frame is None:
                    break
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                tracker.Update(detections,old_gray,frame_gray)
                #print (frame.shape) 

                detections = detect(frame)
                #detections = box_suppression(detections) 
                detections[:,0:2] = detections[:,0:2] + detections[:,2:]/2

        # Images
        #for j in range(1,len(images)):
            #frame = images[j]
                #print("Frame")
                for i in range(len(tracker.tracks)):
                    x,y = (tracker.tracks[i].trace[-2]).astype("int32")
                    w,h = (tracker.dimensions[i]).astype("int32")
                    if(w < h and w*h > 2500):
                    #if(w*h > 2000):
                        old_frame = cv.rectangle(old_frame,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0),2)
                    #print (w,h,w*h)
                    
                cv.imshow('frame',old_frame)
                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break
                # Now update the previous frame and previous points
                old_frame = frame.copy()
                old_gray = frame_gray.copy()
                #time.sleep(0.1)
            #except:
            #    k = cv.waitKey(30) & 0xff
            #    if k == 27:
            #        break

        cv.destroyAllWindows()
        #cap.release()


    def images2video(self,path):
        ## Sequence of Images
        image_list = glob.glob(path+"*.png")
        image_list.sort()
        images = []
        for image in image_list:
            image = cv.imread(image,-1)
            images.append(image)
        images = np.array(images)
        out = cv.VideoWriter(path+'project.avi',cv.VideoWriter_fourcc(*'DIVX'), 50, (image.shape[1],image.shape[0]))

        for image in images:
            out.write(image)
        out.release()

    def video2images(self,path):
        ## Sequence of Images
        cap = cv.VideoCapture(path)
        i = 0
        while True:
            ret,frame = cap.read()
            if frame is None:
                break
            cv.imwrite('./images/soccer'+str(i)+'.png',frame)
            i += 1

        cap.release()


if __name__ == "__main__":
    #path = "../../../0165_2013-11-07 21_05_17.577813000.h264"
    path = "./soccer.mp4"
    track = objectTracking()
    #path = "../../../Data/dataset/NCAA/data/0En5pOUZN5M/clip_46/"
    #track.images2video(path)

    # All Boxes
    #bbs_data_list = glob.glob(path+"*_info.csv")
    #bbs_data_list.sort()

    #detections = []
    #for f in bbs_data_list:
    #    if(re.search("\d+_info.csv",f)):
    #        bbs_data = pd.read_csv(f)
    #        detection = []
    #        for _, bbox in bbs_data.iterrows():
    #            detection.append([bbox.x,bbox.y,bbox.w,bbox.h])
    #        detection = np.array(detection)
    #        detections.append(detection)


    # LK First Box
    #bbs_data = pd.read_csv(path+"01_info.csv")
    #detections = []
    #for _, bbox in bbs_data.iterrows():
    #    detections.append([bbox.x,bbox.y,bbox.w,bbox.h])
    #detections = np.array(detections)



   
    #print (detections)
    track.LK(path)
    #track.kalmanFilter(path,detections=None)
    #track.others(path+"project.avi",detections)
    #track.video2images(path)

    
