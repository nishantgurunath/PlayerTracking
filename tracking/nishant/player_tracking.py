import cv2 as cv
import numpy as np
import skimage
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from tracker_kf import Tracker as tracker_kf
import glob
import time

class objectTracking():
    def LK(self,path,detections): # Lucas Kanade Tomasi
   
 
        ## Video Read 
        #cap = cv.VideoCapture(path)

        ## Sequence of Images
        image_list = glob.glob(path+"*.png")
        image_list.sort()
        images = []
        for image in image_list:
            image = cv.imread(image,-1)
            images.append(image)
        images = np.array(images)

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
        #ret, old_frame = cap.read()
        old_frame = images[0]
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        #p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        #print (p0.dtype)
        #p0 = np.concatenate((p0,p0),axis=1)
        #print (p0[0])
        p0 = detections[:,:2] + detections[:,2:]/2
        p0 = p0.astype("float32")
        #print (p0)
        #return None
        # Create a mask image for drawing purposes
        #mask = np.zeros_like(old_frame)
        #while(1):
        #    ret,frame = cap.read() 
        for j in range(1,len(images)):
            frame = images[j]
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            index = (st==1).reshape(-1)
            good_new = p1[index,:]
            bbox_dimensions = detections[index][:,2:]
            bbox_coordinates = good_new - bbox_dimensions/2
            #print (p0.shape,p1.shape,index.shape,good_new.shape,detections.shape,bbox_dimensions.shape,bbox_coordinates.shape)
            detections = np.concatenate((bbox_coordinates,bbox_dimensions),axis=1)
            good_old = p0[index]
            # draw the tracks
            #for i,(new,old) in enumerate(zip(good_new, good_old)):
                #a,b = new.ravel()
                #c,d = old.ravel()
                #mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                #frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
            #img = cv.add(frame,mask)
            for i,detection in enumerate(detections):
                x,y,w,h = detection.astype("int32")
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            img = frame

            cv.imshow('frame',img)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            #p0 = good_new.reshape(-1,1,2)  
            p0 = good_new 
            time.sleep(0.5) 

    def kalmanFilter(self,path,bb_init):
        # Create opencv video capture object
        cap = cv2.VideoCapture('data/TrackingBugs.mp4')

        # Create Object Tracker
        tracker = Tracker(160, 30, 5, 100)

        # Variables initialization
        skip_frame_count = 0
        track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (0, 255, 255), (255, 0, 255), (255, 127, 255),
                        (127, 0, 255), (127, 0, 127)]
        pause = False

        # Infinite loop to process video frames
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Make copy of original frame
            orig_frame = copy.copy(frame)

            # Skip initial frames that display logo
            if (skip_frame_count < 15):
                skip_frame_count += 1
                continue

            # Detect and return centeroids of the objects in the frame
            #centers = detector.Detect(frame)
            centers = bb_init

            # If centroids are detected then track them
            if (len(centers) > 0):

                # Track object using Kalman Filter
                tracker.Update(centers)

                # For identified object tracks draw tracking line
                # Use various colors to indicate different track_id
                for i in range(len(tracker.tracks)):
                    if (len(tracker.tracks[i].trace) > 1):
                        for j in range(len(tracker.tracks[i].trace)-1):
                            # Draw trace line
                            x1 = tracker.tracks[i].trace[j][0][0]
                            y1 = tracker.tracks[i].trace[j][1][0]
                            x2 = tracker.tracks[i].trace[j+1][0][0]
                            y2 = tracker.tracks[i].trace[j+1][1][0]
                            clr = tracker.tracks[i].track_id % 9
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                     track_colors[clr], 2)

                # Display the resulting tracking frame
                cv2.imshow('Tracking', frame)

            # Display the original frame
            cv2.imshow('Original', orig_frame)

            # Slower the FPS
            cv2.waitKey(50)

            # Check for key strokes
            k = cv2.waitKey(50) & 0xff
            if k == 27:  # 'esc' key has been pressed, exit program.
                break
            if k == 112:  # 'p' has been pressed. this will pause/resume the code.
                pause = not pause
                if (pause is True):
                    print("Code is paused. Press 'p' to resume..")
                    while (pause is True):
                        # stay in this loop until
                        key = cv2.waitKey(30) & 0xff
                        if key == 112:
                            pause = False
                            print("Resume code..!!")
                            break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()



    def others(self,path,bb_init):
        pass
    
        




if __name__ == "__main__":
    #path = "./0165_2013-11-07 21_05_17.577813000.h264"
    path = "../../../Data/dataset/NCAA/data/0En5pOUZN5M/clip_46/"
    track = objectTracking()
    bbs_data = pd.read_csv(path+"01_info.csv")
    detections = []
    for _, bbox in bbs_data.iterrows():
        detections.append([bbox.x,bbox.y,bbox.w,bbox.h])
    detections = np.array(detections)
    #print (detections)
    track.LK(path,detections)

    
