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
import copy
import re
from imutils.video import FPS


class objectTracking():
    def LK(self,path,detections): # Lucas Kanade Tomasi, detections - just the first frame
   
 
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

        # Video
        ret, old_frame = cap.read()

        # Images
        #old_frame = images[0]

        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = detections[:,:2] + detections[:,2:]/2
        p0 = p0.astype("float32")

        # Video
        while(1):
            try:
                ret,frame = cap.read()
                print (frame.shape) 

        # Images
        #for j in range(1,len(images)):
            #frame = images[j]

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
                p0 = good_new 
                time.sleep(0.1)
            except:
                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break

        cv.destroyAllWindows()
        cap.release()

    def kalmanFilter(self,path,detections): #detections for each frame
        # Create opencv video capture object
        cap = cv.VideoCapture(path)

        # Create Object Tracker
        tracker = tracker_kf(160, 30, 5, 100)

        # Variables initialization
        skip_frame_count = 0
        track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (0, 255, 255), (255, 0, 255), (255, 127, 255),
                        (127, 0, 255), (127, 0, 127)]
        pause = False

        # Infinite loop to process video frames
        j = 0
        while(True):
            try:
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Make copy of original frame
                #orig_frame = copy.copy(frame)

                # Skip initial frames that display logo
                #if (skip_frame_count < 15):
                #    skip_frame_count += 1
                #    continue

                # Detect and return centeroids of the objects in the frame
                #centers = detector.Detect(frame)
                centers = detections[j]
                # If centroids are detected then track them
                if (len(centers) > 0):

                    # Track object using Kalman Filter
                    tracker.Update(centers)

                    # For identified object tracks draw tracking line
                    # Use various colors to indicate different track_id
                    #for i in range(len(tracker.tracks)):
                    #    if (len(tracker.tracks[i].trace) > 1):
                    #        for j in range(len(tracker.tracks[i].trace)-1):
                    #            # Draw trace line
                    #            x1 = tracker.tracks[i].trace[j][0][0]
                    #            y1 = tracker.tracks[i].trace[j][1][0]
                    #            x2 = tracker.tracks[i].trace[j+1][0][0]
                    #            y2 = tracker.tracks[i].trace[j+1][1][0]
                    #            clr = tracker.tracks[i].track_id % 9
                    #            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                    #                     track_colors[clr], 2)

                    for track in tracker.tracks:
                        x,y,w,h = track.prediction.astype("int32")
                        frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                    # Display the resulting tracking frame
                    cv.imshow('Tracking', frame)

                # Display the original frame
                #cv.imshow('Original', orig_frame)

                # Slower the FPS
                #cv.waitKey(50)

                # Check for key strokes
                k = cv.waitKey(50) & 0xff
                if k == 27:  # 'esc' key has been pressed, exit program.
                    break
                #if k == 112:  # 'p' has been pressed. this will pause/resume the code.
                #    pause = not pause
                #    if (pause is True):
                #        print("Code is paused. Press 'p' to resume..")
                #        while (pause is True):
                #            # stay in this loop until
                #            key = cv.waitKey(30) & 0xff
                #            if key == 112:
                #                pause = False
                #                print("Resume code..!!")
                #                break
                j += 1
                time.sleep(0.5)
            except:
                k = cv.waitKey(50) & 0xff
                if k == 27:  # 'esc' key has been pressed, exit program.
                    break

        # When everything done, release the capture
        cv.destroyAllWindows()
        cap.release()



    def others(self,path,detections):

        # extract the OpenCV version info
        (major, minor) = cv.__version__.split(".")[:2]
         
        # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
        # function to create our object tracker
        if int(major) == 3 and int(minor) < 3:
            tracker = cv.Tracker_create(args["tracker"].upper())
         
        # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
        # approrpiate object tracker constructor:
        else:
            # initialize a dictionary that maps strings to their corresponding
            # OpenCV object tracker implementations
            OPENCV_OBJECT_TRACKERS = {
                    "csrt": cv.TrackerCSRT_create,
                    "kcf": cv.TrackerKCF_create,
                    "boosting": cv.TrackerBoosting_create,
                    "mil": cv.TrackerMIL_create,
                    "tld": cv.TrackerTLD_create,
                    "medianflow": cv.TrackerMedianFlow_create,
                    "mosse": cv.TrackerMOSSE_create
            }
         
            # grab the appropriate object tracker using our dictionary of
            # OpenCV object tracker objects

        # initialize the bounding box coordinates of the object we are going
        # to track
        vs = cv.VideoCapture(path)
         
        # initialize the FPS throughput estimator
        fps = None
        fps = FPS().start()
        frame = vs.read()[1]

        trackers = []
        for initBB in detections[0]:
            initBB = tuple(initBB.astype('int32'))
            tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
            trackers.append(tracker)
            trackers[-1].init(frame,initBB) 
            

        # loop over frames from the video stream
        while True:
            # check to see if we have reached the end of the stream
            if frame is None:
                break
         
            # resize the frame (so we can process it faster) and grab the
            # frame dimensions
            #frame = imutils.resize(frame, width=500)
            (H, W) = frame.shape[:2]


            # check to see if we are currently tracking an object
            for tracker in trackers:
                # grab the new bounding box coordinates of the object
                (success, box) = tracker.update(frame)
 
                # check to see if the tracking was a success
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
 
                # update the FPS counter
            fps.update()
            fps.stop()
 
            # show the output frame
            cv.imshow("Frame", frame)
            key = cv.waitKey(1) & 0xFF
            #time.sleep(1)

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                    break
            frame = vs.read()[1]
 
        vs.release()
        cv.destroyAllWindows()    
        

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


if __name__ == "__main__":
    #path = "./0165_2013-11-07 21_05_17.577813000.h264"
    path = "../../../Data/dataset/NCAA/data/0En5pOUZN5M/clip_46/"
    track = objectTracking()
    track.images2video(path)

    bbs_data_list = glob.glob(path+"*_info.csv")
    bbs_data_list.sort()
    #bbs_data = pd.read_csv(path+"01_info.csv")
    #detections = []
    #for _, bbox in bbs_data.iterrows():
    #    detections.append([bbox.x,bbox.y,bbox.w,bbox.h])
    #detections = np.array(detections)


    detections = []
    for f in bbs_data_list:
        if(re.search("\d+_info.csv",f)):
            bbs_data = pd.read_csv(f)
            detection = []
            for _, bbox in bbs_data.iterrows():
                detection.append([bbox.x,bbox.y,bbox.w,bbox.h])
            detection = np.array(detection)
            detections.append(detection)

   
    #print (detections)
    #track.LK(path+"project.avi",detections[0])
    #track.kalmanFilter(path+"project.avi",detections)
    track.others(path+"project.avi",detections)

    
