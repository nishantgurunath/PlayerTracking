import cv2
import numpy as np
import sys
sys.path.append("/Users/nishantgurunath/Documents/CMU/18-797/Project/PlayerTracking/tracking/nishant/Python-Multiple-Image-Stitching/code")
from pano import Stitch
import glob
import re
sys.path.append("./Panoramic-Image-Stitching-using-invariant-features")
from panorama import Panaroma
from matchers import matchers
import imutils

def sortkey(path):
    L = path.split('soccer')
    L = L[-1].split('.png')
    return int(L[0])


def stitch1():
    #f = open("pano_images.txt","w")
    #images = glob.glob("./images_raw/*.png")
    #images.sort(key=sortkey)
    #f.write(images[0]+"\n")
    #for i,image in enumerate(images):
    #    if(i>=100 and i%10 == 0):
    #        f.write(image+"\n")
    #    #if(i==200):
    #    #    break
    #f.close() 
    s = Stitch("./pano_images.txt")
    #s = Stitch("./txtlists/files2.txt")
    s.leftshift()
    s.rightshift()
    print ("done")
    cv2.imwrite("test12.jpg", cv2.resize(s.leftImage,(1920,1080)))
    print ("image written")
    cv2.destroyAllWindows()


def stitch2():
    
    cap = cv2.VideoCapture('./soccer.mp4')
    panaroma = Panaroma()

    ret, result = cap.read()
    result = cv2.resize(result,(result.shape[1]//20,result.shape[0]//20))
    i = 0
    while(True):
        ret,frame = cap.read()
        if frame is None:
            break

        if(i>=100 and i%10==0):
            frame = cv2.resize(frame,(frame.shape[1]//10,frame.shape[0]//10))
            (result, matched_points) = panaroma.image_stitch([result, frame], match_status=True)
        if(i==200):
            break
        i += 1

    cv2.imwrite("test2.png",cv.resize(result,(1920,1080)))

def video2images(path):
    ## Sequence of Images
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ret,frame = cap.read()
        if frame is None:
            break
        cv2.imwrite('./images_game1/soccer_'+str(i)+'.png',frame)
        i += 1

    cap.release()

if __name__ == "__main__":
   # #f = open("pano_images.txt","w")
   # #images = glob.glob("./images_raw/*.png")
   # #images.sort(key=sortkey)
   # #f.write(images[0]+"\n")
   # #for i,image in enumerate(images):
   # #    if(i>=100 and i%10 == 0):
   # #        f.write(image+"\n")
   # #    #if(i==200):
   # #    #    break
   # #f.close() 
   # s = Stitch("./pano_images.txt")
   # #s = Stitch("./txtlists/files2.txt")
   # s.leftshift()
   # s.rightshift()
   # print ("done")
   # cv2.imwrite("test12.jpg", s.leftImage)
   # print ("image written")
   # cv2.destroyAllWindows()
 

   #stitch1()
   #stitch2()
   #i1 = cv2.imread('./images_game1/soccer_100.png')

   #kernel = np.ones((5,5),np.float32)/25
   #i1 = cv2.filter2D(i1,-1,kernel)
   #cv2.imshow('frame',i1)

   #hsv = cv2.cvtColor(i1,cv2.COLOR_BGR2HSV)
   #s = 80
   #lower_white = np.array([0,0,255-s], dtype=np.uint8)
   #upper_white = np.array([255,s,255], dtype=np.uint8)
   #mask = cv2.inRange(hsv, lower_white, upper_white)
   #i1 = cv2.bitwise_and(i1,i1, mask= mask)

   #i1 = cv2.Canny(i1,150,200)
   #i1 = cv2.dilate(i1,(5,5))
   ##i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
   #cnts = cv2.findContours(i1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   #cnts = imutils.grab_contours(cnts)
   #mask = np.ones(i1.shape[:2], dtype="uint8") * 255
   #for c in cnts:
   #    peri = cv2.arcLength(c, True)
   #    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
   #    if(peri < 1000):
   #        cv2.drawContours(mask, [c], -1, 0, -1)
   #i1 = cv2.bitwise_and(i1, i1, mask=mask)
   #corners = cv2.cornerHarris(i1,10,3,0.04)*255.0
   #print (np.max(corners),np.min(corners))
   #kp = np.nonzero(i1>250)
   #kp = np.concatenate((kp[0].reshape(-1,1),kp[1].reshape(-1,1)),axis=1)
   #kp = kp[np.logical_not(np.logical_and(kp[:,0]>600, kp[:,1]<128))]
   #kp = kp[np.logical_not(np.logical_and(np.logical_and(kp[:,0]>=65, kp[:,0]<120),np.logical_and(kp[:,1]>=68, kp[:,1]<278)))]
   #kp = kp[np.logical_not(np.logical_and(np.logical_and(kp[:,0]>=645, kp[:,0]<685),np.logical_and(kp[:,1]>=60, kp[:,1]<270)))]
   #kp = kp[np.logical_not(np.logical_and(np.logical_and(kp[:,0]>=576, kp[:,0]<686),np.logical_and(kp[:,1]>=550, kp[:,1]<733)))]
   #kp = kp[np.logical_not(np.logical_and(np.logical_and(kp[:,0]>=16, kp[:,0]<62),np.logical_and(kp[:,1]>=1061, kp[:,1]<1243)))]
   #kp = kp[np.logical_not(np.logical_and(np.logical_and(kp[:,0]>=640, kp[:,0]<688),np.logical_and(kp[:,1]>=1010, kp[:,1]<1223)))]
   #kp = kp[np.logical_not(np.logical_and(np.logical_and(kp[:,0]>=573, kp[:,0]<690),np.logical_and(kp[:,1]>=526, kp[:,1]<750)))]
   #kp = kp[np.logical_not(kp[:,0]>=573)]
   #index = np.arange(0,len(kp))
   #np.random.shuffle(index)
   #kp = kp[index[0:5000]]
   #kp = [cv2.KeyPoint(pt[1],pt[0],1) for pt in kp]
   #print (len(kp))
   #cv2.imshow('frame1',i1)
   #cv2.imshow('frame2',corners)

   #gray = (cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)/255>0.5)*255.0
   #gray = (np.sum((i1/255 > np.ones((i1.shape[0],i1.shape[1],3))*0.85),axis=2) == 3)*1.0
   #i1 = (i1/255 > np.ones((i1.shape[0],i1.shape[1],3))*0.85)*255.0
   #gray = cv2.erode(gray,(1,1))
   #i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
   #lsd = cv2.createLineSegmentDetector(0);
   #lines = lsd.detect(i1)[0];
   #edges = lsd.drawSegments(i1,lines)
   #orb = cv2.ORB_create()
   #kp, des = orb.detectAndCompute(i1, None)
   #edges = cv2.drawKeypoints(i1,kp,None,color=(0,255,0), flags=0)
   #cv2.imshow('frame',edges)
   #gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
   #cv2.imshow('dst1',res)
   #dst = cv2.dilate(dst,None)
   #i1[dst>0.01*dst.max()]=[0,0,255]
   #gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
   #print(i1.shape)
   #ret,i2 = cap.read()
   #matcher = matchers()
   #matcher.match_lsd(i1,i2)

   #sift = cv2.xfeatures2d.SIFT_create()
   #brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
   #kp,des = sift.detectAndCompute(i1,None)
   #kp,des = brief.compute(i1,kp)
   #edges = cv2.drawKeypoints(i1,kp,None,color=(0,255,0), flags=0)
   #cv2.imshow('dst',edges)
   #

   panaroma = Panaroma()
   #filenums = [100,98,96,94,92,90,88,86,84,82,80]
   #filenums = [100,98,95]
   filenums = np.arange(100,60,-2)
   images = []
   for i in filenums:
       name = "./images_game1/soccer_{}.png".format(i)
       images.append(cv2.imread(name))
   print (images[0].shape) 
   print (len(images))
   result = images[0]
   H = np.eye(3)
   for i in range(1,len(images)):
       (_, matched_points, homography) = panaroma.image_stitch([images[i-1], images[i]], match_status=True,lowe_ratio=0.8)
       H = H.dot(homography)
       temp = cv2.warpPerspective(images[i], H, (result.shape[1]+images[i].shape[1] , images[i].shape[0]+100))
       temp[0:result.shape[0],0:result.shape[1]] = result.copy()
       result = temp.copy()
       cv2.imshow('frame',result)
       k = cv2.waitKey(2) & 0xff
       if k == 27:
           break
       print (i)
   cv2.imwrite('result.png',result)
   cv2.imwrite('match.png',matched_points)
   k = cv2.waitKey(0) & 0xff
   if k == 27:
       cv2.destroyAllWindows()


   #path = "/Users/nishantgurunath/Desktop/out.mp4"
   #video2images(path)
