# import libraries

import cv2
import os 
import numpy as np
import glob
import matplotlib.pyplot as plt
import pdb
from sklearn.cluster import KMeans
import colorsys
import math

class Detection: 
  def __init__(self):
      self.data = []
      self.image_files = []
      self.iamge_folders = []
      # Court color range
      self.lower_blue = np.array([70,50,50])
      self.upper_blue = np.array([110,255,255])
  def rgb_to_hsv(self,r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v
      
  def detect_people(self):
    # Save the image for one
    # Reading the video 
    vidcap  = cv2.VideoCapture('cutvideo.mp4')
    success,image = vidcap.read()
    count = 0
    success = True
    idx = 0
    bounding_boxes = self.detect(image)
    for bbox in bounding_boxes:
      self.classify_team(bbox,image)
    
    cv2.imshow('Match Detection',image)

    cv2.waitKey(0)
  def get_jersey_color(self):
    self.team1_jersey = cv2.imread('france.jpeg')
    self.team2_jersey = cv2.imread('belgium.jpeg')
  
  def detect(self, image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower, upper, view = self.get_dominant(hsv)
    return self.get_bbox(hsv, lower, upper)
  def find_histogram(self,clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()
    return hist
  def plot_colors2(self,hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar
  def get_dominant(self,hsv):
    h, w, _ = hsv.shape
    hist, edges = np.histogram(hsv[...,0], bins= 12, range=(0,360))
    idx = np.argmax(hist)
    lower = np.array([edges[idx], 0, 0])
    upper = np.array([edges[idx + 1], 255 , 255])

    if hist[idx] / (h*w) > 0.6 : view = 1
    else: view = 0
    return lower, upper, view
  def get_dominant_jersey(self):
    img_tmp1 = cv2.cvtColor(self.team1_jersey, cv2.COLOR_BGR2RGB)
    img_tmp1 = img_tmp1.reshape((img_tmp1.shape[0]*img_tmp1.shape[1],3))
    clt1 = KMeans(n_clusters = 2)
    clt1.fit(img_tmp1)
    img_tmp2 = cv2.cvtColor(self.team2_jersey, cv2.COLOR_BGR2RGB)
    img_tmp2 = img_tmp2.reshape((img_tmp2.shape[0]*img_tmp2.shape[1],3))
    clt2 = KMeans(n_clusters = 2)
    clt2.fit(img_tmp2)
    #pdb.set_trace()
    self.team1_jersey_color = clt1.cluster_centers_[0]
    self.team2_jersey_color = clt2.cluster_centers_[1]
  
  def find_least(self,color_list,uniform_color):
    dist_min = float("inf")
    h,s,v = self.rgb_to_hsv(uniform_color[0],uniform_color[1],uniform_color[2])
    for color in color_list:
      h1,s1,v1 = self.rgb_to_hsv(color[0],color[1],color[2])
      dist = np.linalg.norm(h1-h)
      if(dist < dist_min):
        min_color = color
        dist_min = dist
      #pdb.set_trace()
    return dist_min
  
  def get_bbox(self,hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(hsv, hsv, mask=mask)
    res_gray = res[..., -1]
    # removing audience using morphology operation
    thresh = cv2.threshold(res_gray,100,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow('Threshold', thresh)
    kernel = np.ones((13,13),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('morphology', thresh)

    #find contours in threshold image  
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0,255,0), 2)
    # cv2.imshow('contours', image)

    bbox = []
    for c in contours:
      x,y,w,h = cv2.boundingRect(c)
      area = w * h
      if(100 < area < 15000 and h > 50):
        bbox.append([x,y,w,h])
    return np.array(bbox)
    
  def classify_team(self,bbox, image):
    #pdb.set_trace()
    font = cv2.FONT_HERSHEY_SIMPLEX
    x,y,w,h = bbox
    player_img = image[y:y+h, x:x+w]
    img_tmp = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
    img_tmp = img_tmp.reshape((img_tmp.shape[0]*img_tmp.shape[1],3))  
    
    clt = KMeans(n_clusters = 4, random_state = 0)
    clt.fit(img_tmp)
    
    team1_dist = self.find_least(clt.cluster_centers_,self.team1_jersey_color)
    team2_dist = self.find_least(clt.cluster_centers_,self.team2_jersey_color)
    #pdb.set_trace()
    
    if(w*h < 2000):
      if(team1_dist < team2_dist):
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
      else:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
    else:
      if(team1_dist < 120):
        cv2.rectangle(image,(x,y),(x+math.floor(w/2),y+h),(255,0,0),3)
        print(team1_dist)
      if(team2_dist < 50):
        cv2.rectangle(image,(x+math.floor(w/2),y),(x+w,y+h),(0,0,255),3)
       
def main():
    obj = Detection()
    obj.get_jersey_color()
    obj.get_dominant_jersey()
    obj.detect_people()

if __name__== "__main__":
    main()

    
