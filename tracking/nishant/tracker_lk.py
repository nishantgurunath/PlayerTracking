'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
from common import dprint
from scipy.optimize import linear_sum_assignment
import cv2

class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = [np.asarray(prediction)]  # trace path


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.dimensions = []
        self.trackIdCount = trackIdCount
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def Update(self, detections,old_gray,frame_gray):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i][0:2], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
                self.dimensions.append(detections[i][2:])

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                diff = self.tracks[i].prediction - detections[j][0:2]
                distance = np.sqrt(diff[0]*diff[0] +
                                       diff[1]*diff[1])
                cost[i][j] = distance
        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                elif((detections[assignment[i]][2]*detections[assignment[i]][3])/(self.dimensions[i][0]*self.dimensions[i][1]) <= 2):
                    self.tracks[i].trace[-1] = detections[assignment[i]][0:2]
                    self.dimensions[i] = detections[assignment[i]][2:]
                    #pass
                else:
                    #self.tracks[i].trace[-1] = detections[assignment[i]][0:2] + np.random.randint(-1,1,(2,))
                    self.tracks[i].trace[-1] = detections[assignment[i]][0:2]
                    self.dimensions[i] = detections[assignment[i]][2:]
                    #pass
            else:
                ## LK
                self.tracks[i].skipped_frames += 1


        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del self.dimensions[id]
                    del assignment[id]
                else:
                    dprint("ERROR: id is greater than length of tracks")

        if(len(del_tracks)):
            print ("del1")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]][0:2],
                              self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
                self.dimensions.append(detections[un_assigned_detects[i]][2:])

        # Update LK state and tracks trace
        p0 = []
        for i in range(len(self.tracks)):
            p0.append(self.tracks[i].trace[-1])
        p0 = np.array(p0).astype("float32")
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
        p1 = list(p1)
        #del_tracks = []
        #for i in range(len(self.tracks)):
        #    if (st[i] != 1):
        #        del_tracks.append(i)

        #if len(del_tracks) > 0:  # only when skipped frame exceeds max
        #    for id in del_tracks:
        #        if id < len(self.tracks):
        #            del self.tracks[id]
        #            del self.dimensions[id]
        #            del assignment[id]
        #            del p1[id]
        #        else:
        #            dprint("ERROR: id is greater than length of tracks")

        #if(len(del_tracks)):
        #    print ("del2",len(del_tracks),len(self.tracks))
        #print(len(self.tracks))

        for i in range(len(self.tracks)):
            self.tracks[i].prediction = p1[i]
            self.tracks[i].trace.append(self.tracks[i].prediction)
