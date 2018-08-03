# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import thread
import cv2
import numpy as np
from tf_pose.estimator import BodyPart, Human
from threading import Lock

def dist(a, b):
    if a.shape != (2) or b.shape != (2):
        print "DIST"
        print a.shape
        print b.shape
        print a
        print b
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b [1])**2)

class OptFlowEst:
    def __init__(self, old_frame, humans, video_feed):
        self.old_frame = old_frame
        self.old_frame_grey = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
        self.humans = humans
        self.video_feed = video_feed
        self.frame_shape = old_frame.shape
        # The index of each body part not initially in frame when interpolation starts, for each person in frame
        #-self.missing_parts = [i for i in range(18) if i not in human.body_parts.keys() for human in humans]
        self.missing_parts = [i for human in humans for i in range(18) if i not in human.body_parts.keys()]
        self.missing_parts.sort()
        self.stopped = True
        self.thread_handle = None
        self.lock = Lock()

    def start(self):
        self.stopped = False
        self.thread_handle = thread.start_new_thread(self.estimate_loop, ())

    def repack_humans(self, p1, human_idx):
        body_part_keys = self.humans[human_idx].body_parts.keys()

        #print "#%d: p1 shape %s body_parts %s" % (human_idx, str(p1.shape), str(self.humans[human_idx].body_parts.keys()))
        #print "p1 shape %s" % str(p1.shape)
        #print "body_parts %s" % str(self.humans[human_idx].body_parts.keys())
        for idx in range(len(p1)):
            self.humans[human_idx].body_parts[body_part_keys[idx]].x = p1[idx][0][0] / self.frame_shape[1]
            self.humans[human_idx].body_parts[body_part_keys[idx]].y = p1[idx][0][1] / self.frame_shape[0]

    def estimate_loop(self):
        opt_flow_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        while not self.stopped:
            frame = self.video_feed.read()
            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Pull data from each human and bodypart -> put into np array w/shape (num_humans, 18, 2) and reshape to (num_humans 18, 1, 2) for use by optical flow
            with self.lock:
                all_human_points = np.asarray([np.asarray([[[body_part.x * self.frame_shape[1], body_part.y * self.frame_shape[0]]] for key, body_part in human.body_parts.iteritems()], dtype=np.float32) for human in self.humans])
                for idx, human_points in enumerate(all_human_points):
                        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame_grey, frame_grey, human_points, None, **opt_flow_params)
                        self.repack_humans(p1, idx)

                        # Grab the points that have gone out of frame
                        oof_points = p1[st!=1]
                        if oof_points.shape != 0:
                            # Get all the matches
                            tmp = np.isin(human_points, oof_points)
                            # Get the indexes of those matches
                            msng_idxz = [msng for msng in range(len(human_points)) if tmp[msng].all()]
                            #print "msng_idxz %s" % str(msng_idxz)
                            cur_part_exist = self.humans[idx].body_parts.keys()
                            for foo_idx in range(len(msng_idxz)):
                                del self.humans[idx].body_parts[cur_part_exist[msng_idxz[foo_idx]]]
                        if len(self.humans[idx].body_parts.keys()) == 0:
                            del self.humans[idx]

            self.old_frame = frame
            self.old_frame_grey = frame_grey.copy()

    def estimate(self):
        with self.lock:
            return self.humans

    def stop(self):
        self.stopped = True

    def refresh(self, old_frame, humans):
        with self.lock:
            self.humans = humans
            self.old_frame = old_frame
            self.old_frame_grey = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
