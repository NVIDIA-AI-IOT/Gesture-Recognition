# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import collections
import thread

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import tf_pose.pafprocess as pafprocess
import tensorflow as tf
import VideoFeed
import Queue
from DisplayThread import DisplayThread

class PoseEstimator:
    '''Pose Estimation in a separate thread'''
    def __init__(self, model, target_size, resize_out_ratio, video_feed, tf_config=None):
        self.pose_est = TfPoseEstimator(get_graph_path(model), target_size=model_wh(target_size), tf_config=tf_config)
        self.resize_to_default = target_size[0] > 0 and target_size[1] > 0
        self.resize_out_ratio = resize_out_ratio
        self.video_feed = video_feed
        self.old_frame = None
        self.humans = None
        self.fresh_data = False
        self.thread_handle = None
        self.stopped = True
        self.display = DisplayThread("Display")

    def start(self):
        '''start thread for Pose Estimation'''
        self.stopped = False
        self.thread_handle = thread.start_new_thread(self.estimate_loop, ())

    def stop(self):
        '''stops Pose Estimation thread'''
        self.isStopped = True

    def estimate_loop(self):
        '''continously does pose estimation on frames from video feed and sends it to display thread for displaying'''
        display_queue = Queue.Queue()
        self.display.start(display_queue)
        while True:
            if self.stopped:
                print "TFPose Exiting"
                break
            frame = self.video_feed.read()
            self.humans = self.pose_est.inference(frame, resize_to_default=self.resize_to_default, upsample_size=self.resize_out_ratio)
            image = self.pose_est.draw_humans(frame, self.humans, imgcopy=False)
            display_queue.put(image)
            self.old_frame = frame
            self.fresh_data = True
            display_queue.join()


    def estimate(self):
        self.fresh_data = False
        return self.humans
