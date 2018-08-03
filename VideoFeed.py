# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import cv2
import numpy as np
import thread
from threading import Lock
class VideoFeed:
    '''Reads video frames and passes them on'''
    def __init__(self, camera_port=0):
        self.cap = cv2.VideoCapture(camera_port)
        self.ret, self.img = self.cap.read()
        self.thread_handle = None
        self.stopped = True
        self.lock = Lock()

    def start(self):
        '''starts new thread for video feed capture'''
        self.stopped = False
        self.thread_handle = thread.start_new_thread(self.frame_update, ())

    def frame_update(self):
        '''Continuously captures and updates frame'''
        while not self.stopped:
            self.ret, self.img = self.cap.read()

    def read(self):
        ''' return image captured'''
        return self.img

    def stop(self):
        '''stops video feed thread'''
        self.stopped = True
