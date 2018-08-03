# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import cv2
import numpy as np
import thread
from threading import Lock
import time
class DisplayThread:
    def __init__(self, window_name):
        self.window_name = window_name
        self.lock = Lock()
        self.stopped = True
        self.thread_handle = None
        self.delay =  1000/v.get_rate()
        self.fps_time = time.time()

    def start(self, queue):
        self.stopped = False
        self.thread_handle = thread.start_new_thread(self.display_loop, (queue,))
        print("\nSTARTING DISPLAY")

    def stop(self):
        self.stopped = True

    def display_loop(self, display_queue):
        # global display_queue
        cv2.namedWindow(self.window_name)
        while not self.stopped:
            fps = 1.0 / (time.time() - self.fps_time)
            self.fps_time = time.time()
            with self.lock:
                img = display_queue.get()
                cv2.putText(img,
					"FPS: %f" % fps,
					(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
                cv2.imshow(self.window_name, img)
                display_queue.task_done()
                cv2.waitKey(self.delay)
                if cv2.waitKey(1) == 27:
                    self.stop()
