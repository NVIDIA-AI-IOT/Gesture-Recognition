# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import argparse
import time
import operator
import collections
import cv2
import numpy as np
import math
import glob
import os
import sys
from Var import Var

from imutils.video import WebcamVideoStream
from tqdm import tqdm
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import tf_pose.pafprocess as pafprocess
import tensorflow as tf

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64MultiArray, Float64, MultiArrayLayout

from VideoFeed import VideoFeed
from OptFlowEst import OptFlowEst
from TFPoseEstThread import PoseEstimator
from PosePub import PosePub
from DisplayThread import DisplayThread


def get_data(body_part):
	return (body_part.x , body_part.y, body_part.score)



if __name__ == '__main__':
	''' Initialize argparse flags '''
	parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
	parser.add_argument('--camera', type=int, default=0)
	parser.add_argument('--frames_to_append', '-f', type=int, default=4)

	parser.add_argument('--resize', type=str, default='368x368',
						help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
	parser.add_argument('--resize-out-ratio', type=float, default=4.0,
						help='if provided, resize heatmaps before they are post-processed. default=1.0')
	parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
	parser.add_argument('--show-process', type=bool, default=False,
						help='for debug purpose, if enabled, speed for inference is dropped.')
	parser.add_argument('--debug', dest='debug', action='store_true')
	parser.add_argument('--bad_data', dest='bad_data',action='store_true')
	parser.add_argument('--use_angles', "-a", dest="use_angles", action='store_true')
	parser.add_argument('--only-arm', '-o', dest="use_arm", action='store_true')
	parser.set_defaults(use_angles=False)
	parser.set_defaults(use_arm=False)
	parser.set_defaults(debug=False)
	parser.set_defaults(bad_data=False)
	args = parser.parse_args()
	w, h = model_wh(args.resize)
	debug = args.debug
	bad_data_flag = args.bad_data
	use_angles= args.use_angles
	use_arm = args.use_arm
	s_tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
	s_tf_config.gpu_options.allow_growth = True

	cap = VideoFeed(args.camera)
	cap.start()
	tf_pose_est = PoseEstimator(model="mobilenet_thin", target_size='368x368', resize_out_ratio=4.0, video_feed=cap, tf_config=s_tf_config)
	tf_pose_est.start()
	v = Var(use_arm)
	NUM_JOINTS = v.get_size()
	FPS = v.get_rate()
	print("Video and Pose started")

	data_out = PosePub(FPS, args.frames_to_append, use_angles, use_arm, bad_data_flag, debug)

	humans = []
	# wait until there are humans in frame
	while True:
		while True:
			if tf_pose_est.fresh_data:
				break
			continue
		humans = tf_pose_est.estimate()
		if len(humans) != 0:
			break

	opt_pose_est = OptFlowEst(old_frame=tf_pose_est.old_frame, humans=humans, video_feed=cap)
	frame = tf_pose_est.old_frame
	opt_pose_est.start()
	print("Estimator started")

	print("#" * 10 + "SPINNNING" + "#" * 10)
	run_opt_est = False
	while data_out.is_ok():
		fps_time = time.time()

		if tf_pose_est.fresh_data:
			humans = tf_pose_est.estimate()
			# No point in running if there are not people detected by the NN version
			if len(humans) == 0:
				sys.stdout.write("\rFPS: %.3f, NumHumans: %d" %  (0.000, len(humans)))
				sys.stdout.flush()
				while len(humans) == 0:
					# opt_pose_est.stop()
					if tf_pose_est.fresh_data:
						humans = tf_pose_est.estimate()
					else:
						data_out.pub_null(len(humans))

			# opt_pose_est.start()
			opt_pose_est.refresh(tf_pose_est.old_frame, humans)
			frame = tf_pose_est.old_frame
		else:
			humans = opt_pose_est.estimate()
			frame = opt_pose_est.old_frame
		data_out.post_n_pub(humans)
		sys.stdout.write("\rFPS: %.2f, NumHumans: %d" % ((1.0/ (time.time() - fps_time), len(humans))))
		sys.stdout.flush()

	cv2.destroyAllWindows()