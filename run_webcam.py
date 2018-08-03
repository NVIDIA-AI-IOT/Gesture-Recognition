# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import os
import sys
import cv2
import math
import glob
import time
import signal
import logging
import argparse
import operator
import collections
import numpy as np
import tensorflow as tf
import tf_pose.pafprocess as pafprocess

from Var import Var
from VideoFeed import VideoFeed
from OptFlowEst import OptFlowEst
from DisplayThread import DisplayThread
from TFPoseEstThread import PoseEstimator
from imutils.video import WebcamVideoStream
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

''' Initialize variables for functions '''
fps_time = 0
arms_crossed = False
y_arms = False
right_dab = False
left_dab = False

''' Make functions '''
def get_data(body_part):
	return (body_part.x , body_part.y, body_part.score)

def x_arms_recognition():
	wrist_dist_x = xs[idx][4] - xs[idx][7]
	if (right_hand_up and left_hand_up) and (wrist_dist_x > 0) and (xs[idx][4] > xs[idx][7]):
		arms_crossed = True
		print("Arms crossed")
	else:
		arms_crossed = False
		print("Arms not crossed")
	return arms_crossed

def y_arms_recognition():
	wrist_dist_x = xs[idx][4] - xs[idx][7]
	elbows_higher_than_shoulders = (ys[idx][6] < ys[idx][5]) and (ys[idx][3] < ys [idx][2])
	if (right_hand_up and left_hand_up) and (not arms_crossed) and ((wrist_dist_x < 0)) and (xs[idx][4] < xs[idx][3] < xs[idx][2]) and (xs[idx][5] < xs[idx][6] < xs[idx][7]) and elbows_higher_than_shoulders:
		y_arms = True
		print("Y arms")
	else:
		y_arms = False
		print("Not y arms")
	return y_arms

def dab_recognition():
	r_dab = ys[idx][2] > ys[idx][3] > ys[idx][4]
	r_elbow_fold = xs[idx][4] > xs[idx][3]
	l_arm_out = xs[idx][5] < xs[idx][6] < xs[idx][7]

	l_dab = ys[idx][5] > ys[idx][6] > ys[idx][7]
	l_elbow_fold = xs[idx][6] > xs[idx][7]
	r_arm_out = xs[idx][4] < xs[idx][3] < xs[idx][2]
	
	if (r_dab) and (r_elbow_fold) and (l_arm_out) and (not arms_crossed):
		''' Right dab'''
		right_dab = True
		left_dab = False
		print("Right dab")
	elif(l_dab) and (l_elbow_fold) and (r_arm_out) and (not arms_crossed):
		''' Left dab '''
		left_dab = True
		right_dab = False
		print("Left dab")
	else:
		''' No dab '''
		right_dab = False
		left_dab = False
		print("No dab")
	return (right_dab, left_dab)


def are_hands_raised():
	''' Right side '''
	if (ys[idx][3] == 0) or (ys[idx][4] == 0):
		print("Right Hand For Human %s Not on Screen" % (idx+1))
	elif right_hand_up:
		print("Right Hand For Human %s Is Up"  % (idx+1))
	elif not right_hand_up:
		print("Right Hand For Human %s Is Not Up" % (idx+1))

	''' Left side '''
	if (ys[idx][6] == 0) or (ys[idx][7] == 0): 
		print("Left Hand For Human %s Not on Screen" % (idx+1))
	elif left_hand_up:
		print("Left Hand For Human %s Is Up" % (idx+1))
	elif not left_hand_up:
		print("Left Hand For Human %s Is Not Up" % (idx+1))

def sigint_handler(sig, iteration):
	cv2.destroyAllWindows()
	sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

if __name__ == '__main__':

	''' Initialize argparse flags '''
	parser = argparse.ArgumentParser(
		description='Extract pose-estimation data from realtime webcam')
	parser.add_argument('--camera', type=int, default=1)
	parser.add_argument('--resize', type=str, default='368x368',
						help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
	parser.add_argument('--resize-out-ratio', type=float, default=4.0,
						help='if provided, resize heatmaps before they are post-processed. default=1.0')
	parser.add_argument('--model', type=str,
						default='mobilenet_thin', help='cmu / mobilenet_thin')
	parser.add_argument('--show-process', type=bool, default=False,
						help='for debug purpose, if enabled, speed for inference is dropped.')
	parser.add_argument('--debug', dest='debug', action='store_true')
	parser.add_argument('--frames-to-append', '-f', dest='frames_to_append', type=int, default=4)
	parser.add_argument('--use-angles', '-a', dest='use_angles', action='store_true')
	parser.add_argument('--only-arm', '-o', dest='use_arm', action='store_true',
						 help='if provided, only saves data from shoulder joint, elbow, and wrist')
	parser.set_defaults(debug=False)
	parser.set_defaults(use_angles=False)
	parser.set_defaults(use_arm=False)
	args = parser.parse_args()
	debug = args.debug
	use_angles = args.use_angles
	use_arm = args.use_arm

	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
	cam = cv2.VideoCapture(args.camera)
	ret_val, image = cam.read()

	v = Var(use_arm)
	NUM_JOINTS = v.get_size()

	num_frames = args.frames_to_append
	data_file = "data"

	last_xs = np.zeros(NUM_JOINTS)
	last_ys = np.zeros(NUM_JOINTS)
	last_scores = np.zeros(NUM_JOINTS)

	x_diffs = np.zeros((1, NUM_JOINTS))
	y_diffs = np.zeros((1, NUM_JOINTS))
	if use_angles:
		dist_diffs = np.zeros((1, NUM_JOINTS))
		angles = np.zeros((1, NUM_JOINTS))
	score_avgs = np.zeros((1, NUM_JOINTS))
	bad_data = np.zeros(1)
	temp_scores = np.zeros((num_frames, NUM_JOINTS))

	while True:
		last_num_humans = 0
		ret_val, image = cam.read()
		cv2.waitKey(200)

		humans = e.inference(image, resize_to_default=(
			w > 0 and h > 0), upsample_size=args.resize_out_ratio)

		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)

		cv2.imshow('tf-pose-estimation result', image)
		fps_time = time.time()

		if cv2.waitKey(1) == 27:
			break

		num_humans = len(humans)
		if num_humans == 0:
			last_num_humans = 0
			x_diffs = np.concatenate(
				(x_diffs, np.zeros((1, NUM_JOINTS))))
			y_diffs = np.concatenate(
				(y_diffs, np.zeros((1, NUM_JOINTS))))
			score_avgs = np.concatenate((score_avgs, np.zeros((1, NUM_JOINTS))))
			bad_data = np.concatenate((bad_data, np.array([0])))
			if use_angles:
				dist_diffs = np.concatenate((dist_diffs, np.zeros((1,NUM_JOINTS))))
				angles = np.concatenate((angles, np.zeros((1,NUM_JOINTS))))
			continue

		elif num_humans > 0 and last_num_humans != num_humans:

			temp_scores = np.zeros((num_humans, num_frames, NUM_JOINTS))
			last_num_humans = num_humans
			scores = np.empty((num_humans, NUM_JOINTS))
			last_scores = np.zeros((num_humans, NUM_JOINTS))

			xs = np.empty((num_humans, NUM_JOINTS))
			ys = np.empty((num_humans, NUM_JOINTS))

			x_dists = np.empty((num_humans, NUM_JOINTS))
			y_dists = np.empty((num_humans, NUM_JOINTS))

			last_xs = np.zeros((num_humans, NUM_JOINTS))
			last_ys = np.zeros((num_humans, NUM_JOINTS))
			temp_xs = np.zeros((num_humans, num_frames, NUM_JOINTS))
			temp_ys = np.zeros((num_humans, num_frames, NUM_JOINTS))

			iter_num = 1

		parts = []

		for idx, human in enumerate(humans):
			ordered_parts = collections.OrderedDict(
				sorted(human.body_parts.items()))
			parts.append(sorted(human.body_parts.keys()))

			for joint_num in range(NUM_JOINTS):
				if joint_num in parts[idx]:
					data = ordered_parts[joint_num]
					datum = get_data(data)
					x, y, score = datum

					xs[idx][joint_num] = x
					ys[idx][joint_num] = y
					scores[idx][joint_num] = score

				else:
					scores[idx][joint_num] = 0.0
					xs[idx][joint_num] = 0.0
					ys[idx][joint_num] = 0.0

			if use_arm:
				right_hand_up = ys[idx][1] > ys[idx][2] if (
					ys[idx][1] != 0 and ys[idx][2] != 0) else False
				left_hand_up = ys[idx][4] > ys[idx][5] if (
					ys[idx][4] != 0 and ys[idx][5] != 0) else False
			else:
				right_hand_up = ys[idx][3] > ys[idx][4] if (
					ys[idx][3] != 0 and ys[idx][4] != 0) else False
				left_hand_up = ys[idx][6] > ys[idx][7] if (
					ys[idx][6] != 0 and ys[idx][7] != 0) else False

			x_dists[idx] = np.array(
				[(x-y)**2 for x, y in zip(xs[idx], last_xs[idx])])
			y_dists[idx] = np.array(
				[(x-y)**2 for x, y in zip(ys[idx], last_ys[idx])])

			if (not left_hand_up and not right_hand_up) or (((ys[idx][2] == 0) or (ys[idx][3] == 0)) and ((ys[idx][6] == 0) or (ys[idx][7] == 0))):
				scores *= -1
			
			bD = np.array([1]) if True in [(ys[i/NUM_JOINTS][i % NUM_JOINTS] == 0 and last_ys[i/NUM_JOINTS][i % NUM_JOINTS] != 0) or (
				ys[i/NUM_JOINTS][i % NUM_JOINTS] != 0 and last_ys[i/NUM_JOINTS][i % NUM_JOINTS] == 0) for i in range(num_humans*NUM_JOINTS)] else np.array([0])
			
			if iter_num < num_frames:
				temp_xs[idx][iter_num - 1] = x_dists[idx]
				temp_ys[idx][iter_num - 1] = y_dists[idx]
				temp_scores[idx][iter_num - 1] = scores[idx]
				if iter_num < i:
					x_diffs = np.concatenate(
						(x_diffs, np.zeros((1, NUM_JOINTS))))
					y_diffs = np.concatenate(
						(y_diffs, np.zeros((1, NUM_JOINTS))))
					score_avgs = np.concatenate(
						(score_avgs, np.zeros((1, NUM_JOINTS))))
					bad_data = np.concatenate((bad_data, np.array([0])))
			else:
				temp_xs[idx] = np.roll(temp_xs[idx], -1, axis=0)
				temp_xs[idx][-1] = xs[idx]
				temp_ys[idx] = np.roll(temp_ys[idx], -1, axis=0)
				temp_ys[idx][-1] = ys[idx]
				temp_scores[idx] = np.roll(temp_scores[idx], -1, axis=0)
				temp_scores[idx][-1] = scores[idx]
				x_travel = np.sum(temp_xs[idx], axis=0).reshape(1, NUM_JOINTS)
				y_travel = np.sum(temp_ys[idx], axis=0).reshape(1, NUM_JOINTS)
				x_diffs = np.concatenate(
					(x_diffs, x_travel), axis=0)
				y_diffs = np.concatenate(
					(y_diffs, y_travel), axis=0)
				score_avgs = np.concatenate((score_avgs, np.divide(
					(np.sum(temp_scores[idx], axis=0)).reshape(1, NUM_JOINTS), num_frames)))
				bad_data = np.concatenate((bad_data, bD), axis=0)
				if use_angles:
					distTravel = np.array([x + y for x,y in zip(x_travel, y_travel)]).reshape(1, NUM_JOINTS)
					angle = np.array([math.atan2(y,x) for x, y in zip(x_travel[0], y_travel[0])]).reshape(1,NUM_JOINTS) 
					dist_diffs = np.concatenate((dist_diffs, distTravel), axis=0)
					angles = np.concatenate((angles, angle), axis=0)

			print("X POSE RECOGNITION")
			x_arms_recognition()

			print("Y POSE RECOGNITION")				
			y_arms_recognition()

			print("DAB RECOGNITION")
			dab_recognition()

			print("HAND RAISE RECOGNITION")				
			are_hands_raised()

			iter_num += 1

			last_xs = np.copy(xs)
			last_ys = np.copy(ys)
			last_scores = np.copy(scores)

		if debug:
			print("x Differences: ", x_diffs)
			print("y Differences: ", y_diffs)
			print("Score Averages: ", score_avgs)
			print("Shape: ", x_diffs.shape)
			print("Bad Data Array: ", bad_data)
			print("Amount of Good data: ", collections.Counter(bad_data))
	
	cv2.destroyAllWindows()
