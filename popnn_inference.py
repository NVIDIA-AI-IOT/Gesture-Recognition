# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.model_selection as ms
import sklearn.preprocessing as pr
import numpy as np
import argparse
import rospy
import signal
import sys
from std_msgs.msg import String, Float64MultiArray, Float64, MultiArrayLayout
import operator
from popnn import Model
from Var import Var
import json

use_cuda = torch.cuda.is_available()

def callNoses(data):
	''' Get positions of noses '''
	global noses
	noses = data

def callXPose(data):
	global x_pose_arr
	x_pose_arr = data

def callYPose(data):
	global y_pose_arr
	y_pose_arr = data

def callDab(data):
	global dab_arr
	dab_arr = data

def callData(data):
	global lstm_data
	lstm_data = data.data

def inference(model, data, classes):
	output = model(data)
	max_idx, val = max(enumerate(output[0]), key=operator.itemgetter(1))

	return classes[max_idx]

def sigint_handler(sig, iteration):
	''' Handles Ctrl + C. '''
	sys.exit(0)

	
if __name__ == "__main__":
	''' Initialize sigint handler '''
	signal.signal(signal.SIGINT, sigint_handler)

	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', dest='debug', action='store_true')
	parser.add_argument('--ckpt_name', '-c', dest='ckpt_name',
						default='popnn000.ckpt', type=str)
	parser.add_argument('--only-arm', '-o', dest='use_arm', action='store_true')
	parser.add_argument('--multiply-by-score', '-m', dest='m_score', action='store_true')	
	parser.set_defaults(use_arm=False)
	parser.set_defaults(m_score=False)	
	parser.set_defaults(debug=False)
	args = parser.parse_args()
	debug = args.debug
	ckpt_fn = args.ckpt_name
	use_arm = args.use_arm
	m_score = args.m_score
	ckpt_name = "lstmpts/popnn/" + ckpt_fn
	rospy.init_node('lstm_inference', anonymous=True)
	v = Var(use_arm)
	rate = rospy.Rate(v.get_rate())

	''' Subscribers '''
	nose_sub = rospy.Subscriber('nose', Float64MultiArray, callNoses)
	x_pose_sub = rospy.Subscriber('xPose', Float64MultiArray, callXPose)
	y_pose_sub = rospy.Subscriber('yPose', Float64MultiArray, callYPose)
	dab_sub = rospy.Subscriber('dab', Float64MultiArray, callDab)
	data_sub = rospy.Subscriber('data', String, callData)

	''' Publishers '''
	pub_x = rospy.Publisher('distPOP', String, queue_size=10)
	pub_score = rospy.Publisher('scorePOP', String, queue_size=10)
	pub_nose = rospy.Publisher('nosePOP', Float64MultiArray, queue_size=10)
	pub_res = rospy.Publisher('popnn', String, queue_size=10)
	pub_x_pose = rospy.Publisher('xPose', Float64MultiArray, queue_size = 10)
	pub_y_pose = rospy.Publisher('yPose', Float64MultiArray, queue_size = 10)
	pub_dab = rospy.Publisher('dab', Float64MultiArray, queue_size = 10)

	input_size = v.get_size()
	num_features = v.get_num_features() - 1 if m_score else v.get_num_features()
	classes = v.get_classes()
	
	model = Model(input_size=input_size, num_features=num_features, dropout=v.get_POPNN()['dropout'])
	model = model.cuda() if use_cuda else model
	model.load_state_dict(torch.load(ckpt_name))
	last_combined = np.zeros(0)

	while True:
		global noses
		global x_pose_arr
		global y_pose_arr
		global dab_arr
		global lstm_data

		inference_data = json.loads(lstm_data)
		x = np.array(inference_data['0']).reshape(1, 1, -1)
		y = np.array(inference_data['1']).reshape(1, 1, -1)
		x_dist = np.array(inference_data['2']).reshape(1, 1, -1)
		y_dist = np.array(inference_data['3']).reshape(1, 1, -1)
		scores = np.array(inference_data['4']).reshape(1, 1, -1)
		score_string = scores.tostring()
		x_string = x.tostring()
		num_humans = x.shape[-1] / input_size
		in_size = input_size*num_humans

		if m_score:
			x = np.stack(x[0][0][i]*scores[0][0][i] for i in range(x.shape[-1])).reshape(1, in_size)
			y = np.stack(y[0][0][i]*scores[0][0][i] for i in range(y.shape[-1])).reshape(1, in_size)
			# xDist = np.stack(xDist[0][0][i]*scores[0][0][i] for i in range(xDist.shape[-1])).reshape(1, input_size)
			# yDist = np.stack(yDist[0][0][i]*scores[0][0][i] for i in range(yDist.shape[-1])).reshape(1, input_size)

		# combined = np.concatenate(
		# 	(newX.reshape(numHumans, 1, -1), newY.reshape(numHumans, 1, -1)), axis=1)
		combined = np.concatenate((x, y), axis=1) if m_score else np.concatenate((x, y, scores), axis=1)
		combined = combined.reshape(num_humans, -1)
		combined = pr.normalize(combined).reshape(num_humans, num_features, input_size)
		outs = []
		if not np.array_equal(combined, last_combined):
			for human in combined:
				datum = torch.from_numpy(human).cuda().float().view(1, num_features, input_size)
				out = inference(model, datum, classes)
				outs.append(out)
				last_combined = combined

		else:
			outs = last_outs

		s = ", ".join(outs)
		x_poses = x_pose_arr.data
		y_poses = y_pose_arr.data
		dabs = dab_arr.data
		print "-" * 25 + "HUMAN REPORT" + "-" * 25
		for idx in range(num_humans):
			pose_string = "Person %d: Action: " % idx
			pose_string += outs[idx]
			pose_string += ", X-Pose : Yes" if x_poses[idx] == 1 else ", X-Pose : No"
			pose_string += ", Y-Pose : Yes" if y_poses[idx] == 1 else ", Y-Pose : No"
			pose_string += ", Dab : "
			if dabs[idx] == 1:
				pose_string += "Right Dab"
			elif dabs[idx] == 2:
				pose_string += "Left Dab"
			else:
				pose_string += "No Dab"
			print (pose_string)

		last_combined = combined		
		last_outs = outs
		outs = []

		pub_res.publish(s)
		pub_nose.publish(noses)
		pub_score.publish(score_string)
		pub_x.publish(x_string)
		pub_x_pose.publish(x_pose_arr)
		pub_y_pose.publish(y_pose_arr)
		pub_dab.publish(dab_arr)
		rate.sleep()

