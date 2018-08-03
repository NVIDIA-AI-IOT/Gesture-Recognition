# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import sklearn.model_selection as ms
import sklearn.preprocessing as pr
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import rospy
import signal
import sys
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from std_msgs.msg import MultiArrayLayout
import operator
from lstm import Model
from Var import Var
import operator
import json

plt.switch_backend('agg')
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

def inference(model, data, states, classes):
	output, states = model(data, states)
	max_idx, val = max(enumerate(output[0]), key=operator.itemgetter(1))
	states = (states[0].detach(), states[1].detach())

	return classes[max_idx], states


def sigint_handler(sig, iteration):
	''' Handles Ctrl + C. '''
	sys.exit(0)

if __name__ == "__main__":

	''' Initialize sigint handler '''
	signal.signal(signal.SIGINT, sigint_handler)

	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', dest='debug', action='store_true')
	parser.add_argument('--only-arm', '-o', dest='use_arm', action='store_true')
	parser.set_defaults(use_arm=False)
	parser.add_argument('--ckpt_name', '-c', dest='ckpt_name',
						default='lstm307.ckpt', type=str)
	parser.add_argument('--multiply-by-score', '-m', dest='m_score', action='store_true')
	parser.set_defaults(m_score=False)	
	parser.set_defaults(debug=False)
	args = parser.parse_args()
	debug = args.debug
	use_arm = args.use_arm
	m_score = args.m_score
	ckpt_name = "lstmpts/" + args.ckpt_name
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
	pubX = rospy.Publisher('distPOP', String, queue_size=10)
	pubScore = rospy.Publisher('scorePOP', String, queue_size=10)
	pubNoses = rospy.Publisher('nosePOP', Float64MultiArray, queue_size=10)
	pubRes = rospy.Publisher('popnn', String, queue_size=10)
	pubXPose = rospy.Publisher('xPose', Float64MultiArray, queue_size = 10)
	pubYPose = rospy.Publisher('yPose', Float64MultiArray, queue_size = 10)
	pubDab = rospy.Publisher('dab', Float64MultiArray, queue_size = 10)

	lstm_vars = v.get_LSTM()
	input_size = v.get_size()
	num_layers = lstm_vars['num_layers']
	hidden_size = lstm_vars['hidden_size']
	seq_len = lstm_vars['seq_len']
	num_features = v.get_num_features() - 1 if m_score else v.get_num_features()
	dropout = lstm_vars['dropout']
	classes = v.get_classes()

	model = Model(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, seq_len=seq_len, num_features=num_features, dropout=dropout, mode="TEST")
	model = model.cuda() if use_cuda else model
	ckpt_fn = args.ckpt_name
	model.load_state_dict(torch.load("lstmpts/%s" % ckpt_fn))
	states = model.init_states()
	lastCombined = np.zeros(0)

	while True:
		global noses
		global x_pose_arr
		global y_pose_arr
		global dab_arr
		global lstm_data

		inference_data = json.loads(lstm_data)
		x = np.array(inference_data['0']).reshape(1, 1, -1)
		y = np.array(inference_data['1']).reshape(1, 1, -1)
		xDist = np.array(inference_data['2']).reshape(1, 1, -1)
		yDist = np.array(inference_data['3']).reshape(1, 1, -1)
		scores = np.array(inference_data['4']).reshape(1, 1, -1)
		scoreString = scores.tostring()
		xString = x.tostring()
		numHumans = x.shape[-1] / input_size
		in_size = input_size*numHumans

		if m_score:
			x = np.stack(x[0][0][i]*scores[0][0][i] for i in range(x.shape[-1])).reshape(1, in_size)
			y = np.stack(y[0][0][i]*scores[0][0][i] for i in range(y.shape[-1])).reshape(1, in_size)
			# xDist = np.stack(xDist[0][0][i]*scores[0][0][i] for i in range(xDist.shape[-1])).reshape(1, input_size)
			# yDist = np.stack(yDist[0][0][i]*scores[0][0][i] for i in range(yDist.shape[-1])).reshape(1, input_size)

		# combined = np.concatenate(
		# 	(newX.reshape(numHumans, 1, -1), newY.reshape(numHumans, 1, -1)), axis=1)
		combined = np.concatenate((x, y), axis=1) if m_score else np.concatenate((x, y, scores), axis=1)
		combined = combined.reshape(numHumans, -1)
		combined = pr.normalize(combined).reshape(numHumans, num_features, input_size)
		outs = []
		if not np.array_equal(combined, lastCombined):
			for human in combined:
				datum = torch.from_numpy(human).cuda().float().view(1, num_features, input_size)
				out, states = inference(model, datum, states, classes)
				outs.append(out)
			
		else:
			outs = lastOuts
		s = ", ".join(outs)

		# xPoses = np.array([int(element) for element in list(x_pose_arr)[1:-1] if (element!='' and element!=' ' and element!='.')])
		# yPoses = np.array([int(element) for element in list(y_pose_arr)[1:-1] if (element!='' and element!=' ' and element!='.')])
		# dabs = ([int(element) for element in list(dab_arr)[1:-1] if (element!='' and element!=' ' and element!='.')])
		xPoses = x_pose_arr.data
		yPoses = y_pose_arr.data
		dabs = dab_arr.data

		print "-" * 25 + "HUMAN REPORT" + "-" * 25
		for idx in range(numHumans):
			try:
				pose_string = "Person %d: Action: " % idx
				pose_string += outs[idx]
				pose_string += ", X-Pose : Yes" if xPoses[idx] == 1 else ", X-Pose : No"
				pose_string += ", Y-Pose : Yes" if yPoses[idx] == 1 else ", Y-Pose : No"
				pose_string += ", Dab : "
				if dabs[idx] == 1:
					pose_string += "Right Dab"
				elif dabs[idx] == 2:
					pose_string += "Left Dab"
				else:
					pose_string += "No Dab"
				print (pose_string)
				
			except:
				pose_string = "No Humans in Frame"
				print (pose_string)
				break

		lastCombined = combined		
		lastOuts = outs
		outs = []

		

		pubRes.publish(s)
		pubNoses.publish(noses)
		pubScore.publish(scoreString)
		pubX.publish(xString)
		pubXPose.publish(x_pose_arr)
		pubYPose.publish(y_pose_arr)
		pubDab.publish(dab_arr)
		rate.sleep()
