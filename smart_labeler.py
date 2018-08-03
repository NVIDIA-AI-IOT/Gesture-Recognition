# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 
''' Imports '''
import argparse
import signal
import sys
import logging
import cv2
import numpy as np
import os
import rospy
from std_msgs.msg import String
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf_pose.pafprocess as pafprocess
import time
from Var import Var

global dist
global score
global res

logger = logging.getLogger('TfPoseEstimator-WebCam')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
	'[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

rospy.init_node('smart_labeler', anonymous=True)

def callRes(data):
	''' Gets result from inference '''
	global res
	res = data.data

def callX(data):
	''' Gets x coordinates '''
	global xs
	xs = data.data

def callY(data):
	''' Gets y coordinates '''
	global ys
	ys = data.data

def callScore(data):
	''' Gets score '''
	global score
	score = data.data

def callData(result, scoreData, xData, yData):
	global res
	global xs
	global ys
	global score
	res = result.data
	xs = xData.data
	ys = yData.data
	score = scoreData.data


resultSub = Subscriber("popnn", String)
xSub = Subscriber("X", String)
ySub = Subscriber("Y", String)
scoreSub = Subscriber("score", String)
ts = ApproximateTimeSynchronizer(
	[resultSub, xSub, ySub, scoreSub], 10, 0.1, allow_headerless=True)
ts.registerCallback(callData)
rate = rospy.Rate(5)

dataFile = "data"

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Smart Data Collector')
	parser.add_argument('--debug', dest='debug', action='store_true')
	parser.set_defaults(debug=False)
	parser.add_argument('--only-arm', '-o', dest='use_arm', action='store_true')
	parser.set_defaults(use_arm=False)
	parser.add_argument('--frames_to_append', "-f", dest="frames_to_append", type=int, default=4, help="number of frames that movement is aggregated over")


	args = parser.parse_args()
	use_angles = args.use_angles
	use_arm = args.use_arm
	debug = args.debug
	time.sleep(1)

	x_diffs = []
	y_diffs = []
	score_avgs = []
	last_x_arr = np.zeros(0)
	v = Var(use_arm)
	NUM_JOINTS = v.get_size()
	classes = v.get_classes()
	NUM_FEATURES = v.get_num_features()

	def sigint_handler(sig, iteration):
		''' Handles Ctrl + C. Save the data into npz files. This data will be inputted into the neural network '''
		# modify features to save features of choice
		data_name = "%s/GestureData/gestureData%s" % (dataFile, str(max_num + 1))
		label_name = "%s/Labels/label%s.txt" % (dataFile, str(max_num + 1))	
		features = [x_arr, y_arr, score_avgs]
		data = {}
		for feature_num in range(NUM_FEATURES):
			data[feature_num] = features[feature_num]
		
		bad_data = np.zeros(features[0].shape[0])
		np.savez(data_name, data=data, isBadData=bad_data)
		with open(label_name, "w+") as fn:
			fn.write(correct_result)
			print("Saved SmartLabels #%s" % str(max_num))
		print("saving %d datapoints and exiting" % x_arr.shape[0])
		cv2.destroyAllWindows()
		sys.exit(0)

	''' Initialize sigint handler '''
	signal.signal(signal.SIGINT, sigint_handler)
	iter_num = 1

	''' Determines the file name '''
	max_num = 0
	for file in os.listdir("%s/Labels" % dataFile):
		if file.endswith(".txt"):
			num = int(file.split('.')[0].split('label')[-1])
			if num > max_num:
				max_num = num

	''' Display message to person '''
	print classes
	label_choice = int(raw_input("Enter number to select action preformed: "))
	try:
		correct_result = classes[label_choice]
	except:
		raise KeyError("Key not in Classes Dict")

	while not rospy.is_shutdown():

		local_res = res
		local_x = xs
		local_y = ys
		local_score = score
		labels = []
		for human in local_res.split(','):
			label = ''
			for key, val in classes.items():
				if human == val:
					labels.append(val)
			
			if label == '': #if not given result, set it to the correct result to not use data
				label = correct_result

		x_arr = np.fromstring(local_x)
		y_arr = np.fromstring(local_y)
		score_arr = np.fromstring(local_score)
		try:
			num_humans = x_arr.shape[1]/NUM_JOINTS
		except:
			num_humans = x_arr.shape[0]/NUM_JOINTS

		x_arr = x_arr.reshape(num_humans, NUM_JOINTS)
		y_arr = y_arr.reshape(num_humans, NUM_JOINTS)
		score_arr = score_arr.reshape(num_humans, NUM_JOINTS)

		if np.array_equal(last_x_arr, x_arr):
			if(debug):
				print("Same Data as last frame")
			rate.sleep()
			continue

		num_data = len(x_diffs)
		if len(labels) != num_humans:
			if(debug):
				print("continuing")
			rate.sleep()
			continue

		for idx, label in enumerate(labels):
			x = x_arr[idx]
			y = y_arr[idx]
			score = score_arr[idx]
			if label != correct_result:
				if False in np.isin(x, 0):
					x_diffs.append(x)
					y_diffs.append(y)
					score_avgs.append(score)
				else:
					print("Human %s not in frame" % idx)

				print("Amount of Data collected:%d. Should've gotten %s but got %s" %
					  (len(x_diffs), correct_result, label))

		print("Humans Added: %d" % (len(x_diffs) - num_data))
		last_x_arr = np.copy(x_arr)
		rate.sleep()
