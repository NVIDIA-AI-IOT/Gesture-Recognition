# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import os
import rospy
import tensorflow as tf
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pr
import time
from std_msgs.msg import String, Float64MultiArray, Float64, MultiArrayLayout
import argparse
import signal
import operator
import json
from Var import Var

def sigint_handle(sig, frame):
    exit(0)

signal.signal(signal.SIGINT, sigint_handle)

''' Set callbacks for ROS '''
def callNoses(data):
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
	global popnn_data
	popnn_data = data.data

if __name__ == "__main__":
	''' Debug argument '''
	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', dest='debug', action='store_true')
	parser.add_argument('--ckpt_name', '-c', dest='ckpt_name', type=str)
	parser.add_argument('--only-arm', '-o', dest='use_arm', action='store_true')
	parser.add_argument('--multiply-by-score', '-m', dest='m_score', action='store_true')
	parser.set_defaults(m_score=False)
	parser.set_defaults(use_arm=False)
	parser.set_defaults(debug=False)     
	args = parser.parse_args()
	debug = args.debug
	use_arm = args.use_arm
	m_score = args.m_score

	working_dir = os.getcwd() + "/"
	v = Var(use_arm)
	input_size = v.get_size()
	num_classes = v.get_num_classes()
	popnn_vars = v.get_POPNN()
	''' Basic TF Data/Info for our net '''
	x = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
	y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32)

	tf.logging.set_verbosity(tf.logging.INFO)
	ninput = input_size
	nhidden1 = popnn_vars['hidden1']
	nhidden2 = popnn_vars['hidden2']
	nhidden3 = popnn_vars['hidden3']
	noutput = popnn_vars['hidden4']
	numJoints = ninput

	''' Initialize rospy node and rate '''
	rospy.init_node('popnninference', anonymous=True)

	rate = rospy.Rate(20)

	''' Set publishers and subscribers '''
	x_pose_sub = rospy.Subscriber('xPose', Float64MultiArray, callXPose)
	y_pose_sub = rospy.Subscriber('yPose', Float64MultiArray, callYPose)
	dab_sub = rospy.Subscriber('dab', Float64MultiArray, callDab)
	nose_sub = rospy.Subscriber('nose', Float64MultiArray, callNoses)
	data_sub = rospy.Subscriber('data', String, callData)

	pub_res = rospy.Publisher('popnn', String, queue_size=10)
	dist_pub = rospy.Publisher('distPOP', String, queue_size=10)
	score_pub = rospy.Publisher('scorePOP', String, queue_size=10)
	nose_pub = rospy.Publisher('nosePOP', Float64MultiArray, queue_size=10)
	x_pose_pub = rospy.Publisher('xPose', Float64MultiArray, queue_size = 10)
	y_pose_pub = rospy.Publisher('yPose', Float64MultiArray, queue_size = 10)
	dab_pub = rospy.Publisher('dab', Float64MultiArray, queue_size = 10)

	''' Define network architecture '''
	weights = {
		'h1': tf.Variable(tf.random_normal([ninput, nhidden1])),
		'h2': tf.Variable(tf.random_normal([nhidden1,nhidden2])),
		'h3': tf.Variable(tf.random_normal([nhidden2, nhidden3])),
		#'h4': tf.Variable(tf.random_normal([nhidden3, nhidden4])),
		'out': tf.Variable(tf.random_normal([nhidden3, noutput]))
	}
	'''b4 is needed because trained model had it in biases'''
	biases = {
		'b1': tf.Variable(tf.random_normal([nhidden1])),
		'b2': tf.Variable(tf.random_normal([nhidden2])),
		'b3': tf.Variable(tf.random_normal([nhidden3])),
		'b4': tf.Variable(tf.random_normal([8])),
		'out': tf.Variable(tf.random_normal([noutput]))
	}

	keep_prob = tf.placeholder("float")

	def network(x, weights, biases, keep_prob):
		layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer1 = tf.nn.relu(layer1)
		layer1 = tf.nn.dropout(layer1, keep_prob)
		layer2 = tf.add(tf.matmul(layer1, weights['h2']),biases['b2'])
		layer2 = tf.nn.relu(layer2)
		layer2 = tf.nn.dropout(layer2, keep_prob)
		layer3 = tf.add(tf.matmul(layer2, weights['h3']),biases['b3'])
		layer3 = tf.nn.relu(layer3)
		layer3 = tf.nn.dropout(layer3, keep_prob)
		#layer4 = tf.add(tf.matmul(layer3, weights['h4']),biases['b4'])
		#layer4 = tf.nn.relu(layer4)
		#layer4 = tf.nn.dropout(layer4, keep_prob)
		outlayer = tf.layers.dense(inputs=layer3, units = num_classes)
		outlayer = tf.nn.softmax(outlayer, name  ="softmax_tensor")
		return outlayer

	''' Define inference settings '''
	predictions = network(x, weights, biases, keep_prob)
	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
	# optimizer = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(cost)
	time.sleep(1)
	print num_classes
	saver = tf.train.Saver(tf.all_variables())
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
	NUM_CORES=4
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)) as sess:
		sess.run(tf.global_variables_initializer())

		''' Restore previous weights for inferencing '''
		saver.restore(sess, working_dir+"ckpts/popnn/" + args.ckpt_name) 
		while True:

			''' Extrapolate np data from score_string and dist_string '''
			inference_data = json.loads(popnn_data)
			dists = np.array(inference_data['2']) #if use_angles method is used with thread inf
			scores = np.array(inference_data['4'])
			dist_string = dists.tostring()
			score_string = scores.tostring()

			shape = dists.shape[-1]
			num_humans = shape/numJoints
			dists = dists.reshape(num_humans, numJoints)
			scores = scores.reshape(num_humans, numJoints)
			''' Format input data '''
			z = np.stack(dists[i]*scores[i] for i in range(len(dists)))
			z = pr.normalize(z)
			''' Predict results based on inputs and append to output string '''
			results = predictions.eval({x:z, keep_prob:1.0})

			x_poses = x_pose_arr.data
			y_poses = y_pose_arr.data
			dabs = dab_arr.data
			ros_string = ''
			if (debug):
				print("RESULTS", results)
			
			print "-" * 25 + "HUMAN REPORT" + "-" * 25
			for idx, out in enumerate(results):
				print_string = "Person %d: Action: " % idx
				try:
					if out[0] > out[1]:
						
						ros_string+="No Wave"
						print_string+="No Wave"
						if (debug):
							print("No Wave For Human %s" % (idx+1))
					elif out[0] == out[1]:
						ros_string+="Inconclusive"
						print_string+="Inconclusive"
						if (debug):
							print("Inconclusive For Human %s" % (idx+1))
					else:
						ros_string += "Wave"
						print_string+="Wave"
						if (debug):
							print("Wave For Human %s" % (idx+1))
					if idx!=len(results)-1:
						ros_string+=", "
					print_string += ", X-Pose : Yes" if x_poses[idx] == 1 else ", X-Pose : No"
					print_string += ", Y-Pose : Yes" if y_poses[idx] == 1 else ", Y-Pose : No"
					print_string += ", Dab : "
					if dabs[idx] == 1:
						print_string += "Right Dab"
					elif dabs[idx] == 2:
						print_string += "Left Dab"
					else:
						print_string += "No Dab"
				except:
					print("No humans in frame")
				print(print_string)
			if (debug):
				print("DIST", np.fromstring(dist_string))
			''' Publish input data, nose position data, and predictions for wave or no wave '''
			pub_res.publish(ros_string)
			dist_pub.publish(dist_string)
			score_pub.publish(score_string)
			nose_pub.publish(noses)
			x_pose_pub.publish(x_pose_arr)
			y_pose_pub.publish(y_pose_arr)
			dab_pub.publish(dab_arr)
			if (debug):
				print("published")
			rate.sleep()
