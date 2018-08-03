# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import collections
import numpy as np
import math
import json
from Var import Var

import rospy
from std_msgs.msg import String, Float64MultiArray, Float64, MultiArrayLayout

class PosePub:

	def __init__(self, refresh_rate, frames_to_append, use_angles=False, use_arm=False, bad_data_flag=False, debug=False):
		'''Initialize ROS, np arrays, flags from thread_inf argparse, etc.'''
		''' num_frames is the amonut of pictures we are stringing together to take the differences from.  '''
		self.num_frames = frames_to_append
		self.use_angles = use_angles
		self.bad_data_flag = bad_data_flag
		self.debug = debug

		self.pub_nose = rospy.Publisher('nose', Float64MultiArray, queue_size=1)
		self.pub_x_pose = rospy.Publisher('xPose', Float64MultiArray, queue_size = 1)
		self.pub_y_pose = rospy.Publisher('yPose', Float64MultiArray, queue_size = 1)
		self.pub_dab = rospy.Publisher('dab', Float64MultiArray, queue_size = 1)
		self.pub_data = rospy.Publisher('data', String, queue_size=1)
		rospy.init_node('lstmTalker', anonymous=True)
		
		self.rate = rospy.Rate(refresh_rate)

		self.var = Var(use_arm)
		self.NUM_JOINTS = self.var.get_size()
		self.iter_num = 1

		self.last_xs = np.zeros(self.NUM_JOINTS)
		self.last_ys = np.zeros(self.NUM_JOINTS)
		self.last_scores = np.zeros(self.NUM_JOINTS)
		self.x_dists = np.empty((1, self.NUM_JOINTS))
		self.y_dists = np.empty((1, self.NUM_JOINTS))

		
		self.tmp_scores = np.zeros((self.num_frames, self.NUM_JOINTS))
		self.tmp_xs = np.zeros((self.num_frames, self.NUM_JOINTS))
		self.tmp_ys = np.zeros((self.num_frames, self.NUM_JOINTS))
		self.scores = np.empty(self.NUM_JOINTS)

		self.xs = np.empty(self.NUM_JOINTS)

		self.ys = np.empty(self.NUM_JOINTS)

		self.dists = np.empty(self.NUM_JOINTS)
		self.last_num_humans = 0

		self.arms_crossed = False
		self.y_arms = False
		self.right_dab = False
		self.left_dab = False
		self.x_poses = np.zeros(1)
		self.y_poses = np.zeros(1)
		self.dabs = np.zeros(1)
		self.data = {'data': {}}

	def get_data(self, body_part):
		'''get x position, y position, and score of a body_part from pose estimation'''
		return body_part.x, body_part.y, body_part.score

	def is_ok(self):
		return not rospy.is_shutdown()

	def x_arms_recognition(self, idx, left_hand_up, right_hand_up): 
		'''Check if arms are in X-Pose'''
		wrist_dist_x = self.xs[idx][4] - self.xs[idx][7]
		if (right_hand_up and left_hand_up) and (wrist_dist_x > 0) and (self.xs[idx][4] > self.xs[idx][7]):
			self.arms_crossed = True
			x_pose_string = 'X-Pose'
		else:
			self.arms_crossed = False
			x_pose_string = 'No X-Pose'
		return x_pose_string

	def y_arms_recognition(self, idx, left_hand_up, right_hand_up):
		'''Check if arms are in Y-Pose'''
		wristDistX = self.xs[idx][4] - self.xs[idx][7]
		elbows_higher_than_shoulders = (self.ys[idx][6] < self.ys[idx][5]) and (self.ys[idx][3] < self.ys[idx][2])
		if (right_hand_up and left_hand_up) and (not self.arms_crossed) and ((wristDistX < 0)) and (self.xs[idx][4] < self.xs[idx][3] < self.xs[idx][2]) and (self.xs[idx][5] < self.xs[idx][6] < self.xs[idx][7]) and elbows_higher_than_shoulders:
			self.y_arms = True
			y_pose_string = 'Y-Pose'
		else:
			self.y_arms = False
			y_pose_string = 'No Y-Pose'
		return  y_pose_string

	def dab_recognition(self, idx):
		'''Checks if humans in frame are dabbing'''
		r_dab = self.ys[idx][2] > self.ys[idx][3] > self.ys[idx][4]
		r_elbow_fold = self.xs[idx][4] > self.xs[idx][3]
		l_arm_out = self.xs[idx][5] < self.xs[idx][6] < self.xs[idx][7]

		l_dab = self.ys[idx][5] > self.ys[idx][6] > self.ys[idx][7]
		l_elbow_fold = self.xs[idx][6] >self.xs[idx][7]
		r_arm_out = self.xs[idx][4] < self.xs[idx][3] < self.xs[idx][2]
		
		if (r_dab) and (r_elbow_fold) and (l_arm_out) and (not self.arms_crossed):
			''' Right dab'''
			self.right_dab = True
			self.left_dab = False
			dab_string = 'Right Dab'
		elif(l_dab) and (l_elbow_fold) and (r_arm_out) and (not self.arms_crossed):
			''' Left dab '''
			self.left_dab = True
			self.right_dab = False
			person_num = (idx + 1)
			dab_string = 'Left Dab'
		else:
			''' No dab '''
			self.right_dab = False
			self.left_dab = False
			dab_string = 'No Dab'
		return  dab_string
					

	def are_hands_raised(self, idx, left_hand_up, right_hand_up):
		'''checks if hands of humans in frame are up'''
		''' Is right hand up? '''
		if (self.ys[idx][3] == 0) or (self.ys[idx][4] == 0):
			print("Right Hand For Human %s Not on Screen" % (idx+1))
		elif right_hand_up:
			print("Right Hand For Human %s Is Up"  % (idx+1))
		elif not right_hand_up:
			print("Right Hand For Human %s Is Not Up" % (idx+1))

		''' Is left hand up? '''
		if (self.ys[idx][6] == 0) or (self.ys[idx][7] == 0): 
			print("Left Hand For Human %s Not on Screen" % (idx+1))
		elif left_hand_up:
			print("Left Hand For Human %s Is Up" % (idx+1))
		elif not left_hand_up:
			print("Left Hand For Human %s Is Not Up" % (idx+1))

	def pub_null(self, num_humans):
		'''publish numpy arrays of zeros if no humans in frame'''
		dummy = np.zeros((1 , self.NUM_JOINTS))
		send_noses = Float64MultiArray()
		send_noses.data = [0.0 for i in range(2)]# 2 for x,y of nose
		pose_dummy = Float64MultiArray()
		pose_dummy.data = [0.0 for i in range(num_humans)]  if num_humans != 0 else [0.0]
		features = [dummy, dummy, dummy, dummy, dummy, dummy] #x, y, xdist, ydist, score
		self.pub_inference_data(features)
		self.pub_nose.publish(send_noses)
		self.pub_x_pose.publish(pose_dummy)
		self.pub_y_pose.publish(pose_dummy)
		self.pub_dab.publish(pose_dummy)

	def pub_inference_data(self, features):
		'''publish all data needed by inference'''
		self.data['data'] = {}
		for idx, feature in enumerate(features):
			feature = feature.reshape(1, -1)[0].tolist()
			self.data['data'][idx] = feature
		
		self.data['data'] = json.dumps(self.data['data'])
		self.pub_data.publish(self.data['data'])
		print("IS PUBLISHING REG")

	def post_n_pub(self, humans):
		'''create data and publish it to inference and robot '''
		num_humans = len(humans)
		if num_humans == 0:
			self.pub_null(num_humans)
			self.rate.sleep()
			return
		elif num_humans > 0 and self.last_num_humans != num_humans:

			self.tmp_xs = np.zeros((num_humans, self.num_frames, self.NUM_JOINTS))
			self.tmp_ys = np.zeros((num_humans, self.num_frames, self.NUM_JOINTS))
			self.tmp_scores = np.zeros((num_humans, self.num_frames, self.NUM_JOINTS))

			self.last_num_humans = num_humans

			self.scores = np.empty((num_humans, self.NUM_JOINTS))

			self.xs = np.empty((num_humans, self.NUM_JOINTS))

			self.ys = np.empty((num_humans, self.NUM_JOINTS))

			self.x_dists = np.empty((num_humans, self.NUM_JOINTS))
			self.y_dists = np.empty((num_humans, self.NUM_JOINTS))

			self.x_poses = np.zeros(num_humans)
			self.y_poses = np.zeros(num_humans)
			self.dabs = np.zeros(num_humans)

			self.scores = np.empty((num_humans, self.NUM_JOINTS))
			self.last_xs = np.zeros((num_humans, self.NUM_JOINTS))
			self.last_ys = np.zeros((num_humans, self.NUM_JOINTS))
			self.last_scores = np.zeros((num_humans, self.NUM_JOINTS))

			self.iter_num = 1

		parts = []
		bad_data_arr = []
		noses = []
		final_noses = []

		for idx, human in enumerate(humans):
			ordered_parts = collections.OrderedDict(
				sorted(human.body_parts.items()))
			parts.append(sorted(human.body_parts.keys()))
			for joint_num in range(self.NUM_JOINTS):
				if joint_num in parts[idx]:
					data = ordered_parts[joint_num]
					datum = self.get_data(data)
					x, y, score = datum

					self.xs[idx][joint_num] = x
					self.ys[idx][joint_num] = y
					self.scores[idx][joint_num] = score
				else:
					''' If a certain joint is not recognized or is not in the frame, its x/y cooridinates and self.scores will be set to 0.0 '''
					self.xs[idx][joint_num] = 0.0
					self.ys[idx][joint_num] = 0.0
					self.scores[idx][joint_num ] = 0.0
			noses.append(self.xs[idx][0])
			noses.append(self.ys[idx][0])
		
			right_hand_up = self.ys[idx][3] > self.ys[idx][4] if (
				self.ys[idx][3] != 0 and self.ys[idx][4] != 0) else False
			left_hand_up = self.ys[idx][6] > self.ys[idx][7] if (
				self.ys[idx][6] != 0 and self.ys[idx][7] != 0) else False

			
			''' Uncomment this statement if rightHand isn't being properly recognized by the neural network. Sometimes, this will fix the issue. '''
			
			''' if right_hand_up and not left_hand_up:
				print("switch")
				self.ys[idx][5] = self.ys[idx][2]
				self.ys[idx][6] = self.ys[idx][3]
				self.ys[idx][7] = self.ys[idx][4]
				self.xs[idx][5] = self.xs[idx][2]
				self.xs[idx][6] = self.xs[idx][3]
				self.xs[idx][7] = self.xs[idx][4]
				self.xs[idx][2] = 0
				self.xs[idx][3] = 0
				self.xs[idx][4] = 0
				self.ys[idx][2] = 0
				self.ys[idx][3] = 0
				self.ys[idx][4] = 0 '''

			# Check for x poses, y poses, dabs	
			x_pose_string = self.x_arms_recognition(idx, left_hand_up, right_hand_up)
			self.x_poses[idx] = 1.0 if x_pose_string == 'X-Pose' else 0.0
			y_pose_string = self.y_arms_recognition( idx, left_hand_up, right_hand_up)
			self.y_poses[idx] = 1.0 if y_pose_string == 'Y-Pose' else 0.0
			dab_string = self.dab_recognition(idx)
			if dab_string == 'Right Dab':
				self.dabs[idx] = 1.0
			elif dab_string == 'Left Dab':
				self.dabs[idx] = 2.0
			else:
				self.dabs[idx] = 0.0

			self.x_dists[idx] = np.array(
				[(x-y)**2 for x, y in zip(self.xs[idx], self.last_xs[idx])])
			self.y_dists[idx] = np.array(
				[(x-y)**2 for x, y in zip(self.ys[idx], self.last_ys[idx])])

			''' If the the left arm or right arm is facing downwards, its score will be multiplied by -1 '''
			if (not left_hand_up and not right_hand_up) or (((self.ys[idx][2] == 0) or (self.ys[idx][3] == 0)) and ((self.ys[idx][6] == 0) or (self.ys[idx][7] == 0))):
				self.scores *= -1
			bad_data = 1 if True in [(self.ys[i/self.NUM_JOINTS][i % self.NUM_JOINTS] == 0 and self.last_ys[i/self.NUM_JOINTS][i % self.NUM_JOINTS] != 0) or (
				self.ys[i/self.NUM_JOINTS][i % self.NUM_JOINTS] != 0 and self.last_ys[i/self.NUM_JOINTS][i % self.NUM_JOINTS] == 0) for i in range(num_humans*self.NUM_JOINTS)] else 0
			if not self.bad_data_flag:
				bad_data = 0
			bad_data_arr.append(bad_data)
			if self.iter_num < self.num_frames and bad_data == 0:
				self.tmp_xs[idx][self.iter_num - 1] = self.x_dists[idx]
				self.tmp_ys[idx][self.iter_num - 1] = self.y_dists[idx]
				self.tmp_scores[idx][self.iter_num - 1] = self.scores[idx]

			elif bad_data == 0:
				self.tmp_xs[idx] = np.roll(self.tmp_xs[idx], -1, axis=0)
				self.tmp_xs[idx][-1] = self.x_dists[idx]
				self.tmp_ys[idx] = np.roll(self.tmp_ys[idx], -1, axis=0)
				self.tmp_ys[idx][-1] = self.y_dists[idx]
				self.tmp_scores[idx] = np.roll(self.tmp_scores[idx], -1, axis=0)
				self.tmp_scores[idx][-1] = self.scores[idx]

		if self.iter_num >= self.num_frames:
			final_xs = []
			final_ys = []
			
			final_scores = []
			tmp_sum_x = np.sum(self.tmp_xs, axis=1)
			tmp_sum_y = np.sum(self.tmp_ys, axis=1)
			tmp_score_avg = np.divide(np.sum(self.tmp_scores, axis=1), self.num_frames)

			if self.use_angles:
				final_dists = []
				final_angles = []
				tmpSumDist = np.array([x + y for x,y in zip (tmp_sum_x, tmp_sum_y)])
				for humanX, humanY in zip(tmp_sum_x, tmp_sum_y):
					tmpSumAngle = np.array([math.atan2(y, x) for x,y in zip(humanX, humanY)])
			
			for idx, bd in enumerate(bad_data_arr):
				if not bd:
					final_xs.append(tmp_sum_x[idx])
					final_ys.append(tmp_sum_y[idx])
					final_scores.append(tmp_score_avg[idx])
					final_noses.append(noses[idx])
					if self.use_angles:
						final_dists.append(tmpSumDist)
						final_angles.append(tmpSumAngle)
			
			x_diffs = np.asarray(final_xs).reshape(
				(1, len(final_xs)*self.NUM_JOINTS))
			y_diffs = np.asarray(final_ys).reshape(
				(1, len(final_ys)*self.NUM_JOINTS))
			score_avgs = np.asarray(final_scores).reshape(
				(1, len(final_scores)*self.NUM_JOINTS))
			all_xs = self.xs.reshape(num_humans, self.NUM_JOINTS)
			all_ys = self.ys.reshape(num_humans, self.NUM_JOINTS)
			if self.use_angles:
				final_dists = np.array(final_dists[0])
				final_angles = np.array(final_angles)
				dist_diffs = final_dists.reshape(1, final_dists.shape[0]*self.NUM_JOINTS)
				angles = final_angles.reshape(1, final_angles.shape[0]*self.NUM_JOINTS)

		self.last_xs = np.copy(self.xs)
		self.last_ys = np.copy(self.ys)
		self.last_scores = np.copy(self.scores)

		if self.iter_num >= self.num_frames:
			x_pose_arr = Float64MultiArray()
			x_pose_arr.data = self.x_poses.reshape(1, -1)[0].tolist()
			y_pose_arr = Float64MultiArray()
			y_pose_arr.data = self.y_poses.reshape(1, -1)[0].tolist()
			dab_arr = Float64MultiArray()
			dab_arr.data = self.dabs.reshape(1, -1)[0].tolist()
			
			if ((x_diffs.shape)[1] > 0):

				send_noses = Float64MultiArray()
				features = [all_xs, all_ys, dist_diffs, angles, score_avgs] if self.use_angles else [all_xs, all_ys, x_diffs, y_diffs, score_avgs]
				self.pub_inference_data(features)
				send_noses.data = noses
				self.pub_nose.publish(send_noses)
				self.pub_x_pose.publish(x_pose_arr)
				self.pub_y_pose.publish(y_pose_arr)
				self.pub_dab.publish(dab_arr)
				print("DAB ARRAY", dab_arr)
				print("X POSE", x_pose_arr)
				print("Y POSE", y_pose_arr)

			else:
				self.pub_null(num_humans)
				print("all bad data")			
		
		if self.debug:
			print("x DIFFERENCES: ", x_diffs)
			print("y DIFFERENCES: ", y_diffs)
			print("SCORE AVERAGES: ", score_avgs)
			print("TEMP SCORES: ", self.tmp_scores)

			''' Is right hand up? '''
			if (self.ys[idx][3] == 0) or (self.ys[idx][4] == 0):
				print("Right Hand For Human %s Not on Screen" % (idx+1))
			elif right_hand_up:
				print("Right Hand For Human %s Is Up" % (idx+1))
			elif not right_hand_up:
				print("Right Hand For Human %s Is Not Up" % (idx+1))

			''' Is left hand up? '''
			if (self.ys[idx][6] == 0) or (self.ys[idx][7] == 0):
				print("Left Hand For Human %s Not on Screen" % (idx+1))
			elif left_hand_up:
				print("Left Hand For Human %s Is Up" % (idx+1))
			elif not left_hand_up:
				print("Left Hand For Human %s Is Not Up" % (idx+1))

		self.rate.sleep()

		self.iter_num += 1

