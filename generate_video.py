# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import argparse
import time
import os
import cv2
import numpy as np
import time
import glob
import sys
import signal
from Var import Var

if __name__ == "__main__":

	''' Initialize argparse flags '''
	parser = argparse.ArgumentParser("Video Capture and labeler")
	parser.add_argument('--camera', type=int, default=0)
	parser.add_argument('--time', type=float, default=100, help='The ammount of time to capture for each label given')
	parser.add_argument('--dir', type=str, default='data', help='The output directory of this script')
	args = parser.parse_args()

	def sigint_handler(sig, iteration):
		''' Handles Ctrl + C. Save the data into npz files. This data will be inputted into the neural network '''
		os.remove(VIDEO_DIR+("%s.avi" % last_file_idx))
		sys.exit(0)

	''' Initialize sigint handler '''
	signal.signal(signal.SIGINT, sigint_handler)

	out_dir = args.dir
	out_dir = out_dir if out_dir[-1] == '/' else out_dir + '/'

	''' Initialize video and label paths '''
	VIDEO_DIR = out_dir + 'video/videos/'
	LABEL_DIR  = out_dir + 'video/labels/'

	''' Initialize options for labels (right now it's binary for either wave or no wave) '''
	v = Var()
	LABEL_OPTS = v.get_classes()

	if not os.path.isdir(VIDEO_DIR):
		os.makedirs(VIDEO_DIR)
	if not os.path.isdir(LABEL_DIR):
		os.makedirs(LABEL_DIR)

	''' Initialize camera '''
	cap = cv2.VideoCapture(args.camera)
	cap_time = args.time

	try:
		last_file_idx = sorted([int(f_name.split('/')[-1].split('.')[0]) for f_name in glob.glob(VIDEO_DIR+"*")], key=int)[-1] + 1
		print "Last file number %d" % last_file_idx
	except:
		print "failed"
		last_file_idx = 1
	fps_time = 0
	fourcc = cv2.VideoWriter_fourcc(*'x264')
	while True:
		_, img = cap.read()
		out = cv2.VideoWriter(VIDEO_DIR + ("%s.avi" % last_file_idx), fourcc, 30.0, (img.shape[1], img.shape[0]))
		start_time = time.time()
		while time.time() - start_time < cap_time:
			_, img = cap.read()
			if _:
				out.write(img)

				''' Show webcam, and display the fps in the upper right corner of the view '''
				cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)),
						(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.imshow("IMG", img)
				if cv2.waitKey(1) == 27:
					os.remove(VIDEO_DIR+("%s.avi" % last_file_idx))
					exit(0)
					
			fps_time = time.time()

		with open(LABEL_DIR + ("%s.txt" % last_file_idx), "w") as label_file:

			''' Print label options for user to choose from '''
			print LABEL_OPTS

			while True:
				label_choice = raw_input("Enter number to select action preformed: ")
				if label_choice == "q":
					os.remove(VIDEO_DIR+("%s.avi" % last_file_idx))
					os.remove(LABEL_DIR+("%s.txt" % last_file_idx))
					out.release()
					cap.release()
					cv2.destroyAllWindows()
					exit(0)
				try:
					label_choice = int(label_choice)
				except:
					continue
				break
			label_file.write(LABEL_OPTS[label_choice])
		last_file_idx += 1
		out.release()
