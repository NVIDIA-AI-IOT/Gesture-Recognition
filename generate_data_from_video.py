# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import argparse
import logging
import time
import collections
import cv2
import numpy as np
import math
import os
import signal
import sys
from tqdm import tqdm
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import tf_pose.pafprocess as pafprocess
from Var import Var

global label_name
logger = logging.getLogger('TfPoseEstimator-WebCam')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
	'[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def sigint_handler(sig, iteration):
	''' Handles Ctrl + C. Save the data into npz files. This data will be inputted into the neural network '''
	global label_name
	os.remove(label_name)
	cv2.destroyAllWindows()
	sys.exit(0)

''' Initialize sigint handler '''
signal.signal(signal.SIGINT, sigint_handler)

if __name__ == '__main__':
    global label_name

    ''' Initialize argparse commands '''
    parser = argparse.ArgumentParser(
        description='tf-pose-estimation realtime webcam')
    parser.add_argument('--resize', type=str, default='368x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str,
                        default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--exit-fps', type=int, default=5,
                        help='frames per second')
    parser.add_argument('--frames-to-append', '-f', dest='frames_to_append', type=int, default=4)
    parser.add_argument('--use-angles', '-a', dest='use_angles', action='store_true')
    parser.add_argument('--start-video', '-s', dest='start_video', default=1, type=int)
    parser.add_argument('--end-video', '-e', dest='end_video', default=None, type=int)
    parser.add_argument('--only-arm', '-o', dest='use_arm', action='store_true',
                         help='if provided, only saves data from shoulder joint, elbow, and wrist')
    
    parser.set_defaults(debug=False)
    parser.set_defaults(use_angles=False)
    parser.set_defaults(use_arm=False)

    args = parser.parse_args()
    debug = args.debug
    use_angles = args.use_angles
    DESIRED_FPS = args.exit_fps
    use_arm = args.use_arm
    
    logger.debug('initialization %s : %s' %
                 (args.model, get_graph_path(args.model)))

    ''' Set the width and height of the camera FOV '''
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')

    ''' Set the working directory to the directory we are currently in '''
    working_dir = os.getcwd() + "/"

    ''' Set data file to the folder the data is in '''
    data_file = "data"
    video_amount = len(next(os.walk(working_dir+data_file+'/video/videos'))[2])
    
    ''' Initialize variables '''
    v = Var(use_arm)
    NUM_JOINTS = v.get_size()
    num_features = v.get_num_features()
     
    ''' Initialize variable for number of frames strung together to take differences from. '''
    num_frames = args.frames_to_append

    label_dir = "%s/Labels/%d/" % (data_file, num_frames)
    data_dir = "%s/GestureData/%d/" % (data_file, num_frames)

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    last_xs = np.zeros(NUM_JOINTS)
    last_ys = np.zeros(NUM_JOINTS)

    def get_data(body_part):
        return (body_part.x, body_part.y, body_part.score)

    end_video = video_amount if args.end_video == None else args.end_video
    max_num = 0
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            file_num = int(file.split('.')[0].split('label')[-1])
            if file_num > max_num:
                max_num = file_num
    for vid_num in range(args.start_video, end_video + 1):
        cam = cv2.VideoCapture(working_dir+data_file +
                               '/video/videos/'+str(vid_num)+'.avi')
        x_diffs = np.zeros((0, NUM_JOINTS))
        y_diffs = np.zeros((0, NUM_JOINTS))
        all_xs = np.zeros((0, NUM_JOINTS))
        all_ys = np.zeros((0, NUM_JOINTS))

        if use_angles:
            dist_diffs = np.zeros((0, NUM_JOINTS))
            angles = np.zeros((0, NUM_JOINTS))

        score_avgs = np.zeros((0, NUM_JOINTS))
        bad_data = np.zeros(0)

        temp_scores = np.zeros((num_frames, NUM_JOINTS))
        last_num_humans = 0
        ret_val, image = cam.read()
        num_iters = int(cam.get(7))
        VIDEO_FPS = int(cam.get(5))
        FRAMES_TO_SKIP = int(VIDEO_FPS/DESIRED_FPS - 1)
        for i in tqdm(range(1, int(num_iters*DESIRED_FPS/VIDEO_FPS))):

            for k in range(FRAMES_TO_SKIP):
                _, _ = cam.read()

            ret_val, image = cam.read()

            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(
                w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            logger.debug('show+')
            fps_time = time.time()
            logger.debug('finished+')

            ''' Runs through this if there are humans in view of the webcam '''
            numHumans = len(humans)
            if numHumans == 0:
                last_num_humans = 0
                x_diffs = np.concatenate(
                    (x_diffs, np.zeros((1, NUM_JOINTS))))
                y_diffs = np.concatenate(
                    (y_diffs, np.zeros((1, NUM_JOINTS))))
                score_avgs = np.concatenate((score_avgs, np.zeros((1, NUM_JOINTS))))
                all_ys = np.concatenate((all_ys, np.zeros((1, NUM_JOINTS))))
                all_xs = np.concatenate((all_xs, np.zeros((1, NUM_JOINTS))))
                bad_data = np.concatenate((bad_data, np.array([0])))
                if use_angles:
                    dist_diffs = np.concatenate((dist_diffs, np.zeros((1,NUM_JOINTS))))
                    angles = np.concatenate((angles, np.zeros((1,NUM_JOINTS))))
                continue

            elif numHumans > 0 and last_num_humans != numHumans:

                temp_scores = np.zeros((numHumans, num_frames, NUM_JOINTS))
                last_num_humans = numHumans
                scores = np.empty((numHumans, NUM_JOINTS))
                lastScores = np.zeros((numHumans, NUM_JOINTS))

                xs = np.empty((numHumans, NUM_JOINTS))
                ys = np.empty((numHumans, NUM_JOINTS))

                x_dists = np.empty((numHumans, NUM_JOINTS))
                y_dists = np.empty((numHumans, NUM_JOINTS))

                last_xs = np.zeros((numHumans, NUM_JOINTS))
                last_ys = np.zeros((numHumans, NUM_JOINTS))
                temp_xs = np.zeros((numHumans, num_frames, NUM_JOINTS))
                temp_ys = np.zeros((numHumans, num_frames, NUM_JOINTS))

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
                        ''' If a certain joint is not recognized or is not in the frame, its x/y cooridinates and scores will be set to 0.0 '''
                        scores[idx][joint_num] = 0.0
                        xs[idx][joint_num] = 0.0
                        ys[idx][joint_num] = 0.0

                right_hand_up = ys[idx][3] > ys[idx][4] if (
                    ys[idx][3] != 0 and ys[idx][4] != 0) else False
                left_hand_up = ys[idx][6] > ys[idx][7] if (
                    ys[idx][6] != 0 and ys[idx][7] != 0) else False

                x_dists[idx] = np.array(
                    [(x-y)**2 for x, y in zip(xs[idx], last_xs[idx])])
                y_dists[idx] = np.array(
                    [(x-y)**2 for x, y in zip(ys[idx], last_ys[idx])])
                
                ''' If the the left arm or right arm is facing downwards, its score will be multiplied by -1 '''
                if (not left_hand_up and not right_hand_up) or (((ys[idx][2] == 0) or (ys[idx][3] == 0)) and ((ys[idx][6] == 0) or (ys[idx][7] == 0))):
                    scores *= -1
                bD = np.array([1]) if True in [(ys[i/NUM_JOINTS][i % NUM_JOINTS] == 0 and last_ys[i/NUM_JOINTS][i % NUM_JOINTS] != 0) or (
                    ys[i/NUM_JOINTS][i % NUM_JOINTS] != 0 and last_ys[i/NUM_JOINTS][i % NUM_JOINTS] == 0) for i in range(numHumans*NUM_JOINTS)] else np.array([0])
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
                        all_xs = np.concatenate((all_xs, xs[idx].reshape(-1, NUM_JOINTS)))
                        all_ys = np.concatenate((all_ys, ys[idx].reshape(-1, NUM_JOINTS)))
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
                    all_xs = np.concatenate((all_xs, xs[idx].reshape(-1, NUM_JOINTS)))
                    all_ys = np.concatenate((all_ys, ys[idx].reshape(-1, NUM_JOINTS)))
                    if use_angles:
                        distTravel = np.array([x + y for x,y in zip(x_travel, y_travel)]).reshape(1, NUM_JOINTS)
                        angle = np.array([math.atan2(y,x) for x, y in zip(x_travel[0], y_travel[0])]).reshape(1,NUM_JOINTS) 
                        dist_diffs = np.concatenate((dist_diffs, distTravel), axis=0)
                        angles = np.concatenate((angles, angle), axis=0)

            iter_num += 1

            last_xs = np.copy(xs)
            last_ys = np.copy(ys)

        f = open(working_dir+data_file+'/video/labels/'+str(vid_num)+'.txt')
        s = f.read()
        f.close()
        tqdm.write("Extracting from Video %d" % vid_num)
        tqdm.write("Saving %d datapoints" % (x_diffs.shape[0]))
        label_name = "%slabel%d.txt" % (label_dir, (max_num + 1))
        data_name = "%sgestureData%d.npz" % (data_dir, (max_num + 1))
        label_file = open(label_name, 'w+')
        label_file.write(s)
        label_file.close()

        # modify features to save features of choice
        # features = [dist_diffs, angles, allXs, allYs, score_avgs] if use_angles else [xDifferences, yDifferences, allXs, allYs, score_avgs]
        features = [x_diffs, y_diffs]
        features.append(score_avgs) # ensure that its last
        data = {}
        for feature_num in range(num_features):
            data[feature_num] = features[feature_num]
        
        np.savez(data_name, data=data, isBadData=bad_data)
        max_num += 1

        if debug:
            print("x Differences", x_diffs)
            print("y Differences", y_diffs)
            print("Score Averages", score_avgs)
            print("Shape", x_diffs.shape)
            print("Bad Data Array", bad_data)
            print("Amount of Good data", collections.Counter(bad_data))

    cv2.destroyAllWindows()
