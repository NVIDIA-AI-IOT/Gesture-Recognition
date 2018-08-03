# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import os
import collections
import argparse
import numpy as np
import sys
sys.path.insert(0, "lstm")
from Var import Var

''' Import argparse flags '''
parser = argparse.ArgumentParser("Data Counter")
parser.add_argument("--numFrames", "-f", dest="numFrames", default=4, type=int, help="required to access data")
args = parser.parse_args()
num_frames = args.numFrames

''' Create label and data directories '''
label_dir = "data/Labels/%d/" % (num_frames)
data_dir = "data/GestureData/%d/" % (num_frames)

v = Var()
classes = v.classes.values()
res = {}
for val in classes:
    res[val] = 0

for label_name in os.listdir(label_dir):
    if os.path.isdir(label_dir + label_name):
        continue
    label_file = open(label_dir + label_name)
    num = label_name.split(".")[0].split("label")[-1]
    data_name = data_dir + "gestureData%s.npz" % num
    num_samples = np.load(data_name)['data'].item()[0].shape[0]
    s = label_file.read()
    for val in classes:
        if s == val:
            res[val] += num_samples

print(res)
