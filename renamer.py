# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import os
import sys
import operator
import argparse

''' Initialize argparse flags '''
parser = argparse.ArgumentParser("Data Renamer")
parser.add_argument("--buffer", "-b", dest="buffer", required=True, type=int, help="increment labels and video numbers by this amount")
args = parser.parse_args()

buffer = args,buffer
data_file = "data"
nums = []
files = []

for data in os.listdir("%s/video/videos" % data_file):
    file_num , _ = data.split('.')
    file_num = int(file_num)
    nums.append(file_num)
    files.append(data)

for i in range(len(nums)):
    idx, num = max(enumerate(nums), key=operator.itemgetter(1))
    fn = files[idx]
    nums.pop(idx)
    files.pop(idx)
    os.rename(fn, "%s/video/videos/%d.npz" % (data_file, num + buffer))
    os.rename("%s/video/labels/%d.txt" % (data_file, num), "%s/video/labels/%d.txt" % (data_file, num + buffer))
