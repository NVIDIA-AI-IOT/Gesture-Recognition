# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import numpy as np
import tensorflow as tf
import os
import argparse
from Var import Var

class DataLoader:
    def __init__(self, num_frames, use_arm, m_score):
        self.num_frames = num_frames
        self.use_arm = use_arm
        self.m_score = m_score

        self.debug = False
        self.working_dir = os.getcwd() + "/"
        self.v = Var(use_arm)
        self.classes = self.v.get_classes()
        self.num_classes = self.v.get_num_classes()
        self.num_features = self.v.get_num_features()
        self.num_joints = self.v.get_size()

    def setDebug(self):
        self.debug = True

    def npz_to_npy(self, fn, label_fn):
        '''Change input npz file to numpy arrays'''
        data = fn['data'].item()
        new_data = {}

        score_idx = self.num_features - 1
        score = data[score_idx]
        det = fn['isBadData']

        new_labels = np.zeros((0, self.num_classes))
        num_data = data[0].shape[0]
        data_size = data[0].shape[1]
        min_data = num_data * 100 #really big number
        for feature_num in range(self.num_features):
            min_data = data[feature_num].shape[0] if data[feature_num].shape[0] < min_data else min_data
            new_data[feature_num] = np.zeros((0, data_size))
        num_data = min_data

        ''' Read label '''
        f = open(label_fn, 'r')
        s = f.read()

        ''' Check and set one hot encoded value '''
        label = np.zeros(self.num_classes)
        for key, val in self.classes.items():
            if s == val:
                label[key] = 1

        ''' Stack as many inputs as needed in data '''
        labels = np.stack(label for i in range(num_data))

        for idx in range(num_data):
            isBad = det[idx]
            if not isBad:
                for feature_num in range(self.num_features):
                    if self.m_score:
                        if feature_num != score_idx:
                            multiplied = data[feature_num][idx].reshape(1, self.num_joints) * score[idx].reshape(1, self.num_joints)
                            new_data[feature_num] = np.concatenate((new_data[feature_num], multiplied))
                    else:
                        new_data[feature_num] = np.concatenate(
                            (new_data[feature_num], data[feature_num][idx].reshape(1, self.num_joints)))
                new_labels = np.concatenate(
                    (new_labels, labels[idx].reshape(1, self.num_classes)))

        if (self.debug):
            print("SHAPE OF INPUTS: ", new_data[0].shape)

        ''' Returns array with inputs and data quality of inputs '''
        return new_data, new_labels


    def load_npz_data(self):
        ''' Uses npzToNpy to take all the npz files in a data folder and generate numpy arrays of inputs and good/bad data samples '''
        score_idx = self.num_features - 1
        ''' Set filepaths for data/label folders '''
        data_path = self.working_dir + "data/GestureData/%d/gestureData" % self.num_frames
        label_path = self.working_dir + 'data/Labels/%d/label' % self.num_frames
        print list(os.walk(self.working_dir+'data/Labels/%d' % self.num_frames))
        try:
            data_amount = len(
                next(os.walk(self.working_dir+'data/Labels/%d' % self.num_frames))[2])
        except:
            raise Exception("your data cannot be found in %s. If you have data in this folder, the next function (for iterators) is not working properly." % data_path)
        info = {}
        labels = np.zeros((0, self.num_classes))
        for feature_num in range(self.num_features):
            if self.m_score and feature_num == score_idx:
                break
            else:
                info[feature_num] = np.zeros((0, self.num_joints))

        for i in range(data_amount):
            datum, label = self.npz_to_npy(np.load(data_path+str(i+1)+'.npz'), label_path+str(i+1)+'.txt')
            
            for feature_num in range(self.num_features):
                if self.m_score and feature_num == score_idx:
                    break
                else:
                    info[feature_num] = np.vstack((info[feature_num], datum[feature_num]))
            labels = np.vstack((labels, label))

        if (self.debug):
            print("FULL INPUT SHAPE: ", labels.shape)
        return info, labels


    def load_all(self):
        ''' Loads full npz file and properly converts and prunes it for training.'''

        ''' Load all input data from the files '''
        data, out = self.load_npz_data()
        values = np.array(data.values())
        for idx, val in enumerate(values):
            if idx == 0:
                combined = val
            else:
                combined = np.concatenate((combined, val), axis=1)
        p = np.random.permutation(combined.shape[0])
        combined = combined[p]
        out = out[p]
        if(self.debug):
            print("IN SIZE: ", combined.shape)
            print("OUT SIZE: ", out.shape)
        return combined, out
