# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
class Var():
    '''Parameter Centralization Class which most scripts reference'''

    def __init__(self, use_arm=False):
        self.input_size = 8 if use_arm else 18
        # , 3: "Y Pose", 4: "Dab", 5: "Sorry"}
        self.classes = {0: "None", 1: "Wave"}#, 2: "X Pose"}
        self.num_classes = len(self.classes)
        self.features = ["xDist", "yDist", "ScoreAvgs"]
        self.features = ["Dist", "ScoreAvgs"]
        self.num_features = len(self.features)
        self.rate = 20

        self.lstm_vars = {
            "num_layers": 2,
            "hidden_size": 128,
            "dropout": 0.2,
            "lr": 0.005,
            "seq_len": 1,
            "batch_size": 256,
            "num_epochs": 1000,
            "print_every": 25,
            "plot_every": 25,
            "hidden1": 32,
            "hidden2": 8,
            "hidden3": self.num_classes
        }

        self.popnn_vars = {
            "lr": 0.001,
            "batch_size": 4,
            "num_epochs": 1200,
            "print_every": 5,
            "plot_every": 25,
            "dropout": 0.2,
            "hidden1": 60,
            "hidden2": 30,
            "hidden3": 18,
            "hidden4": self.num_classes
        }

    def get_rate(self):
        '''returns desired FPS'''
        return self.rate
    def get_size(self):
        '''Get the input size of data/Number of joints'''
        return self.input_size

    def get_LSTM(self):
        '''Get full list of LSTM variables'''
        return self.lstm_vars

    def get_POPNN(self):
        '''Get full list of POPNN variables'''
        return self.popnn_vars

    def get_classes(self):
        '''Get classification classes'''
        return self.classes

    def get_num_features(self):
        '''Get number of features of data collection. 
        For example, using x position, y position, and score would be 3 features.'''
        return self.num_features

    def get_num_classes(self):
        '''Get number of classification classes'''
        return self.num_classes
