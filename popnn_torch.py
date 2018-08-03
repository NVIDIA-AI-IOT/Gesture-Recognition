# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import argparse
import random
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import operator
import os
import sys
import signal
from Var import Var
from DataLoader import DataLoader
import sklearn.model_selection as ms
import sklearn.preprocessing as pr

plt.switch_backend('agg')
use_cuda = torch.cuda.is_available()

v = Var()
num_classes = v.get_num_classes()
popnn_vars = v.get_POPNN()

class Model(nn.Module):
	''' FC Neural Network '''

	def __init__(self, input_size, num_features, dropout=0):
		super(Model, self).__init__()

		self.input_size = input_size
		self.dropout = dropout
		self.num_features = num_features # using x and y is 2 features, only dist is 1, etc.
		self.hidden1 = popnn_vars["hidden1"]
		self.hidden2 = popnn_vars["hidden2"]
		self.hidden3 = popnn_vars["hidden3"]
		self.hidden4 = popnn_vars["hidden4"]

		self.fc1 = nn.Linear(self.num_features * self.input_size, self.hidden1)
		self.drop = nn.Dropout(p=dropout)
		self.fc2 = nn.Linear(self.hidden1, self.hidden2)
		self.fc3 = nn.Linear(self.hidden2, self.hidden3)
		self.fc4 = nn.Linear(self.hidden3, self.hidden4)

	def forward(self, input):
		''' Forward pass through network '''
		output = input.view(-1, self.num_features * self.input_size)
		output = self.drop(F.relu(self.fc1(output)))
		output = self.drop(F.relu(self.fc2(output)))
		output = self.drop(F.relu(self.fc3(output)))
		output = self.drop(F.relu(self.fc4(output)))
		output = F.logsigmoid(output)

		return output

def accuracy(output, label):
	''' Check if network output is equal to the corresponding label '''
	max_idx, val = max(enumerate(output[0]), key=operator.itemgetter(1))
	out = torch.zeros(1, num_classes).cuda() if use_cuda else torch.zeros(1, num_classes)
	out[0][max_idx] = 1

	if torch.eq(out.float(), label).byte().all():
		return 1
	else:
		return 0


def train(model, optim, criterion, datum, label):
	''' Modify weights based off cost from one datapoint '''
	optim.zero_grad()
	output = model(datum)
	output = output.view(1, num_classes)
	is_correct = accuracy(output, label)
	loss = criterion(output, label)
	loss.backward()
	optim.step()

	return loss.item(), is_correct


def test_accuracy(model, x_test, y_test):
	''' Accuracy of Model on test data '''
	num_correct = 0
	for test, label in zip(x_test, y_test):
		output = model(test.view(1, model.num_features, model.input_size))
		is_correct = accuracy(output, label)
		num_correct += is_correct

	return num_correct / float(x_test.size()[0])


def create_plot(loss, train_acc, test_acc, num_epochs, plot_every, fn):
	''' Creates graph of loss, training accuracy, and test accuracy '''
	plt.figure()
	fig, ax = plt.subplots()
	x_ticks = range(plot_every, num_epochs + 1, plot_every)
	y_ticks = np.arange(0, 1.1, 0.1)
	plt.subplot(111)
	plt.plot(x_ticks, loss, label="Average Loss")
	plt.plot(x_ticks, train_acc, label="Training Accuracy")
	plt.plot(x_ticks, test_acc, label="Validation Accuracy")
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0.)
	plt.xticks(x_ticks)
	plt.yticks(y_ticks)
	plt.axis([0, num_epochs, 0, 1])
	plt.ylabel("Average Loss")
	plt.xlabel("Epoch")
	
	''' Save graph '''
	plt.savefig(fn)


def train_iters(model, x_train, y_train, x_test, y_test, fn, lr=popnn_vars['lr'], batch_size=popnn_vars["batch_size"], num_epochs=popnn_vars["num_epochs"], print_every=popnn_vars['print_every'], plot_every=popnn_vars['plot_every']):
	''' Trains neural net for numEpochs iters '''
	num = fn.split('/')[-1].split('.')[0].split('popnn')[-1]
	plot_fn = "graphs/popnn/lossAvg%s.png" % num

	def sigint_handler(sig, iteration):
		''' Handles Ctrl + C. Save the data into npz files. Data inputted into thenetwork '''
		torch.save(model.state_dict(), fn)
		create_plot(plot_loss_avgs, train_accuracies,
				   test_accuracies, num_epochs, plot_every, plot_fn)
		print("Saving model and Exiting")
		sys.exit(0)

	''' Initialize sigint handler '''
	signal.signal(signal.SIGINT, sigint_handler)

	plot_loss_avgs = []
	epochs = []
	train_accuracies = []
	test_accuracies = []
	loss_total = 0
	plot_loss_total = 0
	num_correct = 0
	plot_correct = 0

	optimizer = optim.Adam(model.parameters(), lr=lr)
	if gamma != None:
		scheduler = StepLR(optimizer, step_size=300, gamma=gamma)
	criterion = nn.BCEWithLogitsLoss()

	y_train = torch.from_numpy(y_train).float().cuda(
	) if use_cuda else torch.from_numpy(y_train).float()
	x_train = torch.from_numpy(x_train).float().cuda(
	) if use_cuda else torch.from_numpy(x_train).float()
	y_test = torch.from_numpy(y_test).float().cuda(
	) if use_cuda else torch.from_numpy(y_test).float()
	x_test = torch.from_numpy(x_test).float().cuda(
	) if use_cuda else torch.from_numpy(x_test).float()

	for current_epoch in tqdm(range(num_epochs)):
		if gamma != None:
			scheduler.step()
		for i in range(batch_size):
			frame_num = random.randint(0, x_train.size()[0] - 1)
			frame = x_train[frame_num].view(1, model.num_features, model.input_size)
			label = y_train[frame_num].view(1, num_classes)
			loss, is_correct = train(
				model, optimizer, criterion, frame, label)

			num_correct += is_correct
			loss_total += loss
			plot_correct += is_correct  # Make a copy of numCorrect for plot_every
			plot_loss_total += loss
		
		if (current_epoch + 1) % print_every == 0:
			avg_loss = loss_total / (print_every * batch_size)
			train_acc = num_correct / float(print_every * batch_size)
			test_acc = test_accuracy(model, x_test, y_test)
			tqdm.write("[Epoch %d/%d] Avg Loss: %f, Training Acc: %f, Validation Acc: %f" %
					   (current_epoch + 1, num_epochs, avg_loss, train_acc, test_acc))
			loss_total = 0
			num_correct = 0

		if (current_epoch + 1) % plot_every == 0:
			plot_test_acc = test_accuracy(model, x_test, y_test)
			plot_train_acc = plot_correct / float(plot_every * batch_size)
			train_accuracies.append(plot_train_acc)
			test_accuracies.append(plot_test_acc)
			avg_loss = plot_loss_total / (plot_every * batch_size)
			plot_loss_avgs.append(avg_loss)
			epochs.append(current_epoch + 1)
			plot_correct = 0
			plot_loss_total = 0

	create_plot(plot_loss_avgs, train_accuracies,
			   test_accuracies, num_epochs, plot_every, plot_fn)	

if __name__ == "__main__":
	''' Argparse Flags '''
	parser = argparse.ArgumentParser(description='Fully Connected Feed Forward Net on tf-openpose data')
	parser.add_argument("--transfer", "-t", dest="transfer", action="store_true")
	parser.add_argument('--debug', dest='debug', action='store_true')
	parser.add_argument("--ckpt-fn", "-c", dest="ckpt_fn",
						type=str, default="popnn000.ckpt")
	parser.add_argument("--save-fn", "-s", dest="save_fn",
						type=str, default="popnn000.ckpt")
	parser.add_argument("--learning-rate", "-lr",
						dest="learning_rate", type=float, default=0.000035)
	parser.add_argument("--num-epochs", "-e", dest="num_epochs",
						type=int, default=1000, help='number of training iterations')
	parser.add_argument("--batch-size", "-b", dest="batch_size", type=int,
						default=256, help='number of training samples per epoch')
	parser.add_argument('--num-frames', '-f', dest='num_frames', type=int, default=4,
						help='number of consecutive frames where distances are accumulated')
	parser.add_argument('--lr-decay-rate', '-d', dest='gamma', type=float, default=None)
	parser.add_argument('--only-arm', '-o', dest='use_arm', action='store_true',
						help="only use arm data")
	parser.add_argument('--multiply-by-score', '-m', dest='m_score', action='store_true')	

	''' Set argparse defaults '''
	parser.set_defaults(use_arm=False)
	parser.set_defaults(transfer=False)
	parser.set_defaults(debug=False)
	parser.set_defaults(m_score=False)

	''' Set variables to argparse arguments '''
	args = parser.parse_args()
	transfer = args.transfer
	debug = args.debug
	gamma = args.gamma
	use_arm = args.use_arm
	m_score = m_score

	ckpt_fn = "lstmpts/popnn/%s" % args.ckpt_fn
	fn = "lstmpts/popnn/%s" % args.save_fn

	v = Var(use_arm)
	input_size = v.get_size()
	num_features = v.get_num_features()
	dropout = popnn_vars['dropout']

	model = Model(input_size=input_size, num_features=num_features, dropout=dropout)
	model = model.cuda() if use_cuda else model

	if transfer:
		model.load_state_dict(torch.load(ckpt_fn))
		print("Transfer Learning")
	else:
		print("Not Transfer Learning")

	loader = DataLoader(args.num_frames, use_arm, m_score)
	inp1, out1 = loader.load_all()
	x_train, x_test, y_train, y_test = ms.train_test_split(
		inp1, out1, test_size=0.15, random_state=23)
	x_train = pr.normalize(x_train)
	y_train = pr.normalize(y_train)
	x_test = pr.normalize(x_test)
	y_test = pr.normalize(y_test)
	train_iters(model, x_train, y_train, x_test, y_test, fn)
	torch.save(model.state_dict(), fn)