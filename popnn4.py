# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

''' Imports '''
import tensorflow as tf
from DataLoader import DataLoader
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pr
import os
import argparse
from Var import Var

global debug

parser = argparse.ArgumentParser(description='data loader')
parser.add_argument('--transfer', '-t', dest='transfer', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--dest_ckpt_name', '-d', dest='ckpt_name', type=str)
parser.add_argument('--num_frames', '-f', dest = 'num_frames', type = int, default = 4)
parser.add_argument('--only-arm', '-o', dest='use_arm', action='store_true',
						help="only use arm data")
parser.add_argument('--multiply-by-score', '-m', dest='m_score', action='store_true')


parser.set_defaults(debug=False)
parser.set_defaults(transfer=False)
parser.set_defaults(use_arm=False)
parser.set_defaults(m_score=False)


args = parser.parse_args()
debug = args.debug
ckpt_name = "ckpts/popnn/"  +  args.ckpt_name
transfer = args.transfer
numFrames = args.num_frames
use_arm = args.use_arm
m_score = args.m_score

v = Var(use_arm)
input_size = v.get_size()
num_classes = v.get_num_classes()
popnn_vars = v.get_POPNN()

''' Set working directory '''
working_dir = os.getcwd() + "/"

''' Tensorflow placeholder for inputs '''
x = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32)

''' Log in TensorFlow '''
tf.logging.set_verbosity(tf.logging.INFO)

''' Load data from dataloader '''
loader = DataLoader(numFrames, use_arm, m_score)
inp1, out1 = loader.load_all()

''' Separate data "randomly" using sklearn! '''
trin, valin, trout, valout = ms.train_test_split(inp1, out1, test_size=0.2, random_state=21)
print trin, trin.shape
if (debug):
    print("TRAIN INPUT SHAPE: ", trin.shape)
    print("TRAINING OUTPUT SHAPE: ", trout.shape)
    print("VALIDATION INPUT SHAPE: ", valin.shape)
    print("VALIDATION OUTPUT SHAPE: ", valout.shape)

''' Normalize data using sklearn!!! '''
trin = pr.normalize(trin)
valin = pr.normalize(valin)

''' Define network architecture (3 layer relu) '''
tf.logging.set_verbosity(tf.logging.INFO)
ninput = input_size
nhidden1 = popnn_vars['hidden1']
nhidden2 = popnn_vars['hidden2']
nhidden3 = popnn_vars['hidden3']
# nhidden4 = 8
noutput = popnn_vars['hidden4']
EPOCHS = popnn_vars['num_epochs']
BATCH_SIZE = popnn_vars['batch_size']

weights = {
	'h1': tf.Variable(tf.random_normal([ninput, nhidden1])),
	'h2': tf.Variable(tf.random_normal([nhidden1,nhidden2])),
	'h3': tf.Variable(tf.random_normal([nhidden2, nhidden3])),
	#'h4': tf.Variable(tf.random_normal([nhidden3, nhidden4])),
	'out': tf.Variable(tf.random_normal([nhidden3, noutput]))
}

#b4 remains to keep consistency with best model, 4311
biases = {
	'b1': tf.Variable(tf.random_normal([nhidden1])),
	'b2': tf.Variable(tf.random_normal([nhidden2])),
	'b3': tf.Variable(tf.random_normal([nhidden3])),
    'b4': tf.Variable(tf.random_normal([8])),
	'out': tf.Variable(tf.random_normal([noutput]))
}

keep_prob = tf.placeholder("float")


def network(x, weights, biases, keep_prob):
    '''define network components'''
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
    outlayer = tf.layers.dense(inputs=layer3, units = noutput)
    outlayer = tf.nn.softmax(outlayer, name  ="softmax_tensor")
    return outlayer

''' Define cost, prediction, accuracy, etc. '''
predictions = network(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=popnn_vars['lr']).minimize(cost)
correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accTr = tf.reduce_mean(tf.cast(correct_prediction, "float"))
accVal = tf.reduce_mean(tf.cast(correct_prediction, "float"))

''' Define saver to save network checkpoints (model weights) '''
saver = tf.train.Saver()

''' Define summaries '''
tf_cost_summary = tf.summary.scalar('Cost', cost)
tf_accTr_summary = tf.summary.scalar('Train_Accuracy', accTr)
tf_accVal_summary = tf.summary.scalar('Validation_Accuracy', accVal)

summaries = tf.summary.merge([tf_cost_summary,tf_accTr_summary])
valSummary = tf.summary.merge([tf_accVal_summary])
name = ""
if transfer:
    name = raw_input("Enter the ckpt file to transfer from: ")

''' Run training '''
with tf.Session() as sess:

    ''' Initialize variables '''
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('/tmp/popnn/summaries/first', sess.graph)
    
    ''' Restore previous weights (Comment if this is original running of the network, aka no transfer learning) '''
    if transfer:
        saver.restore(sess, working_dir+'ckpts/popnn/'+name)

    ''' Run network through epochs, and print progress at the end of each loop '''
    for epoch in range(EPOCHS):
        avg_cost = 0.0
        total_batch = int(len(trin) / BATCH_SIZE)
        print total_batch, len(trin), BATCH_SIZE
        x_batches = np.array_split(trin, total_batch)
        y_batches = np.array_split(trout, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _, c, summ = sess.run([optimizer, cost, summaries], 
                            feed_dict={
                                x: batch_x, 
                                y: batch_y, 
                                keep_prob: 0.8
                            })
            
            avg_cost += c / total_batch
            writer.add_summary(summ, (float(epoch)+float(i)/float(total_batch)))
        summVal = sess.run([valSummary], feed_dict={x:valin,y:valout, keep_prob:1.0})
        writer.add_summary(summVal[0], epoch)
        saver.save(sess, working_dir + ckpt_name)

    	print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost), "TrainAcc=", accTr.eval({x: trin, y: trout, keep_prob: 1.0}), \
                "ValAcc=", accVal.eval({x: valin, y: valout, keep_prob: 1.0}))

    print("Finished!")
    
    ''' Save the model checkpoint, change the name based on preference '''
    if (debug):
        print("SAVED!")
    ''' Do final runthrough of network and print final accuracy in train and validation '''
    print("Train Accuracy:", accTr.eval({x: trin, y: trout, keep_prob: 1.0}))
    print("Validation Accuracy:", accVal.eval({x: valin, y: valout, keep_prob: 1.0}))
