from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import collections
import argparse
import time
import sys

import logger

tf.logging.set_verbosity(tf.logging.INFO)
global FLAGS

class ModelBuilder(object):

    def __init__(self, xtrain, ytrain, xtest, ytest, outs, params=None):
        self.log = logger.Logging()
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.outs = outs
        if params is not None:
            self.params = params
            self.log.info('parameter values passed in: %s'%params.keys())

    def set_xtrain(self, data):
        self.xtrain = data

    def set_ytrain(self, data):
        self.ytrain = data

    def set_xtest(self, data):
        self.xtest = data

    def set_xtest(self, data):
        self.xtest = data
    
    def set_parameters(self, params):
        self.params = params
        self.log.info('parameter values set: %s'%params.keys())

    def reshape_vector(self, vec, sizes=[-1,28,28,1]):
        return tf.reshape(vec, sizes)

    def cnn_model_fn(self, features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        he_init = tf.contrib.layers.variance_scaling_initializer()
        #input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
        input_layer = features['x']
        print(input_layer.shape)
        #print(input_layer.shape)
        
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=tf.cast(input_layer, tf.float32),filters=42,kernel_size=[5, 5],padding="same",activation=tf.nn.elu, kernel_initializer=he_init)
        print(conv1.shape)
        # Pooling Layer #1
        conv2 = tf.layers.conv2d(inputs=conv1,filters=84,kernel_size=[3, 3],padding="same",activation=tf.nn.elu, kernel_initializer=he_init)
        # Convolutional Layer #2
        print(conv2.shape)
        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        print('pool 1',pool1.shape)
        # Convolutional Layer #3
        conv3 = tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=[5, 5],strides=[1,1],padding="same",activation=tf.nn.elu, kernel_initializer=he_init)
        print(conv3.shape)
        conv4 = tf.layers.conv2d(inputs=conv3,filters=48,kernel_size=[3, 3],strides=[1,1],padding="same",activation=tf.nn.elu, kernel_initializer=he_init)
        print(conv4.shape)
        # Pooling Layer #2
        pool2 = tf.layers.average_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
        print('pool 2',pool2.shape)
        # dense layer
        pool2_flat = tf.reshape(pool2, [-1, 14*14*48])
        print(pool2_flat.shape)
        dense1 = tf.layers.dense(inputs=pool2_flat, units=900, activation=tf.nn.elu, kernel_initializer=he_init)
        print(dense1.shape)
        dense2 = tf.layers.dense(inputs=dense1, units=400, activation=tf.nn.elu, kernel_initializer=he_init)
        print(dense2.shape)
        dropout1 = tf.layers.dropout(inputs=dense2, rate=0.6, training=mode == tf.estimator.ModeKeys.TRAIN)
        print(dropout1.shape)
        #dense3 = tf.layers.dense(inputs=dense2, units=200, activation=tf.nn.elu, kernel_initializer=he_init)
        #print(dense3.shape)
        #dropout = tf.layers.dropout(inputs=dense3, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
        #print(dropout.shape)
        logits = tf.layers.dense(inputs=dropout1, units=self.outs)
        print(logits.shape)
        predictions = { 'classes' : tf.argmax(input=logits, axis=1), 'probabilities' : tf.nn.softmax(logits, name='softmax_tensor')}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.outs)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0003)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = { 'accuracy' : tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data, passing in train and/or test filenames')
    parser.add_argument('-f1', dest='file1', default=None, help='first .pkl file to use as images')
    parser.add_argument('-l1', dest='label1', default=None, help='first y labels file')
    parser.add_argument('-f2', dest='file2', default=None, help='second .pkl file to use as images')
    parser.add_argument('-l2', dest='label2', default=None, help='second y labels file')
    parser.add_argument('-split', dest='split', default=10, type=int, help='percent of data to use as test data')
    parser.add_argument('-batch', dest='batch_size', default=100, type=int, help='NN batch size for stochastic gradient descent')
    parser.add_argument('-steps', dest='steps', default=5000, type=int, help='number of training epochs')
    parser.add_argument('-buckets', dest='buckets', default=3, type=int, help='number of buckets to use')
    parser.add_argument('-upper', dest='upper', default=-1.0, type=float, help='number of buckets to use')
    parser.add_argument('-lower', dest='lower', default=1.0, type=float, help='number of buckets to use')

    FLAGS = parser.parse_args()
    

    #open file 1 from pickle
    with open(FLAGS.file1) as f1:
        images = pickle.load(f1) 
    #may want more than one type of file say tech vs fin to help train
    if FLAGS.file2 is not None:
        with open(FLAGS.file2) as f2:
            images2 = pickle.load(f2)
        images = np.concatenate((np.array(images), np.array(images2)), axis=0)
    else:
        images = np.array(images)
    images = images / 255.0
    #use parameter to set number of training vs test images
    perc_split = int((FLAGS.split/100.00) * len(images))
    #split train test
    train_xvals, test_xvals = images[:perc_split], images[perc_split:]
    yvals = np.genfromtxt(FLAGS.label1, delimiter=',')
    #if second file, format y labels for that too
    if FLAGS.file2 is not None:
        yvals2 = np.genfromtxt(FLAGS.label2, delimiter=',')
        yvals = np.concatenate((yvals, yvals2), axis=0)
    #yvals is a 
    yvals = yvals * 100
    print(images.shape, yvals.shape)
    #yvals = yvals.reshape(-1,1)
    if FLAGS.buckets < 3:
        buckets = np.array([FLAGS.lower, FLAGS.upper])
    else:
        buckets = np.linspace(FLAGS.lower, FLAGS.upper, FLAGS.buckets-1)
    print(buckets)
    #drop each into bucket
    labels = np.digitize(yvals, buckets)
    
    #print(np.max(labels), np.min(labels))
    #labels = np.zeros_like(yvals)
    #labels[yvals>0.0] = 1
    train_yvals, test_yvals = labels[:perc_split], labels[perc_split:]
    mdl = ModelBuilder(train_xvals, train_yvals, test_xvals, test_yvals, outs=len(buckets))
    classifier = tf.estimator.Estimator(model_fn=mdl.cnn_model_fn, model_dir=str(time.time()) +'tmp_convnet_model')
    tensors_to_log = {'probabilities' : 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':train_xvals}, y=train_yvals, batch_size=FLAGS.batch_size, num_epochs=None, shuffle=False)
    classifier.train(input_fn=train_input_fn, steps=FLAGS.steps, hooks=[logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':test_xvals}, y=test_yvals, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(images.shape, yvals.shape)
    print(buckets)
    print(np.max(labels), np.min(labels))

