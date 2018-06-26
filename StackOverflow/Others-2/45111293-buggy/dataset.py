## Convolutional Neural Network on tensorflow-gpu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import time
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pandas as pd
import numpy as np
import sklearn as sk
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)
random.seed(20180130)

n_data = 88041

## Cast to 32bit
y = [random.randint(0, 3) for _ in range(n_data)]
X = np.asarray([np.random.normal(0, 1, [28, 28]) for _ in range(n_data)])

## Set NaNs to 10e-6
# X[np.isnan(X)] = 0

## Feature Scaling and split the data into training and test sets
# X_scaled = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

## Convert label to one hot format
y_1Hot_train = tf.one_hot(y_train, 4)
y_1Hot_test = tf.one_hot(y_test, 4)

hm_epochs = 3
n_classes = 4
batch_size = 10
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder(tf.float32, [None, n_chunks, chunk_size])
y = tf.placeholder(tf.float32)


# saver = tf.train.Saver(max_to_keep=10)

def rnn_model(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    ## start recording time ##
    start_time = time.time()

    prediction = rnn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    learning_rate = 0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #    optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            itere = int(X_train.shape[0] / batch_size)
            last = 0
            add = 1

            i = 0
            while i < len(X_train):
                start = i
                end = i + batch_size
                batch_x = np.array(X_train[start:end])
                batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))
                batch_y = np.array(y_1Hot_train.eval()[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size
            sess_end = time.time() - start_time
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss, "Elapsed Time: ", sess_end)
        t_elapsed_time = time.time() - start_time
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: X_test.reshape((-1, n_chunks, chunk_size)), y: y_1Hot_test.eval()}),
              "Processing time:", t_elapsed_time)
        tf.reset_default_graph()


train_neural_network(x)
