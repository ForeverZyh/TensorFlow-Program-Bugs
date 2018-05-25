import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

n_data = 100
X = [np.random.uniform(0, 255, 900) for _ in range(n_data)]
X = np.asarray(X, dtype=np.float32)
y = []
for _ in range(n_data):
    y.append(np.zeros([62]))
    y[-1][random.randint(0, 61)] = 1

# normalise the features
X = (X - 255 / 2) / 255

y = np.float32(y)
X = np.float32(X)
Xtr, Xte, Ytr, Yte = train_test_split(X, y, train_size=0.7)

batch_size = 10

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 900])
y_ = tf.placeholder(tf.float32, shape=[None, 62])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 30, 30, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 62])
b_fc2 = bias_variable([62])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    offset = (i * batch_size) % (Ytr.shape[0] - batch_size)
    batch_x = Xtr[offset:(offset + batch_size), :]
    batch_y = Ytr[offset:(offset + batch_size), :]
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(
        feed_dict={x: Xtr[offset:(offset + batch_size), :], y_: Ytr[offset:(offset + batch_size), :], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: Xte, y_: Yte, keep_prob: 1.0}))
