import tensorflow as tf
import numpy as np
import random

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)
random.seed(20180130)

n_feature = 10
n_data = 500
data = np.random.normal(0, 0.1, [n_data, n_feature])
label = [[random.randint(0, 1) for _ in range(n_data)]]

sizeOfRow = len(data[0])
x = tf.placeholder("float", shape=[None, sizeOfRow])
y = tf.placeholder("float")


def neuralNetworkModel(x):
    W = tf.Variable(tf.zeros([sizeOfRow, 2]))
    B = tf.Variable(tf.zeros([2]))
    return tf.matmul(x, W) + B


def neuralNetworkTrain(x):
    prediction = neuralNetworkModel(x)
    # using softmax function, normalize values to range(0,1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimize = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, loss = sess.run([optimize, cost], feed_dict={x: data, y: label})
        print("loss %g"% loss)


neuralNetworkTrain(x)
