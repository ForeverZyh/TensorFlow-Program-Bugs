import tensorflow as tf
import numpy as np
import random

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)
random.seed(20180130)

niter = 1000

nclasses = 15
nfeature = 250
itrain = 25000
itest = 1000
lrate = 0.01
X = [np.random.normal(0, 1, nfeature) for _ in range(itrain + itest)]
X = np.asarray(X)
Y = []
for i in range(itrain + itest):
    Y.append(np.zeros(nclasses))
    Y[-1][random.randint(0, nclasses - 1)] = 1
Y = np.asarray(Y)

x = tf.placeholder(tf.float32, [None, nfeature])
y_ = tf.placeholder(tf.float32, [None, nclasses])
W = tf.Variable(tf.random_normal([nfeature, nclasses]))
B = tf.Variable(tf.random_normal([nclasses]))
y = tf.matmul(x, W) + B
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(lrate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(niter):
    # Train
    _, c = sess.run([train_step, cost], feed_dict={x: X[:itrain], y_: Y[:itrain]})
    print(sess.run(accuracy, feed_dict={x: X[:itrain], y_: Y[:itrain]}))

    # EDIT: Calculate test error
    print(sess.run(accuracy, feed_dict={x: X[itrain:itrain + itest, :], y_: Y[itrain:itrain + itest, :]}))

sess.close()
