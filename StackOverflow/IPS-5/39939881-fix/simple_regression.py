import tensorflow as tf
import numpy as np
import random

_w = random.uniform(-8, 8)
_b = random.uniform(-8, 8)
print(_w, _b)
size = 8
X = [random.uniform(-8, 8) for _ in range(size)]
Y = [_w * x + _b + random.uniform(-1, 1) for x in X]
x = tf.placeholder(tf.float32, [None])
y = tf.placeholder(tf.float32, [None])
W = tf.Variable(tf.zeros([1]))
B = tf.Variable(tf.zeros([1]))
y_ = tf.multiply(x, W) + B
loss = tf.reduce_mean(tf.square(y_ - y))
optimizer = tf.train.AdamOptimizer(1e-1).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(X, Y)
for i in range(100):
    _, los, w, b = sess.run([optimizer, loss, W, B], feed_dict={x: X, y: Y})
    print("loss %g, w %g, b %g" % (los, w, b))
