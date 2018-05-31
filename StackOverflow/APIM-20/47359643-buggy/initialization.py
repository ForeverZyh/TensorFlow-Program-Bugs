import numpy as np
import tensorflow as tf

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    in_size = 100
    h1_size = 10

    x = tf.placeholder(tf.float32, (None, in_size))
    w = tf.Variable(tf.random_normal([in_size, h1_size]))
    b = tf.Variable(tf.ones([h1_size]))

    xw = tf.matmul(x, w)
    z = tf.add(xw, b)

    a = tf.nn.relu(z)

    yhat = sess.run(a, feed_dict={x: np.random.random([100000, in_size])})
