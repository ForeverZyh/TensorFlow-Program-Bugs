import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None])
x.set_shape([1028178])
y = tf.identity(x)
y = tf.reshape(y, [478, 717, 3])
X = np.random.normal(0, 0.1, 1028178)

sess = tf.Session()
Y = sess.run(y, feed_dict={x: X})
