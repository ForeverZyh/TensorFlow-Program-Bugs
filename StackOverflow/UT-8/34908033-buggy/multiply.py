import tensorflow as tf
import numpy as np

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)

data = np.array([0.1, 0.2])
x = tf.placeholder("float", shape=[2])
T1 = tf.Variable(tf.ones([2, 2]))
l1 = tf.matmul(T1, x)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(l1, feed_dict={x: data}))
