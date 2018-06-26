import tensorflow as tf
import numpy as np

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)

x = tf.placeholder(tf.float32, [None])
x.set_shape([1028178])
y = tf.identity(x)
y.set_shape([478, 717, 3])
X = np.random.normal(0, 0.1, 1028178)

sess = tf.Session()
Y = sess.run(y, feed_dict={x: X})
print(Y)