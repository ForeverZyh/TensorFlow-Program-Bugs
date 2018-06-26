import tensorflow as tf
import numpy as np

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)

sess = tf.Session()
input = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
idx_list = np.array([0, 2])
output = input[:, idx_list]
print(sess.run(output))
