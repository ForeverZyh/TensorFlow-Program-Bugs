import tensorflow as tf
import numpy as np

sess = tf.Session()
input = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
idx_list = np.array([0, 2])
output = input[:, idx_list]
print(sess.run(output))
