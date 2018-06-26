import tensorflow as tf
import random
import numpy as np

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)
random.seed(20180130)


class Play:
    def __init__(self, input_data, labels):
        # the input shape is (batch_size, input_size)
        input_size = tf.shape(input_data)[1]

        # labels in one-hot format have shape (batch_size, num_classes)
        num_classes = tf.shape(labels)[1]

        stddev = 1.0 / tf.cast(input_size, tf.float32)

        w_shape = tf.stack([input_size, num_classes])
        normal_dist = tf.truncated_normal(w_shape, stddev=stddev, name='normaldist')
        self.w = tf.Variable(normal_dist, name='weights')
        print(self.w)


n_feature = 10
n_classes = 7
play = Play(tf.placeholder(tf.float32, [None, n_feature]), tf.placeholder(tf.int32, [None, n_classes]))
