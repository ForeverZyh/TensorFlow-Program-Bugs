import tensorflow as tf
import numpy as np

session = tf.Session()
some_test = tf.constant(
    np.random.normal(loc=0.0, scale=1.0, size=(2, 2)).astype(np.float32))
session.run(some_test)
