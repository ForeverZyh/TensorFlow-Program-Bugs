import tensorflow as tf
import numpy as np

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)

session = tf.Session()
some_test = tf.constant(
    np.random.normal(loc=0.0, scale=1.0, size=(2, 2)).astype(np.float32))
print(session.run(some_test))
