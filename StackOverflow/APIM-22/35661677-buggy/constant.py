import tensorflow as tf

session = tf.Session()
some_test = tf.constant(tf.random_normal([2, 2], mean=0.0, stddev=1.0, dtype=tf.float32))
session.run(some_test)
