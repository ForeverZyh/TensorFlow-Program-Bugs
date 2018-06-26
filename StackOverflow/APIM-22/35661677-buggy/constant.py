import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

session = tf.Session()
some_test = tf.constant(tf.random_normal([2, 2], mean=0.0, stddev=1.0, dtype=tf.float32))
print(session.run(some_test))
