import tensorflow as tf

zero_tsr = tf.zeros([1, 2])
tf.assign(zero_tsr, [4, 5])
