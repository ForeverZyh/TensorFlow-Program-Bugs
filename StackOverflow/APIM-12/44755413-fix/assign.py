import tensorflow as tf

zero_tsr = tf.Variable([0, 0])
tf.assign(zero_tsr, [4, 5])
