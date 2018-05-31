import tensorflow as tf

x = tf.placeholder(tf.float32, [3, 4])
y = tf.placeholder(tf.int32, [4, 5])
z = tf.matmul(x, y)
