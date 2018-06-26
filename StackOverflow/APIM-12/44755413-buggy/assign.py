import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

zero_tsr = tf.zeros([1, 2])
op = tf.assign(zero_tsr, [4, 5])
sess = tf.Session()
_ = sess.run(op)
print(sess.run(zero_tsr))