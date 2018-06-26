import tensorflow as tf

assert tf.__version__ == "1.0.0"
tf.set_random_seed(20180130)

v_1 = tf.Variable(1)
v_2 = tf.Variable(v_1.initialized_value())

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(v_2))
