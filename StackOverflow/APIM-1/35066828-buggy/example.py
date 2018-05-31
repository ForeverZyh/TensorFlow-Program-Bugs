import tensorflow as tf

x = tf.Variable(tf.random_normal([2, 2], stddev=0.35))
init_op = tf.initialize_all_variables()
sess = tf.Session(init_op)
