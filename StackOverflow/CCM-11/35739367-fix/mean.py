import tensorflow as tf
import numpy as np

isTrain = tf.placeholder(tf.bool)
user_input = tf.placeholder(tf.float32)

# ema = tf.train.ExponentialMovingAverage(decay=.5)

with tf.device('/cpu:0'):
    beta = tf.Variable(tf.ones([1]))

    batch_mean = beta.assign(user_input)

    ema = tf.train.ExponentialMovingAverage(decay=0.5)


    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean)


    mean = tf.cond(isTrain,
                   mean_var_with_update,
                   lambda: ema.average(batch_mean))

# ======= End Here ==========
saver = tf.train.Saver()
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

u_input = [[2], [3], [4]]
for u in u_input:
    aa = sess.run([mean], feed_dict={user_input: u, isTrain: True})
    print("Train", aa)

for u in u_input:
    aa = sess.run([mean], feed_dict={user_input: u, isTrain: False})
    print("Test", aa)
