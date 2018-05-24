import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

dtype = tf.float16
shape = (4, 16, 16, 3)

inpt = tf.placeholder(dtype, shape, name='input')
net = slim.conv2d(inpt, 16, [3, 3], scope='conv',
                  weights_initializer=tf.zeros_initializer(),
                  # normalizer_fn=slim.batch_norm
                  )
loss = tf.reduce_mean(net)
opt = tf.train.AdamOptimizer(1e-3)
train_op = slim.learning.create_train_op(loss, opt)

val = np.zeros(shape)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2):
        print(sess.run(train_op, feed_dict={inpt: val}))
