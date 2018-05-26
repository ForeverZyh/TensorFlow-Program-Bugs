import tensorflow as tf
import random

a = tf.placeholder(tf.int32)
zero = tf.constant(0)
x1, y1 = 1, 2
x2, y2 = 3, 4

weightsLayer1 = tf.cond(tf.equal(a, zero), lambda: tf.truncated_normal([x1, y1]),
                        lambda: tf.random_normal([x2, y2]))
with tf.Session() as sess:
    for i in range(10):
        A = random.randint(0, 1)
        print(A, sess.run(weightsLayer1, feed_dict={a: A}))
