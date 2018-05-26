import tensorflow as tf
import random

a = tf.placeholder(tf.int32)
x1, y1 = 1, 2
x2, y2 = 3, 4

if a == 0:
    weightsLayer1 = tf.Variable(tf.truncated_normal([x1, y1]))
else:
    weightsLayer1 = tf.Variable(tf.random_normal([x2, y2]))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        A = random.randint(0, 1)
        print(A, sess.run(weightsLayer1, feed_dict={a: A}))
