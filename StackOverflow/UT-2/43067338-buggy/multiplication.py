import numpy as np
import tensorflow as tf

M = 5
N = 2
T = 3
h = 2
s = 3
A_np = np.random.randn(M, h)
C_np = np.random.randn(s, T)
B_np = np.random.randn(h, N, s)

A_tf = tf.Variable(A_np)
C_tf = tf.Variable(C_np)
B_tf = tf.Variable(B_np)

# Tensorflow
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(A_tf))
    p = tf.matmul(A_tf, B_tf)
    sess.run(p)
