import tensorflow as tf
import numpy as np

T = 100
noise = 10 * np.random.random(size=T).astype(np.float32)
x = np.array([np.arange(T), np.ones(T)]).astype(np.float32)
w = np.array([[2, 4]]).astype(np.float32)
y = w.dot(x) + noise
w.dot(x)
X = tf.placeholder(tf.float32, [2, T], name="X")
W = tf.Variable(tf.ones([1, 2]), name="W")
Yhat = tf.matmul(W, X)
Y = tf.placeholder(tf.float32, [1, T], name="Y")
MSE = (1. / (2 * T)) * tf.reduce_sum(tf.pow(Y - Yhat, 2))
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
trainer = tf.train.GradientDescentOptimizer(0.0005).minimize(MSE)
for _ in range(100):
    sess.run(trainer, feed_dict={X: x, Y: y})
    print(sess.run(MSE, feed_dict={X: x, Y: y}))
