import tensorflow as tf
import numpy as np

# tf.version == 0.7.0
# 0.7.0 version tensorflow even does not have the attribute '__version__'
np.random.seed(20180130)
# Make 100 phony data points in NumPy.
x_data = np.random.rand(2, 100)  # Random input
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# Construct a linear model.
b = tf.Variable(tf.zeros([1], dtype=np.float64))
W = tf.Variable(tf.random_uniform([1, 2], minval=-1.0, maxval=1.0, dtype=np.float64))
y = tf.matmul(W, x_data) + b

# Minimize the squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))


class DoubleGDOptimizer(tf.train.GradientDescentOptimizer):
    def _valid_dtypes(self):
        return set([tf.float32, tf.float64])


optimizer = DoubleGDOptimizer(tf.constant(0.5, tf.float64))
train = optimizer.minimize(loss)

# For initializing the variables.
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the plane.
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
